import torch
import torch.nn.functional as F
from .losses import *
from .save_and_log import *
from .stft_loss import *
from .augmentation import shift_phase
import numpy as np


def ensure_3d(x):
    """
    辅助函数：确保 Tensor 是 3D [Batch, Channels, Time]。
    """
    if x.dim() == 2:  # [Batch, Time] -> 增加 Channel 维
        return x.unsqueeze(1)
    elif x.dim() == 4:  # [Batch, 1, 1, Time] -> 压缩多余维
        return x.squeeze(1)
    return x  # 已经是 [Batch, Channels, Time]


def runEpoch(loader, config, netG, netD, optG, optD, device, epoch,
             steps, writer, gen_autoclip, disc_autoclip, adv_autobalancer, optD_spec=None, netD_spec=None,
             validation=False):
    costs = [[0, 0, 0, 0, 0, 0, 0]]
    gan_loss_calculator = GANLoss(netD).to(device)
    # 初始化 output_aud 为空列表
    output_aud = [np.array([]), np.array([]), np.array([])]
    validation_song_seconds = 0
    multi_scale_mel_loss = MultiResolutionSTFTLoss().to(device)

    if validation:
        netG.eval()
        netD.eval()
    else:
        netG.train()
        netD.train()

    # DataLoader 返回 4 个值: (dirty, clean, dirty_copy, filename)
    for iterno, data in enumerate(loader):
        # 解包数据
        x_t = data[0].to(device)  # Dirty Input
        y_t = data[1].to(device)  # Clean Target
        # data[2] 是 mix 副本
        # data[3] 是 filename (tuple or list)
        filenames = data[3] if len(data) > 3 else []

        # 维度修正
        x_t_0 = ensure_3d(x_t).float()
        x_t_1 = ensure_3d(y_t).float()

        # 增加 padding (兼容模型)
        inp = F.pad(x_t_0, (4000, 4000), "constant", 0)

        if not validation:
            optG.zero_grad()

        # 生成预测
        try:
            x_pred_t = netG(inp, x_t_0)
        except TypeError:
            x_pred_t = netG(inp)

        if isinstance(x_pred_t, (list, tuple)):
            x_pred_t = x_pred_t[0]

        x_pred_t = ensure_3d(x_pred_t)

        # 计算 Loss
        wav_loss = F.l1_loss(x_t_1, x_pred_t)

        if not config.mono:
            x_pred_t_mono = (x_pred_t[:, 0, :] + x_pred_t[:, 1, :])
            x_pred_t_mono /= torch.max(torch.abs(x_pred_t_mono))
            x_t_1_mono = (x_t_1[:, 0, :] + x_t_1[:, 1, :])
            x_t_1_mono /= torch.max(torch.abs(x_t_1_mono))
            mel_reconstruction_loss = mel_spec_loss(x_pred_t_mono, x_t_1_mono)
        else:
            # Mel Loss 需要 [B, T]
            spec_convergence, log_mel_loss = multi_scale_mel_loss(x_pred_t.squeeze(1), x_t_1.squeeze(1))
            if config.use_both_reconstruction:
                mel_reconstruction_loss = log_mel_loss + spec_convergence
            else:
                mel_reconstruction_loss = log_mel_loss

        # SDR Loss
        sdr = SISDRLoss()
        if config.mono:
            # SDR 期望 [B, T, 1]
            sdr_loss = sdr(x_pred_t.permute(0, 2, 1), x_t_1.permute(0, 2, 1))
        else:
            sdr_loss = sdr(x_pred_t_mono.unsqueeze(2), x_t_1_mono.unsqueeze(2))

        # 判别器训练
        fake = x_pred_t.detach()
        real = x_t_1

        if config.augment:
            factor = np.random.uniform(-np.pi, np.pi)
            fake_aug = shift_phase(fake.squeeze(1), factor).unsqueeze(1)
            real_aug = shift_phase(real.squeeze(1), factor).unsqueeze(1)
            loss_D = gan_loss_calculator.discriminator_loss(fake_aug, real_aug)
        else:
            loss_D = gan_loss_calculator.discriminator_loss(fake, real)

        if not validation and epoch >= config.pretrain_epoch:
            netD.zero_grad()
            loss_D.backward()
            disc_autoclip(netD)
            optD.step()

        # 生成器训练
        loss_G, loss_feat = gan_loss_calculator.generator_loss(x_pred_t, x_t_1)

        if not validation:
            if epoch >= config.pretrain_epoch:
                if config.adv_only:
                    total_generator_loss = sum(adv_autobalancer(loss_G, loss_feat))
                else:
                    total_generator_loss = sum(adv_autobalancer(loss_G, loss_feat, mel_reconstruction_loss))
                total_generator_loss.backward()
            else:
                mel_reconstruction_loss.backward()

            _, gen_grad_norm = gen_autoclip(netG)
            optG.step()

            costs = [
                [loss_D.item(), loss_G.item(), loss_feat.item(),
                 mel_reconstruction_loss.item(),
                 -1 * sdr_loss.item(), wav_loss.item(), gen_grad_norm]]
        else:
            curr_costs = [loss_D.item(), loss_G.item(), loss_feat.item(),
                          mel_reconstruction_loss.item(),
                          -1 * sdr_loss.item(), wav_loss.item(), 0]
            for i in range(len(costs[0])):
                costs[0][i] += curr_costs[i]

        # 日志记录
        if not validation:
            basic_logs(costs, writer, steps, epoch, iterno)
        else:
            validation_writer(epoch, steps)
        steps += 1

        # --- 验证集音频记录逻辑 (核心修复) ---
        if validation and len(filenames) > 0:
            current_file = filenames[0]

            # 判断是否需要记录：如果没指定特定歌曲，或者匹配到了特定歌曲
            should_log = (config.validation_song is None) or (current_file == config.validation_song)

            # 限制记录时长 (防止数据过大)
            if should_log and validation_song_seconds < 5:
                # 获取第一个样本并转为 numpy
                dirty_np = x_t_0[0].squeeze().cpu().numpy()
                clean_np = x_t_1[0].squeeze().cpu().numpy()
                pred_np = x_pred_t[0].squeeze().cpu().numpy()

                # 拼接到 output_aud 中
                output_aud[0] = np.concatenate((output_aud[0], dirty_np))
                output_aud[1] = np.concatenate((output_aud[1], clean_np))
                output_aud[2] = np.concatenate((output_aud[2], pred_np))

                validation_song_seconds += 1

    if validation:
        for i in range(len(costs[0])):
            costs[0][i] /= (iterno + 1)

        # --- 兜底逻辑：防止 output_aud 为空导致 Crash ---
        if len(output_aud[0]) == 0:
            print(
                f"Warning: No validation audio logged. Appending silence to prevent crash. (Check validation_song config: {config.validation_song})")
            # 创建 1秒静音
            silence = np.zeros(config.sample_rate)
            output_aud = [silence, silence, silence]

    return steps, costs, output_aud