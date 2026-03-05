# import torch
# import torch.nn.functional as F
# from .losses import *
# from .save_and_log import *
# from .stft_loss import *
# from .augmentation import shift_phase
# import numpy as np
#
#
# def ensure_3d(x):
#     """
#     辅助函数：确保 Tensor 是 3D [Batch, Channels, Time]。
#     """
#     if x.dim() == 2:  # [Batch, Time] -> 增加 Channel 维
#         return x.unsqueeze(1)
#     elif x.dim() == 4:  # [Batch, 1, 1, Time] -> 压缩多余维
#         return x.squeeze(1)
#     return x  # 已经是 [Batch, Channels, Time]
#
#
# def runEpoch(loader, config, netG, netD, optG, optD, device, epoch,
#              steps, writer, gen_autoclip, disc_autoclip, adv_autobalancer, optD_spec=None, netD_spec=None,
#              validation=False):
#     costs = [[0, 0, 0, 0, 0, 0, 0]]
#     gan_loss_calculator = GANLoss(netD).to(device)
#     # 初始化 output_aud 为空列表
#     output_aud = [np.array([]), np.array([]), np.array([])]
#     validation_song_seconds = 0
#     multi_scale_mel_loss = MultiResolutionSTFTLoss().to(device)
#
#     if validation:
#         netG.eval()
#         netD.eval()
#     else:
#         netG.train()
#         netD.train()
#
#     # 初始化 iterno
#     iterno = -1
#
#     for iterno, data in enumerate(loader):
#         # 解包数据
#         x_t = data[0].to(device)  # Dirty Input
#         y_t = data[1].to(device)  # Clean Target
#         filenames = data[3] if len(data) > 3 else []
#
#         # 1. 维度修正 [Batch, Channels, Time]
#         x_t_0 = ensure_3d(x_t).float()
#         x_t_1 = ensure_3d(y_t).float()
#
#         # 2. 增加 padding
#         inp = F.pad(x_t_0, (4000, 4000), "constant", 0)
#
#         if not validation:
#             optG.zero_grad()
#
#         # 3. 生成预测
#         try:
#             # 尝试传入条件 (有些 Demucs 实现需要)
#             x_pred_t = netG(inp, x_t_0)
#         except TypeError:
#             x_pred_t = netG(inp)
#
#         if isinstance(x_pred_t, (list, tuple)):
#             x_pred_t = x_pred_t[0]
#
#         x_pred_t = ensure_3d(x_pred_t)
#
#         # 4. 计算 Loss
#         wav_loss = F.l1_loss(x_t_1, x_pred_t)
#
#         # [立体声适配逻辑]
#         if not config.mono and x_pred_t.shape[1] > 1:
#             # 如果是立体声，混合成单声道来计算 Mel Loss (节省显存且稳定)
#             # 或者你可以分别计算再平均，这里沿用 MSG 原版逻辑：Downmix
#             x_pred_t_mono = torch.mean(x_pred_t, dim=1)  # [B, T]
#             x_t_1_mono = torch.mean(x_t_1, dim=1)  # [B, T]
#
#             # 归一化防止幅度溢出影响 Loss
#             if torch.max(torch.abs(x_t_1_mono)) > 0:
#                 x_t_1_mono /= torch.max(torch.abs(x_t_1_mono))
#             if torch.max(torch.abs(x_pred_t_mono)) > 0:
#                 x_pred_t_mono /= torch.max(torch.abs(x_pred_t_mono))
#
#             # 计算 Mel Loss
#             spec_convergence, log_mel_loss = multi_scale_mel_loss(x_pred_t_mono, x_t_1_mono)
#             mel_reconstruction_loss = log_mel_loss + spec_convergence
#         else:
#             # 单声道直接计算
#             spec_convergence, log_mel_loss = multi_scale_mel_loss(x_pred_t.squeeze(1), x_t_1.squeeze(1))
#             if config.use_both_reconstruction:
#                 mel_reconstruction_loss = log_mel_loss + spec_convergence
#             else:
#                 mel_reconstruction_loss = log_mel_loss
#
#         # SDR Loss
#         sdr = SISDRLoss()
#         if config.mono:
#             sdr_loss = sdr(x_pred_t.permute(0, 2, 1), x_t_1.permute(0, 2, 1))
#         else:
#             # 立体声 SDR: 需要 [Batch, Channels, Time] -> [Batch, Time, Channels] ?
#             # SISDRLoss 通常期望 (B, C, T) 或者 (B, T, C) 取决于实现
#             # 这里保持原代码逻辑，通常 sdr 库期望 (Batch, n_sources, Time)
#             # 如果我们把它当做 source 维度
#             sdr_loss = sdr(x_pred_t.unsqueeze(2), x_t_1.unsqueeze(2))
#             # 注意：如果显存不够，这一步可以 comment 掉，SDR 仅用于日志不参与梯度
#
#         # 5. 判别器训练
#         fake = x_pred_t.detach()
#         real = x_t_1
#
#         loss_D = gan_loss_calculator.discriminator_loss(fake, real)
#
#         if not validation and epoch >= config.pretrain_epoch:
#             netD.zero_grad()
#             loss_D.backward()
#             disc_autoclip(netD)
#             optD.step()
#
#         # 6. 生成器训练
#         loss_G, loss_feat = gan_loss_calculator.generator_loss(x_pred_t, x_t_1)
#
#         if not validation:
#             if epoch >= config.pretrain_epoch:
#                 if config.adv_only:
#                     total_generator_loss = sum(adv_autobalancer(loss_G, loss_feat))
#                 else:
#                     total_generator_loss = sum(adv_autobalancer(loss_G, loss_feat, mel_reconstruction_loss))
#                 total_generator_loss.backward()
#             else:
#                 mel_reconstruction_loss.backward()
#
#             _, gen_grad_norm = gen_autoclip(netG)
#             optG.step()
#
#             costs = [
#                 [loss_D.item(), loss_G.item(), loss_feat.item(),
#                  mel_reconstruction_loss.item(),
#                  -1 * sdr_loss.item(), wav_loss.item(), gen_grad_norm]]
#         else:
#             curr_costs = [loss_D.item(), loss_G.item(), loss_feat.item(),
#                           mel_reconstruction_loss.item(),
#                           -1 * sdr_loss.item(), wav_loss.item(), 0]
#             for i in range(len(costs[0])):
#                 costs[0][i] += curr_costs[i]
#
#         # 日志
#         if not validation:
#             basic_logs(costs, writer, steps, epoch, iterno)
#         else:
#             validation_writer(epoch, steps)
#         steps += 1
#
#         # --- 验证集音频记录 (立体声兼容修复) ---
#         if validation:
#             should_log = False
#             if len(filenames) > 0:
#                 current_file = filenames[0]
#                 if config.validation_song is None or current_file == config.validation_song:
#                     should_log = True
#             elif config.validation_song is None:
#                 should_log = True
#
#             if should_log and validation_song_seconds < 5:
#                 # 获取 numpy 数据 [Channels, Time]
#                 dirty_np = x_t_0[0].cpu().numpy()
#                 clean_np = x_t_1[0].cpu().numpy()
#                 pred_np = x_pred_t[0].cpu().numpy()
#
#                 # [关键修复] 如果是立体声 (2, T)，混合为单声道 (T,) 以便 concatenate
#                 # 因为 output_aud 初始化的是 1D 数组
#                 if dirty_np.ndim > 1 and dirty_np.shape[0] > 1:
#                     dirty_np = np.mean(dirty_np, axis=0)
#                     clean_np = np.mean(clean_np, axis=0)
#                     pred_np = np.mean(pred_np, axis=0)
#
#                 # 确保挤压掉多余维度
#                 dirty_np = dirty_np.squeeze()
#                 clean_np = clean_np.squeeze()
#                 pred_np = pred_np.squeeze()
#
#                 output_aud[0] = np.concatenate((output_aud[0], dirty_np))
#                 output_aud[1] = np.concatenate((output_aud[1], clean_np))
#                 output_aud[2] = np.concatenate((output_aud[2], pred_np))
#
#                 validation_song_seconds += 1
#
#     if validation:
#         for i in range(len(costs[0])):
#             costs[0][i] /= (iterno + 1)
#
#         if len(output_aud[0]) == 0:
#             print(f"Warning: No validation audio logged.")
#             silence = np.zeros(config.sample_rate)
#             output_aud = [silence, silence, silence]
#
#     return steps, costs, output_aud
import torch
import torch.nn.functional as F
from .losses import *
from .save_and_log import *
from .stft_loss import *
from .augmentation import shift_phase
import numpy as np


def runEpoch(loader, config, netG, netD, optG, optD, device, epoch,
             steps, writer, gen_autoclip, disc_autoclip, adv_autobalancer, optD_spec=None, netD_spec=None,
             validation=False):
    costs = [[0, 0, 0, 0, 0, 0, 0]]
    gan_loss_calculator = GANLoss(netD)
    output_aud = [np.array([]), np.array([]), np.array([])]
    validation_song_seconds = 0
    multi_scale_mel_loss = MultiResolutionSTFTLoss().to(device)
    for iterno, x_t in enumerate(loader):
        if config.mono:
            x_t_0 = x_t[0].unsqueeze(1).float().to(device)
            x_t_1 = x_t[1].unsqueeze(1).float().to(device)
            x_t_2 = x_t[2].unsqueeze(1).float().to(device)
        else:
            x_t_0 = x_t[0].float().to(device)
            x_t_1 = x_t[1].float().to(device)
            x_t_2 = x_t[2].float().to(device)
            x_t_1_mono = (x_t_1[:, 0, :] + x_t_1[:, 1, :])
            x_t_1_mono /= torch.max(torch.abs(x_t_1_mono))

        inp = F.pad(x_t_0, (4000, 4000), "constant", 0)

        x_pred_t = netG(inp, x_t_0.unsqueeze(1)).squeeze(1)
        wav_loss = F.l1_loss(x_t_1, x_pred_t)

        if not config.mono:
            x_pred_t_mono = (x_pred_t[:, 0, :] + x_pred_t[:, 1, :])
            x_pred_t_mono /= torch.max(torch.abs(x_pred_t_mono))
            mel_reconstruction_loss = mel_spec_loss(x_pred_t_mono.squeeze(1), x_t_1_mono.squeeze(1))
        else:
            spec_convergence, log_mel_loss = multi_scale_mel_loss(x_pred_t.squeeze(1), x_t_1.squeeze(1))
            if config.use_both_reconstruction:
                mel_reconstruction_loss = log_mel_loss + spec_convergence
            else:
                mel_reconstruction_loss = log_mel_loss

        #######################
        # L1, SDR Loss        #
        #######################

        sdr = SISDRLoss()
        if config.mono:
            sdr_loss = sdr(x_pred_t.squeeze(1).unsqueeze(2),
                           x_t_1.squeeze(1).unsqueeze(2))
        else:
            sdr_loss = sdr(x_pred_t_mono.unsqueeze(2), x_t_1_mono.unsqueeze(2))

        #######################
        # Train Discriminator #
        #######################

        fake = x_pred_t.to(device)
        real = x_t_1.to(device)
        if config.augment:
            factor = np.random.uniform(-np.pi, np.pi)
            loss_D = gan_loss_calculator.discriminator_loss(shift_phase(fake.squeeze(1), factor).unsqueeze(1),
                                                            shift_phase(real.squeeze(1), factor).unsqueeze(1))
        else:
            loss_D = gan_loss_calculator.discriminator_loss(fake, real)

        if not validation and epoch >= config.pretrain_epoch:
            netD.zero_grad()
            loss_D.backward()
            optD.step()

        ###################
        # Train Generator #
        ###################
        loss_G, loss_feat = gan_loss_calculator.generator_loss(fake, real)

        if not validation:
            netG.zero_grad()
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
                 -1 * sdr_loss, wav_loss.item(), gen_grad_norm]]
        else:
            curr_costs = [loss_D.item(), loss_G.item(), loss_feat.item(),
                          mel_reconstruction_loss.item(),
                          -1 * sdr_loss, wav_loss.item(), 0]
            for i in range(len(costs[0])):
                costs[0][i] += curr_costs[i]
        # Call basic log info
        if not validation:
            basic_logs(costs, writer, steps, epoch, iterno)
        else:
            validation_writer(epoch, steps)
        steps += 1
        if validation and x_t[3][0] == config.validation_song:
            if config.mono and validation_song_seconds >= config.validation_song_start and validation_song_seconds <= config.validation_song_end:
                output_aud[0] = np.concatenate((output_aud[0], x_t_0.squeeze(0).squeeze(0).cpu().numpy()))
                output_aud[1] = np.concatenate((output_aud[1], x_t_1.squeeze(0).squeeze(0).cpu().numpy()))
                output_aud[2] = np.concatenate((output_aud[2], x_pred_t.squeeze(0).squeeze(0).cpu().numpy()))
            validation_song_seconds += 1
    if validation:
        for i in range(len(costs[0])):
            costs[0][i] /= (iterno + 1)
    return steps, costs, output_aud