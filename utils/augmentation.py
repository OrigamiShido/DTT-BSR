# import torch
# import numpy as np
#
# def shift_phase(waveform,factor):
#     stft_data = torch.stft(waveform, n_fft=1024,return_complex=True)
#     magnitude = torch.abs(stft_data)
#     phase = torch.angle(stft_data)
#     new_phase = phase + factor
#     stft_data = magnitude * torch.exp(1j*new_phase)
#     new_waveform = torch.istft(stft_data, n_fft=1024)
#     return new_waveform


import torch
import numpy as np


def shift_phase(audio, shift):
    """
    对音频进行相位偏移 (Phase Shift) 数据增强。
    Args:
        audio (Tensor): 输入音频张量，形状可以是 [Batch, Time] 或 [Batch, 1, Time]。
        shift (float): 相位偏移量 (弧度)。
    Returns:
        Tensor: 相位偏移后的音频张量，形状与输入一致。
    """
    # 记录原始维度
    original_dim = audio.dim()
    if original_dim == 3:
        audio = audio.squeeze(1)

    n_fft = 2048
    hop_length = 512
    win_length = 2048

    device = audio.device

    window = torch.hann_window(win_length).to(device)

    # 1. Forward STFT
    # 返回复数张量
    stft_data = torch.stft(audio, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, window=window,
                           return_complex=True)

    # 2. 计算相位偏移量
    # e^(j * shift) = cos(shift) + j*sin(shift)
    shift_val = torch.tensor(shift).float()
    # 创建复数相位旋转因子
    shift_complex = torch.complex(torch.cos(shift_val), torch.sin(shift_val)).to(device)

    # 3. 应用相位偏移
    # 在频域乘以旋转因子相当于在时域做全通滤波，只改变相位不改变幅度
    stft_shifted = stft_data * shift_complex

    # 4. Inverse STFT
    # 必须传入相同的 window 以保证无损重建 (在重叠相加满足条件的情况下)
    shifted_audio = torch.istft(stft_shifted, n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window=window)

    # 5. 长度对齐
    if shifted_audio.shape[-1] < audio.shape[-1]:
        # 如果短了，在末尾填充
        pad_len = audio.shape[-1] - shifted_audio.shape[-1]
        shifted_audio = torch.nn.functional.pad(shifted_audio, (0, pad_len))
    elif shifted_audio.shape[-1] > audio.shape[-1]:
        # 如果长了，截断末尾
        shifted_audio = shifted_audio[..., :audio.shape[-1]]

    # 6. 恢复原始维度
    if original_dim == 3:
        shifted_audio = shifted_audio.unsqueeze(1)

    return shifted_audio
