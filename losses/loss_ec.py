import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
import numpy as np

class ConsistencyLoss(nn.Module):
    """
    实现了论文 "A-Phase-and-Magnitude-Aware-Loss-Function-for-Speech-Enhancement"
    中的显式一致性损失函数 (Explicit Consistency Loss, L_EC)。 (已优化版本)

    该损失函数直接作用于音频波形，内部计算其STFT并衡量其一致性。
    此版本通过使用分组卷积 (Grouped Convolution) 进行了向量化，显著提升了计算速度。
    同时，它修复了在半精度训练 (AMP) 中可能出现的 'ComplexHalf' NotImplementedError。

    参数:
        n_fft (int): FFT 点数 (论文中的 N)。
        hop_length (int): 帧移长度 (论文中的 R)。
        window (str): 所用窗函数的名称 (例如 'hann')。
                      分析窗 W 和合成窗 S 将使用相同的窗。
    """
    def __init__(self, n_fft: int, hop_length: int, window: str = 'hann'):
        super().__init__()

        if n_fft % hop_length != 0:
            raise ValueError("n_fft 必须是 hop_length 的整数倍。")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.Q = n_fft // hop_length

        W_np = get_window(window, self.win_length)
        S_np = get_window(window, self.win_length)

        self.register_buffer('window', torch.from_numpy(W_np).float())

        alpha = self._compute_alpha(W_np, S_np)

        alpha_conv_kernel = alpha.unsqueeze(1).repeat(self.n_fft, 1, 1)
        self.register_buffer('alpha_conv_kernel', alpha_conv_kernel)


    def _compute_alpha(self, W: np.ndarray, S: np.ndarray) -> torch.Tensor:
        """
        根据论文中的公式 (9) 预计算 alpha 系数。
        """
        N = self.n_fft
        R = self.hop_length

        q_indices = torch.arange(-(self.Q - 1), self.Q)

        alpha = torch.zeros(2 * self.Q - 1, 2 * N - 1, dtype=torch.complex64)

        for q_idx, q in enumerate(q_indices):
            S_shifted = torch.zeros(N)
            s_start = max(0, -q * R)
            s_end = min(N, N - q * R)
            if s_start < s_end:
                S_shifted[s_start:s_end] = torch.from_numpy(S[q * R + s_start : q * R + s_end])

            term = torch.from_numpy(W) * S_shifted
            fft_term = torch.fft.fft(term, n=2*N-1)
            alpha[q_idx, :] = torch.fft.fftshift(fft_term) / N

        alpha[self.Q - 1, N - 1] -= 1.0

        return alpha

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        计算音频波形的一致性损失。(已优化, 修复半精度错误)

        参数:
            waveform (torch.Tensor): 输入的音频波形，形状为 (batch, n_samples)。

        返回:
            loss (torch.Tensor): 标量损失值。
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        H = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window, return_complex=True
        ).permute(0, 2, 1)

        batch_size, time_frames, _ = H.shape
        N = self.n_fft
        H_conj_flip = torch.conj(torch.flip(H[:, :, 1:-1], [2]))
        H_full = torch.cat([H, H_conj_flip], dim=2)

        H_permuted = H_full.permute(0, 2, 1)

        conv_out = F.conv1d(
            H_permuted,
            self.alpha_conv_kernel,
            padding='same',
            groups=N
        )

        conv_tensor = conv_out.view(batch_size, N, 2 * self.Q - 1, time_frames)

        q_indices = torch.arange(-(self.Q - 1), self.Q, device=H.device)
        n_indices = torch.arange(N, device=H.device)

        phase_term = torch.exp(1j * 2 * torch.pi * torch.outer(n_indices, q_indices) / N)

        # *** 修复点 ***
        # 将输入强制转换为 complex64，以避免在半精度训练中出现 "baddbmm_cuda" not implemented for 'ComplexHalf' 的错误
        C = torch.einsum('bnqt,nq->bnt',
                         conv_tensor.to(torch.complex64),
                         phase_term.to(torch.complex64)
                         )

        loss = torch.sum(torch.abs(C) ** 2)

        # 将最终损失的类型转换回输入波形的浮点类型，以确保混合精度训练的连续性
        return loss.to(waveform.dtype) / (batch_size * time_frames)

# --- 使用示例 ---
if __name__ == '__main__':
    import librosa
    import time

    n_fft = 512
    hop_length = 128
    target_sample_rate = 16000

    # 确保文件存在，否则使用随机数据
    try:
        audio_file_path = "/home/student/shihongtan/database/MSRBench/Vocals/result_DTTNet_RoPE/0_DT0.flac"
        waveform_np, _ = librosa.load(audio_file_path, sr=target_sample_rate, mono=True)
        waveform_full = torch.from_numpy(waveform_np)
        print(f"--- 从文件 '{audio_file_path}' 计算一致性损失 ---")
    except Exception:
        print("--- 音频文件未找到，使用随机数据进行测试 ---")
        waveform_full = torch.randn(target_sample_rate * 10) # 10秒随机音频

    # 截取一小段以进行快速测试
    waveform = waveform_full[:target_sample_rate * 5].float().unsqueeze(0) # 取5秒

    print(f"音频已加载，信号张量形状: {waveform.shape}")

    consistency_loss_fn = ConsistencyLoss(n_fft=n_fft, hop_length=hop_length)

    # 模拟半精度训练 (AMP)
    use_amp = torch.cuda.is_available()

    if use_amp:
        print("\n--- 测试半精度 (AMP) 模式 ---")
        consistency_loss_fn = consistency_loss_fn.to('cuda')
        waveform_cuda = waveform.to('cuda')

        try:
            # 使用 autocast 上下文管理器
            with torch.cuda.amp.autocast():
                start_time = time.time()
                loss_amp = consistency_loss_fn(waveform_cuda)
                end_time = time.time()

            print(f"半精度 (AMP) 损失: {loss_amp.item()}")
            print(f"半精度 (AMP) 计算耗时: {end_time - start_time:.4f} 秒")
            assert loss_amp.dtype == torch.float16, f"期望损失类型为 float16，但得到 {loss_amp.dtype}"
            print("半精度测试成功！")

        except Exception as e:
            print(f"半精度测试失败: {e}")

    # 测试全精度
    print("\n--- 测试全精度 (float32) 模式 ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    consistency_loss_fn_fp32 = consistency_loss_fn.to(device)
    waveform_fp32 = waveform.to(device)

    start_time = time.time()
    loss_fp32 = consistency_loss_fn_fp32(waveform_fp32)
    end_time = time.time()

    print(f"全精度损失: {loss_fp32.item()}")
    print(f"全精度计算耗时: {end_time - start_time:.4f} 秒")
