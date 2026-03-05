import torch
import torch.nn.functional as F
import torch.nn as nn


def stft(x, fft_size, hop_size, win_length, window):
    """
    执行 STFT 并返回幅度谱。
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)

    # x_stft 形状: (B, F, T) (Complex)
    # 计算幅度: (B, F, T) (Real)
    mag = torch.abs(x_stft)

    # 避免 NaN
    mag = torch.clamp(mag, min=1e-7)

    # 转置
    return mag.transpose(2, 1)


class SpectralConvergengeLoss(nn.Module):
    """频谱收敛损失模块。"""

    def __init__(self):
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(nn.Module):
    """对数 STFT 幅度损失模块。"""

    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(nn.Module):
    """STFT 损失模块。"""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        # 注册 buffer 以便自动处理设备移动
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        window = self.window.to(x.device)

        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """多分辨率 STFT 损失模块。"""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss, self.factor_mag * mag_loss