# python
import torch
import torch.nn as nn

class BandSplit(nn.Module):
    def __init__(self, n_band: int):
        super().__init__()
        self.n_band = n_band
        self._pad = 0

    def split(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("期望输入形状为 B*F*T*C。")
        b, f, t, c = x.shape
        pad = (-f) % self.n_band
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
        self._pad = pad
        f_per = x.shape[1] // self.n_band
        x = x.view(b, self.n_band, f_per, t, c)
        return x.permute(0, 2, 3, 1, 4).reshape(b, f_per, t, c * self.n_band)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("期望输入形状为 B*(F/N_band)*T*(C*N_band)。")
        b, f_per, t, c_times = x.shape
        if c_times % self.n_band:
            raise ValueError("通道数不能被 N_band 整除。")
        c = c_times // self.n_band
        y = x.view(b, f_per, t, self.n_band, c).permute(0, 3, 1, 2, 4)
        y = y.reshape(b, f_per * self.n_band, t, c)
        if self._pad:
            y = y[:, :-self._pad]
        return y


