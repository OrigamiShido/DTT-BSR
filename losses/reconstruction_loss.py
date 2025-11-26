import torch
import torchaudio
from torch import nn
from typing import List

def safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clip(x, min=1e-7))

class MelSpecReconstructionLoss(nn.Module):
    def __init__(
        self, sample_rate: int = 48000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 100,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=True, power=1,
        )

    def forward(self, y_hat, y) -> torch.Tensor:
        mel_hat = safe_log(self.mel_spec(y_hat))
        mel = safe_log(self.mel_spec(y))

        loss = torch.nn.functional.l1_loss(mel, mel_hat)

        return loss

class MultiMelSpecReconstructionLoss(nn.Module):
    def __init__(
        self, 
        sample_rate: int = 48000, 
        n_fft: List[int] = [1024, 2048, 4096],
        hop_length: List[int] = [256, 512, 1024], 
        n_mels: List[int] = [80, 160, 320],
    ):
        super().__init__()
        assert len(n_fft) == len(hop_length) == len(n_mels), "n_fft, hop_length, and n_mels must have the same length"
        self.mel_specs = nn.ModuleList([
            MelSpecReconstructionLoss(sample_rate, n_f, h_l, n_m)
            for n_f, h_l, n_m in zip(n_fft, hop_length, n_mels)
        ])

    def forward(self, y_hat, y) -> torch.Tensor:
        loss = torch.zeros(1, device=y_hat.device, dtype=y_hat.dtype)
        for mel_spec in self.mel_specs:
            loss = loss + mel_spec(y_hat, y)
        loss = loss / len(self.mel_specs)
        return loss

class ComplexSpecReconstructionLoss(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 512, alpha=0.5):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha=alpha

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(torch.float32).eps
        y_hat_flat = y_hat.view(-1, y_hat.shape[-1])
        y_flat = y.view(-1, y.shape[-1])
        window = torch.hann_window(self.n_fft).to(y_hat.device).float()
        est_spec = torch.stft(
            y_hat_flat, n_fft=self.n_fft, hop_length=self.hop_length,
            window=window, return_complex=True
        )
        target_spec = torch.stft(
            y_flat, n_fft=self.n_fft, hop_length=self.hop_length,
            window=window, return_complex=True
        )
        loss=torch.norm(est_spec - target_spec, p='fro')**2
        return loss

class MultiComplexSpecReconstructionLoss(nn.Module):
    def __init__(
        self,
        n_fft: list = [32, 64, 128, 256, 512, 1024, 2048],
        hop_length: list = None,
        alpha=0.5,# 相位在这个loss中的比重
    ):
        super().__init__()
        if hop_length is None:
            hop_length = [n // 2 for n in n_fft]
        assert len(n_fft) == len(hop_length), "n_fft and hop_length must have the same length"
        self.alpha=alpha
        self.spec_losses = nn.ModuleList([
            ComplexSpecReconstructionLoss(n_f, h_l)
            for n_f, h_l in zip(n_fft, hop_length)
        ])

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=y_hat.device, dtype=y_hat.dtype)
        for spec_loss in self.spec_losses:
            loss = loss + spec_loss(y_hat, y)
        loss = loss / len(self.spec_losses)
        return loss

class WaveformReconstructionLoss(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.loss_fn=torch.nn.MSELoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # loss = self.loss_fn(y_hat, y)
        loss = torch.nn.functional.l1_loss(y_hat, y)
        return loss

if __name__ == '__main__':
    y_hat = torch.randn(1, 1, 48000)
    y = torch.randn(1, 1, 48000)
    mel_spec_loss = MelSpecReconstructionLoss()(y_hat, y)
    print(mel_spec_loss)
    multi_mel_spec_loss = MultiMelSpecReconstructionLoss()(y_hat, y)
    print(multi_mel_spec_loss)
    complex_spec_loss = ComplexSpecReconstructionLoss()(y_hat, y)
    print(complex_spec_loss)
    multi_complex_spec_loss = MultiComplexSpecReconstructionLoss()(y_hat, y)
    print(multi_complex_spec_loss)
    waveform_loss = WaveformReconstructionLoss()(y_hat, y)
    print(waveform_loss)