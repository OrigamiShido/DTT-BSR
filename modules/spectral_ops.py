import torch
import torch.nn as nn
from einops import rearrange
import math
import librosa
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class Fourier(nn.Module):
    def __init__(self,
        n_fft: int = 2048,
        hop_length: int = 441
    ):
        super(Fourier, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        complex_stft = torch.stft(
            input=waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            return_complex=True
        )
        
        sp = torch.view_as_real(complex_stft)
        return sp

    def istft(self, sp: torch.Tensor, original_length: int) -> torch.Tensor:
        complex_sp = torch.view_as_complex(sp)

        y = torch.istft(
            complex_sp,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=original_length,
            normalized=False
        )
        return y

class Band(nn.Module):
    def __init__(
        self,
        sr: float,
        n_fft: int,
        bands_num: int,
        in_channels: int,
        out_channels: int,
        scale: str = 'mel'
    ) -> None:
        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.bands_num = bands_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.F = self.n_fft // 2 + 1 # Number of frequency bins

        filterbanks, ola_window = self.init_scale_banks()
        self.register_buffer(name='filterbanks', tensor=torch.from_numpy(filterbanks).float())
        self.register_buffer(name='ola_window', tensor=torch.from_numpy(ola_window).float())

        nonzero_indexes = [torch.from_numpy(fb.nonzero()[0]) for fb in filterbanks]
        
        lengths = torch.tensor([len(idx) for idx in nonzero_indexes])
        max_band_width = lengths.max()

        f_idxes = pad_sequence(
            sequences=nonzero_indexes,
            batch_first=True,
            padding_value=0 
        )
        self.register_buffer(name="f_idxes", tensor=f_idxes.flatten())
        
        range_tensor = torch.arange(max_band_width, device=f_idxes.device)[None, :]
        mask = (range_tensor < lengths[:, None]).flatten()
        self.register_buffer(name="mask", tensor=mask.to(self.filterbanks.dtype))
        
        self.pre_bandnet = BandLinear(
            bands_num=self.bands_num,
            in_channels=max_band_width * self.in_channels,
            out_channels=self.out_channels
        )

        self.post_bandnet = BandLinear(
            bands_num=self.bands_num,
            in_channels=self.out_channels,
            out_channels=max_band_width * self.in_channels
        )

    def init_scale_banks(self) -> tuple[np.ndarray, np.ndarray]:
        if self.scale == 'mel':
            filterbanks = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.bands_num, norm=None, fmax=self.sr/2)
        elif self.scale == 'linear':
            points = np.linspace(0, self.F, self.bands_num + 2)
            bins = points.astype(int)
            filterbanks = np.zeros((self.bands_num, self.F))
            for i in range(self.bands_num):
                start, mid, end = bins[i], bins[i+1], bins[i+2]
                if start >= mid or mid >= end: continue
                filterbanks[i, start:mid] = np.linspace(0, 1, mid - start)
                filterbanks[i, mid:end] = np.linspace(1, 0, end - mid)
        else:
            raise ValueError(f"Unsupported scale: {self.scale}")

        ola_window = np.sum(filterbanks, axis=0)
        ola_window[ola_window < 1e-8] = 1.0
        return filterbanks, ola_window

    def split(self, x: torch.Tensor) -> torch.Tensor:
        B, F, T, C = x.shape
        x = x.permute(0, 2, 3, 1) # -> (B, T, C, F)
        
        x = x[:, :, :, self.f_idxes] # -> (B, T, C, K*W)
        x = x * self.mask[None, None, None, :] # Apply mask to zero out padded values

        x = rearrange(x, 'b t c (k w) -> b t k (w c)', k=self.bands_num)

        x = self.pre_bandnet(x)
        
        x = rearrange(x, 'b t k d -> b d t k')
        return x

    def unsplit(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b d t k -> b t k d')

        x = self.post_bandnet(x)

        x = rearrange(x, 'b t k (w c) -> b t c (k w)', c=self.in_channels)
        x = x * self.mask[None, None, None, :] # Apply mask before scattering

        B, T, C, _ = x.shape
        buffer = torch.zeros(B, T, C, self.F, device=x.device, dtype=x.dtype)

        x_permuted = x.permute(0, 2, 1, 3) # -> (B, C, T, K*W)
        buffer = buffer.permute(0, 2, 1, 3) # -> (B, C, T, F)

        index = self.f_idxes.expand(B, C, T, -1)
        buffer.scatter_add_(dim=3, index=index, src=x_permuted)
        
        buffer = buffer.permute(0, 2, 1, 3) # -> (B, T, C, F)
        buffer /= self.ola_window[None, None, None, :]
        
        buffer = buffer.permute(0, 3, 1, 2) # -> (B, F, T, C)
        return buffer

class BandLinear(nn.Module):
    def __init__(self, bands_num: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(bands_num, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(bands_num, out_channels))
        bound = 1 / math.sqrt(in_channels) if in_channels > 0 else 0
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('btki,kio->btko', x, self.weight) + self.bias[None, None, :, :]
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    sample_rate = 48000
    original_waveform = torch.randn(2, sample_rate * 2).to(device) # B=2, T=2s
    fourier = Fourier(n_fft=2048, hop_length=512).to(device)

    print("--- Testing Fourier Module ---")
    sp = fourier.stft(original_waveform)
    reconstructed_waveform = fourier.istft(sp, original_waveform.shape[-1])

    print(f"Original waveform shape: {original_waveform.shape}")
    print(f"Spectrogram shape (B, F, T, C): {sp.shape}")
    print(f"Reconstructed waveform shape: {reconstructed_waveform.shape}")

    assert original_waveform.shape == reconstructed_waveform.shape, \
        "Reconstructed waveform shape does not match original."
    
    mse_error = torch.mean((original_waveform - reconstructed_waveform)**2)
    print(f"Fourier Reconstruction MSE: {mse_error.item():.2e}\n")
    
    print("--- Testing Band Module ---")

    x = sp
    
    scales_to_test = ['mel', 'linear']

    for scale in scales_to_test:
        print(f"Testing with '{scale}' scale...")
        try:
            band_module = Band(
                sr=sample_rate,
                n_fft=2048,
                bands_num=64,
                in_channels=x.shape[3],
                out_channels=32,
                scale=scale
            ).to(device)

            y = band_module.split(x)
            x_hat = band_module.unsplit(y)

            print(f"  Input shape to Band: {x.shape}")
            print(f"  Processed (banded) shape: {y.shape}")
            print(f"  Reconstructed shape from Band: {x_hat.shape}")
            
            assert x.shape == x_hat.shape, f"Reconstruction failed for {scale} scale."
            
            recon_error = torch.mean((x - x_hat)**2)
            print(f"  Spectrogram Reconstruction MSE on '{scale}' scale: {recon_error.item():.2e}\n")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  An error occurred during '{scale}' scale test: {e}\n")