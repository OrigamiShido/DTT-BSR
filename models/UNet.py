import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.generator.ConvNeXt2DBlock import ConvNeXt2DBlock
import modules.spectral_ops as spectral_ops

class MelUNet(nn.Module):
    def __init__(self, hidden_channels, num_layers, upsampling_factor, window_size, hop_size, sample_rate):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.upsampling_factor = upsampling_factor
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.num_bands = 64

        self.fourier = spectral_ops.Fourier(n_fft=window_size, hop_length=hop_size)
        self.band = spectral_ops.Band(sr=sample_rate, n_fft=window_size, bands_num=self.num_bands, in_channels=2, out_channels=hidden_channels, scale='mel')
        self.upsampling_blocks = nn.ModuleList([ConvNeXt2DBlock(kernel_size=(7, 7), stride=(2, 2), input_dim=hidden_channels * (2 ** (i + 1)), output_dim=hidden_channels * (2 ** i), mode='transposed') for i in range(self.num_layers)])
        self.downsampling_blocks = nn.ModuleList([ConvNeXt2DBlock(kernel_size=(7, 7), stride=(2, 2), input_dim=hidden_channels * (2 ** i), output_dim=hidden_channels * (2 ** (i + 1)), mode='normal') for i in range(self.num_layers)])
        
    def forward(self, x):
        original_length = x.shape[1]
        x = self.fourier.stft(x)
        x = self.band.split(x) # (B, C, T, F)

        residuals = []
        for i in range(self.num_layers):
            residuals.append(x)
            x = self.downsampling_blocks[i](x)

        for i in range(self.num_layers):
            residual = residuals[self.num_layers - i - 1]
            x = self.upsampling_blocks[self.num_layers - i - 1](x, original_shape=(residual.shape[2], residual.shape[3]))
            if i < self.num_layers - 1:
                x = x + residual
        
        x = self.band.unsplit(x)
        x = self.fourier.istft(x.contiguous(), original_length)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelUNet(hidden_channels=64, num_layers=4, upsampling_factor=2, window_size=2048, hop_size=512, sample_rate=48000)
    
    x = torch.randn(4, 96000)
    x = x.to(device)
    model = model.to(device)
    output = model(x)
    print("Output shape:", output.shape)

    from thop import profile
    macs, params = profile(model, inputs=(x,))
    print(f"MACs: {macs / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")