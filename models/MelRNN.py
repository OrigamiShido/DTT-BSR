import torch
import torch.nn as nn
from einops import rearrange
from modules.generator.RNNBlock import RNNBlock
import modules.spectral_ops as spectral_ops

class MelRNN(nn.Module):
    def __init__(self, hidden_channels, num_layers, num_groups, window_size, hop_size, sample_rate):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.num_bands = 64
        self.max_seq_len = 200
        self.max_seq_len = self.max_seq_len * window_size // hop_size

        self.fourier = spectral_ops.Fourier(n_fft=window_size, hop_length=hop_size)
        self.band = spectral_ops.Band(sr=sample_rate, n_fft=window_size, bands_num=self.num_bands, in_channels=2, out_channels=hidden_channels, scale='mel')

        self.freq_blocks = nn.ModuleList([RNNBlock(hidden_channels, 2, num_groups, "gru", True) for _ in range(self.num_layers)])
        self.time_blocks = nn.ModuleList([RNNBlock(hidden_channels, 2, num_groups, "gru", False) for _ in range(self.num_layers)])

    def forward(self, x):
        original_length = x.shape[1]
        x = self.fourier.stft(x)
        x = self.band.split(x) # (B, C, T, F)
        
        x = rearrange(x, 'b c t f -> b t f c')
        b, t, f, c = x.shape
        for i in range(self.num_layers):
            x = rearrange(x, 'b t f c -> (b t) f c')
            x = self.freq_blocks[i](x)
            x = rearrange(x, '(b t) f c -> b t f c', t=t)

            x = rearrange(x, 'b t f c -> (b f) t c')
            x = self.time_blocks[i](x)
            x = rearrange(x, '(b f) t c -> b t f c', f=f)
        
        x = rearrange(x, 'b t f c -> b c t f')
        x = self.band.unsplit(x)
        x = self.fourier.istft(x.contiguous(), original_length)
        return x

if __name__ == "__main__":
    model = MelRNN(hidden_channels=128, num_layers=4, num_groups=4, window_size=2048, hop_size=512, sample_rate=48000)
    
    x = torch.randn(4, 96000)
    
    output = model(x)
    print("Output shape:", output.shape)

    from thop import profile
    macs, params = profile(model, inputs=(x,))
    print(f"MACs: {macs / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")