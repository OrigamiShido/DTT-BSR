import torch.nn as nn
import torch

from models.DTTNet.dp_tdf.modules import TFC_TDF, TFC_TDF_Res1, TFC_TDF_Res2
from models.DTTNet.dp_tdf.bandsequence import BandSequenceModelModule
from models.DTTNet.layers import get_norm
from models.DTTNet.dp_tdf.SnM import SplitAndMerge
from modules.spectral_ops import Fourier

class DPTDFNet(nn.Module):
    def __init__(self,
                 num_blocks,
                 l,
                 g,
                 k,
                 bn,
                 bias,
                 bn_norm,
                 bandsequence,
                 block_type,
                 dim_f,
                 n_fft,
                 hop_length,
                 audio_ch,
                 sample_rate,
                 hidden_channels,
                 **kwargs):

        super().__init__()

        self.num_blocks = num_blocks
        self.l = l
        self.g = g
        self.k = k
        self.bn = bn
        self.bias = bias

        self.n = num_blocks // 2
        scale = (2, 2)

        self.dim_c_in = audio_ch * 2
        self.dim_c_out = audio_ch * 2
        self.dim_f = dim_f
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.audio_ch = audio_ch

        self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
        self.freq_pad = nn.Parameter(torch.zeros([1, self.dim_c_out, self.n_bins - self.dim_f, 1]), requires_grad=False)

        if block_type == "TFC_TDF":
            T_BLOCK = TFC_TDF
        elif block_type == "TFC_TDF_Res1":
            T_BLOCK = TFC_TDF_Res1
        elif block_type == "TFC_TDF_Res2":
            T_BLOCK = TFC_TDF_Res2
        else:
            raise ValueError(f"Unknown block type {block_type}")

        first_in_channels = self.dim_c_in

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=first_in_channels, out_channels=g, kernel_size=(1, 1)),
            get_norm(bn_norm, g),
            nn.ReLU(),
        )

        f = self.dim_f
        c = g

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c_out, kernel_size=(1, 1)),
        )

        self.fourier = Fourier(n_fft=n_fft, hop_length=hop_length)

        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()
        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()

        for i in range(self.n):
            c_in = c

            n_groups = 1

            layer_k = 3 if (i == 0 or i == self.n - 1) else k

            self.encoding_blocks.append(
                SplitAndMerge(c_in, c, f, n_groups, bn_norm, n_split_en=l, n_split_de=1, bn=bn, k=layer_k, bias=bias)
            )
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c + g),
                    nn.ReLU()
                )
            )

            self.decoding_blocks.insert(
                0, SplitAndMerge(c, c, f, n_groups, bn_norm, n_split_en=1, n_split_de=1, bn=bn, k=layer_k, bias=bias)
            )

            f = f // 2
            c += g

            self.us.insert(0, UpSamplingBlock(c, g, scale, bn_norm))

        n_groups_bottleneck = 1

        self.bottleneck_block1 = SplitAndMerge(
            c, c, f, n_groups_bottleneck, bn_norm, n_split_en=l, n_split_de=1, bn=bn, k=3, bias=bias
        )
        self.bottleneck_block2 = BandSequenceModelModule(
            **bandsequence,
            input_dim_size=c,
            hidden_dim_size=2 * c
        )

    def forward(self, x):
        original_length = x.shape[-1]
        sp = self.fourier.stft(x)
        x = sp.permute([0, 3, 1, 2])

        x = self.first_conv(x)
        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        x = self.bottleneck_block1(x)
        x = self.bottleneck_block2(x)

        for i in range(self.n):
            x = self.us[i](x, output_size=ds_outputs[-i - 1].shape)
            x = x * ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)
        x = self.final_conv(x)
        x = x.permute([0, 2, 3, 1])

        x = self.fourier.istft(x.contiguous(), original_length)

        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, c, g, scale, bn_norm):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale
        )
        self.norm = get_norm(bn_norm, c - g)
        self.relu = nn.ReLU()

    def forward(self, x, output_size):
        x = self.conv(x, output_size=output_size)
        x = self.norm(x)
        x = self.relu(x)
        return x