import torch.nn as nn
import torch

from models.DTTNet.dp_tdf.modules import TFC_TDF, TFC_TDF_Res1, TFC_TDF_Res2
from models.DTTNet.dp_tdf.bandsequence import BandSequenceModelModule

from models.DTTNet.dp_tdf.RoPETransformer import RoPETransformer

from models.DTTNet.layers import (get_norm)
# from models.DTTNet.dp_tdf.abstract import AbstractModel

from modules.spectral_ops import Fourier, Band

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
                 RoPEParams,
                 **kwargs):

        super().__init__()
        # self.save_hyperparameters()

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

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_c_in, out_channels=g, kernel_size=(1, 1)),
            get_norm(bn_norm, g),
            nn.ReLU(),
        )

        f = self.dim_f
        c = g

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c_out, kernel_size=(1, 1)),
        )

        self.fourier = Fourier(n_fft=n_fft, hop_length=hop_length)

        self.num_bands=64
        self.band = Band(sr=sample_rate, n_fft=n_fft, bands_num=self.num_bands, in_channels=2, out_channels=hidden_channels, scale='mel')

        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()

        for i in range(self.n):
            c_in = c

            self.encoding_blocks.append(T_BLOCK(c_in, c, l, f, k, bn, bn_norm, bias=bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c + g),
                    nn.ReLU()
                )
            )


            self.decoding_blocks.insert(0,T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias))
            f = f // 2
            c += g

            self.us.insert(0,UpSamplingBlock(c,g,scale,bn_norm))

        self.bottleneck_block1 = T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias)
        self.bottleneck_block3 = RoPETransformer(c,**RoPEParams)
        self.bottleneck_block2 = BandSequenceModelModule(
            **bandsequence,
            input_dim_size=c,
            hidden_dim_size=2*c
        )

    def forward(self, x):
        '''
        Args:
            x: (batch, c*2, 2048, 256)
        '''

        origianl_length=x.shape[-1]
        x=self.fourier.stft(x)# B,F,T,C

        x=x.permute([0,3,1,2])  # B,C,F,T

        x = self.first_conv(x)

        x = x.transpose(-1, -2)# B,C,T,F

        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        # print(f"bottleneck in: {x.shape}")
        x = self.bottleneck_block1(x)

        x=self.bottleneck_block2(x)

        x=x.permute([0,2,3,1])# B,T,F,C

        x = self.bottleneck_block3(x)

        x=x.permute([0,3,1,2])# B,C,T,F

        for i in range(self.n):
            x = self.us[i](x,output_size=ds_outputs[-i - 1].shape)
            # print(f"us{i} in: {x.shape}")
            # print(f"ds{i} out: {ds_outputs[-i - 1].shape}")
            x = x * ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        x=x.permute([0,2,3,1])  # B,F,T,C
        # x=self.band.unsplit(x)
        x=self.fourier.istft(x.contiguous(),origianl_length)

        return x

class UpSamplingBlock(nn.Module):
    def __init__(self,c,g,scale,bn_norm):
        super().__init__()
        self.conv=nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale)
        self.norm=get_norm(bn_norm, c - g)
        self.relu=nn.ReLU()

    def forward(self, x,output_size):
        x=self.conv(x,output_size=output_size)
        x=self.norm(x)
        x=self.relu(x)
        return x