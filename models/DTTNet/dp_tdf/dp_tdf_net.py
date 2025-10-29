import torch.nn as nn
import torch

from models.DTTNet.dp_tdf.modules import TFC_TDF, TFC_TDF_Res1, TFC_TDF_Res2
from models.DTTNet.dp_tdf.bandsequence import BandSequenceModelModule

from models.DTTNet.layers import (get_norm)
from models.DTTNet.dp_tdf.abstract import AbstractModel

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
                 dim_t,
                 n_fft,
                 hop_length,
                 audio_ch, **kwargs):

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
        self.dim_t = dim_t
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
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()

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
            f = f // 2
            c += g

        self.bottleneck_block1 = T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias)
        self.bottleneck_block2 = BandSequenceModelModule(
            **bandsequence,
            input_dim_size=c,
            hidden_dim_size=2*c
        )

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            # print(f"i: {i}, in channels: {c}")
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c - g),
                    nn.ReLU()
                )
            )

            f = f * 2
            c -= g

            self.decoding_blocks.append(T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c_out, kernel_size=(1, 1)),
        )

    def forward(self, x):
        '''
        Args:
            x: (batch, c*2, 2048, 256)
        '''

        x=self.stft(x)

        x = self.first_conv(x)

        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        # print(f"bottleneck in: {x.shape}")
        x = self.bottleneck_block1(x)
        x = self.bottleneck_block2(x)

        for i in range(self.n):
            x = self.us[i](x)
            # print(f"us{i} in: {x.shape}")
            # print(f"ds{i} out: {ds_outputs[-i - 1].shape}")
            if x.shape != ds_outputs[-i - 1].shape:
                # 裁剪 x 的最后两个维度以匹配 skip_connection
                x = x[..., :ds_outputs[-i - 1].shape[-2], :ds_outputs[-i - 1].shape[-1]]
            x = x * ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        x=self.istft(x)

        return x

    def stft(self, x):
        '''
        Args:
            x: (batch, c, 261120)
        '''
        dim_b = x.shape[0]
        x=x.unsqueeze(1)
        x = x.reshape([dim_b * self.audio_ch, -1]) # (batch*c, 261120)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True,return_complex=False) # (batch*c, 3073, 256, 2)
        x = x.permute([0, 3, 1, 2]) # (batch*c, 2, 3073, 256)
        x = x.reshape([dim_b, self.audio_ch, 2, self.n_bins, -1]).reshape([dim_b, self.audio_ch * 2, self.n_bins, -1]) # (batch, c*2, 3073, 256)
        return x[:, :, :self.dim_f] # (batch, c*2, 2048, 256)

    def istft(self, x):
        '''
        Args:
            x: (batch, c*2, 2048, 256)
        '''
        dim_b = x.shape[0]
        x = torch.cat([x, self.freq_pad.repeat([x.shape[0], 1, 1, x.shape[-1]])], -2) # (batch, c*2, 3073, 256)
        x = x.reshape([dim_b, self.audio_ch, 2, self.n_bins, -1]).reshape([dim_b * self.audio_ch, 2, self.n_bins, -1]) # (batch*c, 2, 3073, 256)
        x = x.permute([0, 2, 3, 1]) # (batch*c, 3073, 256, 2)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True) # (batch*c, 261120)
        return x.reshape([dim_b, self.audio_ch, -1]) # (batch,c,261120)