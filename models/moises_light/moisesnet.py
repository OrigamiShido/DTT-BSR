import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from . import (get_norm)

from . import RoPETransformer

from . import SplitAndMerge,SplitModule
from . import BandSplit

from modules.spectral_ops import Fourier

class MoisesNet(nn.Module):
    def __init__(self,
                 dim_time,
                 n_fft,
                 hop_length,
                 bn_norm,
                 G,
                 bn,
                 bias,
                 N_band,
                 N_enc,
                 N_dec,
                 N_RoPE,
                 N_split_enc,
                 N_split_dec,
                 RoPEParams,
                 **kwargs
                 ):
        super(MoisesNet, self).__init__()

        assert N_split_enc % N_split_dec == 0 , "N_split_enc must be multiple of N_split_dec"

        self.enc2decRatio=int(N_enc/N_dec)

        self.dim_time=dim_time

        self.N_band=N_band
        self.N_enc=N_enc
        self.N_dec=N_dec
        self.N_RoPE=N_RoPE

        self.fourier=Fourier(n_fft=n_fft, hop_length=hop_length)
        self.band=BandSplit(self.N_band)

        self.bottleneck_rope=nn.ModuleList()

        self.channel=2*N_band

        self.first_split=SplitModule(self.channel,1,G,self.N_band,1)
        self.final_split=SplitModule(G,1,self.channel,self.N_band,1)

        point_dim=math.ceil(dim_time/N_band)
        point_c=G

        sample_kernel=(2,2)

        self.encoding_blocks=nn.ModuleList()
        self.decoding_blocks=nn.ModuleList()
        self.ds=nn.ModuleList()
        self.us=nn.ModuleList()

        cnt=1

        for i in range(self.N_enc):
            point_c_in=point_c

            self.encoding_blocks.append(SplitAndMerge(point_c_in,point_c,N_Band=N_band,N_split_enc=N_split_enc,N_split_dec=N_split_dec,bn=bn,hidden_layers=point_dim,bn_norm=bn_norm,bias=bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=point_c,out_channels=point_c+G,kernel_size=sample_kernel,stride=sample_kernel),
                    get_norm(bn_norm, point_c+G),
                    nn.ReLU()
                )
            )
            if cnt==1:
                self.decoding_blocks.insert(0,SplitAndMerge(point_c_in,point_c,N_Band=N_band,N_split_enc=N_split_enc,N_split_dec=N_split_dec,bn=bn,hidden_layers=point_dim,bn_norm=bn_norm,bias=bias))

            point_dim=point_dim//2
            point_c+=G
            if cnt== self.enc2decRatio:
                self.us.insert(0, UpSamplingBlock(point_c, G * self.enc2decRatio, sample_kernel, bn_norm))
                cnt=0

            cnt+=1

        self.bottleneck_splitmerge=SplitAndMerge(point_c,point_c,N_Band=N_band,N_split_enc=N_split_enc,N_split_dec=N_split_dec,bn=bn,hidden_layers=point_dim,bn_norm=bn_norm,bias=bias)

        for i in range(self.N_RoPE):
            self.bottleneck_rope.append(RoPETransformer(point_c,**RoPEParams))

    def forward(self, x):
        original_length=x.shape[-1]
        x=self.fourier.stft(x)# b,f,t,c

        x=self.band.split(x)# b,f,t,c

        x=x.permute(0,3,1,2)  # b,c,f,t

        x=self.first_split(x)
        x=x.transpose(-1,-2)  # b,c,t,f

        # downsampling
        ds_outputs=[]
        for i in range(self.N_enc):
            x=self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x=self.ds[i](x)

        # bottleneck
        x=self.bottleneck_splitmerge(x)

        x=x.permute([0,2,3,1])# B,T,F,C

        for i in range(self.N_RoPE):
            x=self.bottleneck_rope[i](x)

        x=x.permute([0,3,1,2])# B,C,T,F

        # upsampling
        for i in range(self.N_dec):
            x=self.us[i](x,output_size=ds_outputs[-((i+1)*self.enc2decRatio)].shape)
            x=x*ds_outputs[-((i+1)*self.enc2decRatio)]
            x=self.decoding_blocks[i](x)

        x=x.transpose(-1,-2)  # b,c,f,t

        x=self.final_split(x)

        x=x.permute([0,2,3,1])  # b,f,t,c

        # rejoin
        x=self.band.reconstruct(x)

        #istft
        x=self.fourier.istft(x,original_length)

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