import torch
import torch.nn as nn
import torch.nn.functional as F

from . import (get_norm)

class SplitAndMerge(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 N_Band,
                 N_split_enc,
                 N_split_dec,
                 bn,
                 hidden_layers,
                 bn_norm,
                 bias=True,
                 ):
        super(SplitAndMerge, self).__init__()

        # split1
        self.split=SplitModule(dim_in,3,dim_out,N_Band,N_split_enc)

        # split2
        self.unsplit=SplitModule(dim_in,3,dim_out,N_Band,N_split_dec)

        # bottleneck
        self.bottleneck=nn.Sequential(
            nn.Linear(hidden_layers, hidden_layers // bn, bias=bias),
            get_norm(bn_norm, dim_out),
            nn.ReLU(),
            nn.Linear(hidden_layers // bn, hidden_layers, bias=bias),
            get_norm(bn_norm, dim_out),
            nn.ReLU()
        )

        # skip connection
        self.skip=nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=3, stride=1, padding=3//2),
            get_norm(bn_norm, dim_out),
            nn.ReLU(),
        )

    def forward(self, x):
        res=self.skip(x)
        x=self.split(x)
        x=x+self.bottleneck(x)
        x=self.unsplit(x)
        x=x+res
        return x

class SplitModule(nn.Module):
    def __init__(self,
                 dim_in,
                 K,
                 G,
                 N_Band,
                 N_Modules=1,
                 ):
        super(SplitModule, self).__init__()
        self.split=nn.ModuleList()
        for i in range(N_Modules):
            self.split.append(nn.Conv2d(in_channels=dim_in,out_channels=G,kernel_size=K,groups=N_Band,stride=1,padding=K//2))

    def forward(self, x):
        for single in self.split:
            x=single(x)
        return x