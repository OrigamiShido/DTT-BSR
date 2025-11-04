from torch import nn

from models.DTTNet.layers import (get_norm)

class SplitAndMerge(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 f,
                 n_band,
                 bn_norm,
                n_split_en=3,
                 n_split_de=3,
                 bn=2,
                 k=3,
                bias=False,
                 ):
        super().__init__()

        self.use_tdf = bn is not None

        self.encoder=nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        for i in range(n_split_en):
            self.encoder.append(
                SplitModule(
                    c_in=c_in,
                    c_out=c_out,
                    k=k,
                    n_groups=n_band,
                )
            )

        for i in range(n_split_de):
            self.decoder.append(
                SplitModule(
                    c_in=c_in,
                    c_out=c_out,
                    k=k,
                    n_groups=n_band,
                )
            )

        self.res=nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2),
            get_norm(bn_norm, c_out),
            nn.ReLU(),
        )# 残差链接

        if self.use_tdf:
            if bn == 0:
                # print(f"TDF={f},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )
            else:
                # print(f"TDF={f},{f // bn},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )

    def forward(self, x):
        residual=self.res(x)
        for enc in self.encoder:
            x=enc(x)
        if self.use_tdf:
            x=x+self.tdf(x)
        for dec in self.decoder:
            x=dec(x)
        x=x+residual
        return x

class SplitModule(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 n_groups,
                 k,
                 ):
        super().__init__()
        self.split=nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2,groups=n_groups)

    def forward(self, x):
        x=self.split(x)
        return x