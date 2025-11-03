import torch
import torch.nn as nn

from models.DTTNet.layers import (get_norm)

class TFC(nn.Module):
    def __init__(self, c_in, c_out, l, k, bn_norm):
        super(TFC, self).__init__()

        self.H = nn.ModuleList()
        for i in range(l):
            if i == 0:
                c_in = c_in
            else:
                c_in = c_out
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x


class DenseTFC(nn.Module):
    def __init__(self, c_in, c_out, l, k, bn_norm):
        super(DenseTFC, self).__init__()

        self.conv = nn.ModuleList()
        for i in range(l):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.conv[:-1]:
            x = torch.cat([layer(x), x], 1)
        return self.conv[-1](x)


class TFC_TDF(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, dense=False, bias=True):
        super(TFC_TDF, self).__init__()

        self.use_tdf = bn is not None
        self.f = f  # 频率维度
        self.c_out = c_out

        self.tfc = DenseTFC(c_in, c_out, l, k, bn_norm) if dense else TFC(c_in, c_out, l, k, bn_norm)

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    # 移除BatchNorm，因为线性层输出是2D
                    nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    # 移除BatchNorm，因为线性层输出是2D
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    # 移除BatchNorm，因为线性层输出是2D
                    nn.ReLU()
                )

    def forward(self, x):
        # x的形状应该是 [batch, channels, freq, time]
        x = self.tfc(x)
        
        if self.use_tdf:
            batch, channels, freq, time = x.shape
            
            # 如果频率维度不匹配，使用自适应处理
            if freq != self.f:
                # 使用插值调整频率维度
                x_tdf = torch.nn.functional.interpolate(x, size=(self.f, time), mode='nearest')
            else:
                x_tdf = x
            
            # 转换维度: [batch, channels, freq, time] -> [batch*channels*time, freq]
            x_reshaped = x_tdf.permute(0, 1, 3, 2).contiguous()  # [batch, channels, time, freq]
            x_reshaped = x_reshaped.view(-1, self.f)  # [batch*channels*time, freq]
            
            # 应用TDF（没有BatchNorm）
            tdf_output = self.tdf(x_reshaped)  # [batch*channels*time, freq]
            
            # 恢复形状
            tdf_output = tdf_output.view(batch, channels, time, self.f)  # [batch, channels, time, freq]
            tdf_output = tdf_output.permute(0, 1, 3, 2)  # [batch, channels, freq, time]
            
            # 如果之前调整了尺寸，现在调整回来
            if freq != self.f:
                tdf_output = torch.nn.functional.interpolate(tdf_output, size=(freq, time), mode='nearest')
            
            x = x + tdf_output
        
        return x


class TFC_TDF_Res1(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, dense=False, bias=True):
        super(TFC_TDF_Res1, self).__init__()

        self.use_tdf = bn is not None
        self.f = f
        self.c_out = c_out

        self.tfc = DenseTFC(c_in, c_out, l, k, bn_norm) if dense else TFC(c_in, c_out, l, k, bn_norm)
        self.res = TFC(c_in, c_out, 1, k, bn_norm)

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    nn.ReLU()
                )

    def forward(self, x):
        res = self.res(x)
        x = self.tfc(x)
        x = x + res
        
        if self.use_tdf:
            batch, channels, freq, time = x.shape
            
            if freq != self.f:
                x_tdf = torch.nn.functional.interpolate(x, size=(self.f, time), mode='nearest')
            else:
                x_tdf = x
            
            x_reshaped = x_tdf.permute(0, 1, 3, 2).contiguous()
            x_reshaped = x_reshaped.view(-1, self.f)
            
            tdf_output = self.tdf(x_reshaped)
            
            tdf_output = tdf_output.view(batch, channels, time, self.f)
            tdf_output = tdf_output.permute(0, 1, 3, 2)
            
            if freq != self.f:
                tdf_output = torch.nn.functional.interpolate(tdf_output, size=(freq, time), mode='nearest')
            
            x = x + tdf_output
        
        return x


class TFC_TDF_Res2(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, dense=False, bias=True):
        super(TFC_TDF_Res2, self).__init__()

        self.use_tdf = bn is not None
        self.f = f
        self.c_out = c_out

        self.tfc1 = TFC(c_in, c_out, l, k, bn_norm)
        self.tfc2 = TFC(c_in, c_out, l, k, bn_norm)
        self.res = TFC(c_in, c_out, 1, k, bn_norm)

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    nn.ReLU()
                )

    def forward(self, x):
        res = self.res(x)
        x = self.tfc1(x)
        
        if self.use_tdf:
            batch, channels, freq, time = x.shape
            
            if freq != self.f:
                x_tdf = torch.nn.functional.interpolate(x, size=(self.f, time), mode='nearest')
            else:
                x_tdf = x
            
            x_reshaped = x_tdf.permute(0, 1, 3, 2).contiguous()
            x_reshaped = x_reshaped.view(-1, self.f)
            
            tdf_output = self.tdf(x_reshaped)
            
            tdf_output = tdf_output.view(batch, channels, time, self.f)
            tdf_output = tdf_output.permute(0, 1, 3, 2)
            
            if freq != self.f:
                tdf_output = torch.nn.functional.interpolate(tdf_output, size=(freq, time), mode='nearest')
            
            x = x + tdf_output
        
        x = self.tfc2(x)
        x = x + res
        return x
