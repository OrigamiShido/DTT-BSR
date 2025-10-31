import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.generator.ConvNeXt2DBlock import ConvNeXt2DBlock
import modules.spectral_ops as spectral_ops

class MoiseUNet(nn.Module):
    def __init__(self,









                 ):
        super().__init__()

