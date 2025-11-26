from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from torch import nn
import torch.nn.functional as F

class SISNRLoss(nn.Module):
    def __init__(self, return_type='direct', reduction='mean'):
        super(SISNRLoss, self).__init__()
        self.reduction = reduction
        self.return_type=return_type

    def forward(self, est_source, target_source):
        batch_num = est_source.shape[0]
        loss = scale_invariant_signal_noise_ratio(est_source, target_source).sum()  # 越大越好
        loss= loss / batch_num                                 # 原始损失

        if self.return_type=='direct':
            loss= -loss
        if self.return_type=='softplus':
            loss = F.softplus(-loss)                 # ln(1 + e^(-loss))

        return loss