import torch
import torch.nn as nn
import numpy as np


def get_omnidirectional_kernels(device=None, dtype=torch.float32):
    """Return 9 3x3 omnidirectional kernels (center + 8 neighbors).

    These kernels follow the topology in the paper (Figure 2). If the
    paper specifies normalization coefficients, replace the integers below
    with the exact coefficients.
    Output shape: [9, 1, 3, 3]
    """
    kernels = torch.tensor([
        [[[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]],
        [[[-1, -1, -1], [-1, 0, 1], [-1, -1, -1]]],
        [[[-1, -1, -1], [1, 0, -1], [-1, -1, -1]]],
        [[[-1, -1, -1], [-1, 0, -1], [-1, 1, -1]]],
        [[[-1, 1, -1], [-1, 0, -1], [-1, -1, -1]]],
        [[[-1, -1, -1], [-1, 0, -1], [1, 1, -1]]],
        [[[1, 1, -1], [-1, 0, -1], [-1, -1, -1]]],
        [[[-1, -1, 1], [-1, 0, -1], [-1, 1, -1]]],
        [[[1, -1, -1], [-1, 0, -1], [-1, -1, 1]]],
    ], dtype=dtype)
    if device is not None:
        kernels = kernels.to(device=device, dtype=dtype)
    return kernels


def signed_wrap(x: torch.Tensor) -> torch.Tensor:
    """Map angles to (-pi, pi] preserving sign (works on tensors).

    Uses torch.remainder so the operation remains on tensors and devices.
    """
    two_pi = 2.0 * np.pi
    return torch.remainder(x + np.pi, two_pi) - np.pi


def abs_wrap(x: torch.Tensor) -> torch.Tensor:
    """Absolute smallest-angle difference (non-negative)."""
    return torch.abs(signed_wrap(x))


def compute_omnidirectional_phase_derivatives(phase: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """Compute omnidirectional phase derivatives via 2D conv.

    phase: [B, 1, L, K]
    kernels: [9, 1, 3, 3]
    returns: [B, 9, L, K]
    """
    # ensure kernels on same device/dtype
    kernels = kernels.to(device=phase.device, dtype=phase.dtype)
    padded = nn.functional.pad(phase, pad=[1, 1, 1, 1], mode='reflect')
    deriv = nn.functional.conv2d(padded, kernels, groups=1)
    return deriv


class WOPLoss(nn.Module):
    """Weighted Omnidirectional Phase loss (paper eq.6).

    Uses absolute wrapped phase-derivative differences, weighted by
    normalized magnitude.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('kernels', get_omnidirectional_kernels())

    def forward(self, target_mag: torch.Tensor, target_phase: torch.Tensor, pred_phase: torch.Tensor) -> torch.Tensor:
        # target_phase/pred_phase: [B,1,L,K], target_mag: [B,1,L,K]
        targ_deriv = compute_omnidirectional_phase_derivatives(target_phase, self.kernels)
        pred_deriv = compute_omnidirectional_phase_derivatives(pred_phase, self.kernels)

        deriv_diff = abs_wrap(targ_deriv - pred_deriv)  # [B,9,L,K]

        # weight = |Y| / max(|Y|) per-sample
        max_mag = torch.max(target_mag, dim=[2, 3], keepdim=True)[0]
        weight = target_mag / (max_mag + 1e-8)

        B = target_mag.shape[0]
        _, _, L, K = target_mag.shape
        loss = (weight * deriv_diff).sum() / (9.0 * B * L * K)
        return loss


class ORILoss(nn.Module):
    """Omnidirectional Real/Imag loss (paper eq.7).

    Computes real/imag components using cos/sin of signed wrapped derivatives
    and compares target/pred components with L1 or L2 reduction.
    """

    def __init__(self, distance_type: str = 'L1'):
        super().__init__()
        self.register_buffer('kernels', get_omnidirectional_kernels())
        self.distance = nn.L1Loss(reduction='sum') if distance_type == 'L1' else nn.MSELoss(reduction='sum')

    def forward(self, target_mag: torch.Tensor, target_phase: torch.Tensor, pred_mag: torch.Tensor, pred_phase: torch.Tensor) -> torch.Tensor:
        targ_deriv = compute_omnidirectional_phase_derivatives(target_phase, self.kernels)
        pred_deriv = compute_omnidirectional_phase_derivatives(pred_phase, self.kernels)

        # use signed wrap so cos/sin receive angles in (-pi,pi]
        targ_deriv = signed_wrap(targ_deriv)
        pred_deriv = signed_wrap(pred_deriv)

        targ_real = target_mag * torch.cos(targ_deriv)
        targ_imag = target_mag * torch.sin(targ_deriv)
        pred_real = pred_mag * torch.cos(pred_deriv)
        pred_imag = pred_mag * torch.sin(pred_deriv)

        B = target_mag.shape[0]
        _, _, L, K = target_mag.shape
        real_loss = self.distance(targ_real, pred_real)
        imag_loss = self.distance(targ_imag, pred_imag)
        loss = (real_loss + imag_loss) / (9.0 * B * L * K)
        return loss


class CORILoss(nn.Module):
    """Coupled ORI loss (paper eq.8).

    Implements the coupling between amplitude difference and phase-derivative
    difference per time-frequency bin.
    """

    def __init__(self, distance_type: str = 'L1'):
        super().__init__()
        self.register_buffer('kernels', get_omnidirectional_kernels())
        self.distance = nn.L1Loss(reduction='sum') if distance_type == 'L1' else nn.MSELoss(reduction='sum')

    def forward(self, target_mag: torch.Tensor, target_phase: torch.Tensor, pred_mag: torch.Tensor, pred_phase: torch.Tensor) -> torch.Tensor:
        targ_deriv = compute_omnidirectional_phase_derivatives(target_phase, self.kernels)
        pred_deriv = compute_omnidirectional_phase_derivatives(pred_phase, self.kernels)

        # per-TF relative amplitude difference (stable)
        eps = 1e-8
        mag_diff = self.distance(target_mag, pred_mag) / (target_mag.numel())  # 点态距离归一化
        mag_diff = mag_diff.expand(-1, 9, -1, -1)  # -> [B,9,L,K]

        deriv_diff = abs_wrap(targ_deriv - pred_deriv)  # [B,9,L,K]

        B = target_mag.shape[0]
        _, _, L, K = target_mag.shape
        coupled_term = (mag_diff * deriv_diff).sum()
        loss = (2.0 / (9.0 * np.pi)) * coupled_term / (B * L * K)
        return loss


if __name__ == '__main__':
    # local quick sanity check
	wop = WOPLoss()
	ori = ORILoss()
	cori = CORILoss()
	mel = nn.L1Loss()

	B, L, K = 2, 100, 513
	target_mag = torch.randn(B, 1, L, K).abs()
	target_phase = (torch.randn(B, 1, L, K) % (2 * np.pi)) - np.pi
	pred_mag = torch.randn(B, 1, L, K).abs()
	pred_phase = (torch.randn(B, 1, L, K) % (2 * np.pi)) - np.pi
	target_mel = torch.randn(B, 80, L)
	pred_mel = torch.randn(B, 80, L)

	total = (mel(pred_mel, target_mel)
			 + 0.1 * wop(target_mag, target_phase, pred_phase)
			 + 0.1 * cori(target_mag, target_phase, pred_mag, pred_phase))
	print('total loss:', float(total))


