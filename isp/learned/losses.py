"""Differentiable ISP loss: L1 + Lab colour loss + gradient loss (F-HY-3). Inputs are sRGB in [0, 1]."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-6
_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) sRGB in [0, 1] -> CIE Lab (D65). Gradient-safe at 0."""
    c = rgb.clamp(0.0, 1.0)
    lin = torch.where(c > 0.04045, ((c + 0.055) / 1.055).clamp(min=_EPS) ** 2.4, c / 12.92)
    xyz = torch.einsum("ij,bjhw->bihw", _RGB2XYZ.to(rgb), lin) / _WHITE.to(rgb).view(1, 3, 1, 1)
    f = torch.where(xyz > 0.008856, xyz.clamp(min=_EPS) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], dim=1)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between horizontal and vertical finite differences; penalises lost detail."""
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return F.l1_loss(dx(pred), dx(target)) + F.l1_loss(dy(pred), dy(target))


def isp_loss(pred: torch.Tensor, target: torch.Tensor, w_l1: float = 1.0,
             w_lab: float = 1.0, w_grad: float = 0.5) -> torch.Tensor:
    lab = F.l1_loss(rgb_to_lab(pred), rgb_to_lab(target)) / 100.0
    return w_l1 * F.l1_loss(pred, target) + w_lab * lab + w_grad * gradient_loss(pred, target)
