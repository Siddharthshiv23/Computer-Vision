"""Network input construction, shared by training and inference so they cannot drift apart."""
from __future__ import annotations

import numpy as np
import torch

from isp.types import RawImage

_MAD_TO_SIGMA = 1.4826


def pack_bayer(bayer: np.ndarray) -> np.ndarray:
    """(H, W) RGGB mosaic -> (4, H/2, W/2) channels [R, G1, G2, B]."""
    return np.stack([bayer[0::2, 0::2], bayer[0::2, 1::2], bayer[1::2, 0::2], bayer[1::2, 1::2]])


def estimate_bayer_sigma(bayer: np.ndarray) -> float:
    """Robust noise std from differences of same-colour neighbours (distance 2); 0.0 on flat frames."""
    d = (bayer[:, 2:] - bayer[:, :-2]).astype(np.float64)
    mad = np.median(np.abs(d - np.median(d)))
    return float(_MAD_TO_SIGMA * mad / np.sqrt(2.0))


def rawnet_input(raw: RawImage) -> torch.Tensor:
    """(5, H/2, W/2): packed Bayer + constant noise sigma channel."""
    packed = pack_bayer(raw.bayer.astype(np.float32))
    sigma = np.full((1, *packed.shape[1:]), estimate_bayer_sigma(raw.bayer), np.float32)
    return torch.from_numpy(np.concatenate([packed, sigma]).astype(np.float32))


def liteisp_input(rgb: np.ndarray, noise_a: float, noise_b: float, exposure: float = 1.0) -> torch.Tensor:
    """(4, H, W): classical sRGB + sigma map, lin ~ rgb**2.2 (approximate linearisation).

    noise_a and noise_b describe the RAW signal, before the front end's exposure gain g. The image
    here is after that gain, so sigma = g * sqrt(a * lin / g + b). Dropping g would understate noise
    most on dark frames, which is where the network needs the map.
    """
    gain = max(float(exposure), 1e-6)
    lin = np.clip(rgb, 0.0, 1.0) ** 2.2
    sigma = gain * np.sqrt(np.maximum(noise_a * lin.mean(axis=2) / gain + noise_b, 0.0))
    x = np.concatenate([rgb.transpose(2, 0, 1), sigma[None]]).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(x))
