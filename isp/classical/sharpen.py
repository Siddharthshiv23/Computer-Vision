"""Sharpening (F-CV-6)."""
from __future__ import annotations

import cv2
import numpy as np

from isp.classical.color import rgb_to_ycbcr


def noise_aware_unsharp(rgb: np.ndarray, sigma: float | np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask on luma with soft-threshold coring at 2 sigma of the noise.

    Why: plain unsharp masking amplifies noise as much as detail. Detail weaker than
    ~2 noise sigmas is indistinguishable from noise, so it is cored away; stronger
    detail is boosted. Working on Y only (the same delta added to R, G, B) sharpens
    without colour fringes.
    """
    y = rgb_to_ycbcr(rgb)[..., 0]
    detail = y - cv2.GaussianBlur(y, (0, 0), 1.0)
    t = 2.0 * np.asarray(sigma, np.float32)
    cored = np.sign(detail) * np.maximum(np.abs(detail) - t, 0)
    return np.clip(rgb + amount * cored[..., None], 0, 1).astype(np.float32)
