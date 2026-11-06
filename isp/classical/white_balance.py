"""Automatic white balance (F-CV-2).

Each estimator returns per-channel gains (r, g, b) with g == 1, to be multiplied
into linear camera RGB. Degenerate frames (all black, all clipped) return ones.
"""
from __future__ import annotations

import cv2
import numpy as np

_EPS = 1e-6
_SAT = 0.98  # pixels above this are treated as clipped and ignored


def _gains(illum: np.ndarray) -> np.ndarray:
    """Illuminant colour -> gains normalised to green; ones if the estimate is unusable."""
    illum = np.asarray(illum, np.float64)
    if not np.isfinite(illum).all() or illum.min() < _EPS:
        return np.ones(3, np.float32)
    return (illum[1] / illum).astype(np.float32)


def _valid(rgb: np.ndarray) -> np.ndarray:
    """Pixels that are neither clipped nor black, as an (N, 3) array."""
    p = rgb.reshape(-1, 3)
    keep = (p.max(1) < _SAT) & (p.max(1) > _EPS)
    return p[keep]


def _minkowski(rgb: np.ndarray, p: float) -> np.ndarray:
    v = _valid(rgb)
    if len(v) == 0:
        return np.ones(3, np.float32)
    return _gains(np.power(np.mean(v.astype(np.float64) ** p, 0), 1 / p))


def gray_world(rgb: np.ndarray) -> np.ndarray:
    """Gray-world: assume the scene averages to grey (Minkowski p=1).

    Why: the classic one-line baseline; robust unless one colour dominates.
    """
    return _minkowski(rgb, 1.0)


def shades_of_gray(rgb: np.ndarray, p: float = 6.0) -> np.ndarray:
    """Shades-of-gray: Minkowski p-norm, between gray-world (p=1) and white-patch (p=inf).

    Why: a higher p lets bright, likely-neutral pixels dominate without trusting one noisy maximum.
    """
    return _minkowski(rgb, p)


def gray_edge(rgb: np.ndarray, p: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Gray-edge: assume the average of edge (gradient) colours is grey.

    Why: gradients cancel large uniform colour regions, fixing gray-world's main failure.
    """
    s = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigma)
    gx, gy = cv2.Sobel(s, cv2.CV_32F, 1, 0), cv2.Sobel(s, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2).reshape(-1, 3).astype(np.float64)
    if mag.max() < _EPS:
        return np.ones(3, np.float32)
    return _gains(np.power(np.mean(mag ** p, 0), 1 / p))


def white_patch(rgb: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """White-patch (max-RGB): the brightest pixels are assumed to reflect the light source.

    Why: exact when a white object exists; the percentile (not max) resists hot pixels.
    """
    v = _valid(rgb)
    if len(v) == 0:
        return np.ones(3, np.float32)
    return _gains(np.percentile(v, percentile, axis=0))


def fuse_awb(rgb: np.ndarray) -> np.ndarray:
    """Adaptive median fusion of the four estimators.

    Why: every single estimator has a failure scene, but they rarely fail together,
    so the per-channel median of their gains is more robust than any one. White-patch
    is dropped when >2% of pixels are clipped, since clipping hides the true maximum.
    """
    est = [gray_world(rgb), shades_of_gray(rgb), gray_edge(rgb)]
    if (rgb.max(-1) >= _SAT).mean() <= 0.02:
        est.append(white_patch(rgb))
    illum = 1.0 / np.median(np.stack(est), axis=0)
    return _gains(illum)


def apply_gains(rgb: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Multiply linear RGB by gains and clip to [0, 1]."""
    return np.clip(rgb * np.asarray(gains, np.float32), 0, 1).astype(np.float32)
