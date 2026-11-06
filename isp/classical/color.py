"""Colour correction matrix (F-CV-3) and colour-space transforms (F-CV-6)."""
from __future__ import annotations

import numpy as np

from isp.types import InputError

# Average-camera camera-RGB -> linear sRGB. Rows sum to 1 so white stays white after AWB.
DEFAULT_CCM = np.array([[1.60, -0.40, -0.20],
                        [-0.25, 1.50, -0.25],
                        [-0.05, -0.45, 1.50]], np.float32)

_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], np.float32)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ).astype(np.float32)
_D65 = _RGB2XYZ.sum(1)
_YCC = np.array([[0.299, 0.587, 0.114],
                 [-0.168736, -0.331264, 0.5],
                 [0.5, -0.418688, -0.081312]], np.float32)
_YCC_INV = np.linalg.inv(_YCC).astype(np.float32)


def resolve_ccm(ccm: np.ndarray | None) -> np.ndarray:
    """CCM from DNG metadata if given, else the average-camera default.

    Why: rows are renormalised to sum to 1 so a bad or unscaled matrix can never
    tint neutral greys; malformed matrices raise a friendly InputError.
    """
    if ccm is None:
        return DEFAULT_CCM
    m = np.asarray(ccm, np.float64)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise InputError("The colour matrix must be a finite 3x3 array.")
    sums = m.sum(1, keepdims=True)
    if np.any(np.abs(sums) < 1e-6):
        raise InputError("The colour matrix has a row that sums to zero.")
    return (m / sums).astype(np.float32)


def apply_ccm(rgb: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """Camera RGB -> linear sRGB, clipped to [0, 1]."""
    return np.clip(rgb @ ccm.T, 0, 1).astype(np.float32)


def srgb_encode(x: np.ndarray) -> np.ndarray:
    """Linear -> sRGB transfer curve (IEC 61966-2-1)."""
    x = np.clip(x, 0, 1).astype(np.float32)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def srgb_decode(y: np.ndarray) -> np.ndarray:
    """sRGB transfer curve -> linear."""
    y = np.clip(y, 0, 1).astype(np.float32)
    return np.where(y <= 0.04045, y / 12.92, np.power((y + 0.055) / 1.055, 2.4)).astype(np.float32)


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """BT.601 full-range YCbCr (Cb, Cr offset by 0.5). Used so luma can be filtered apart from chroma."""
    return (rgb @ _YCC.T + np.array([0, 0.5, 0.5], np.float32)).astype(np.float32)


def ycbcr_to_rgb(ycc: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_ycbcr (not clipped)."""
    return ((ycc - np.array([0, 0.5, 0.5], np.float32)) @ _YCC_INV.T).astype(np.float32)


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Linear sRGB -> CIE XYZ (D65)."""
    return (rgb @ _RGB2XYZ.T).astype(np.float32)


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    """CIE XYZ (D65) -> linear sRGB (not clipped)."""
    return (xyz @ _XYZ2RGB.T).astype(np.float32)


_EPS, _KAPPA = 216 / 24389, 24389 / 27


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Linear sRGB -> CIE Lab, the perceptual space behind DeltaE."""
    t = rgb_to_xyz(rgb) / _D65
    f = np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE Lab -> linear sRGB (not clipped)."""
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    t = np.where(f ** 3 > _EPS, f ** 3, (116 * f - 16) / _KAPPA)
    return xyz_to_rgb((t * _D65).astype(np.float32))
