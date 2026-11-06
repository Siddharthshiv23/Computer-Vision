"""Exposure, gamma and tone mapping (F-CV-5)."""
from __future__ import annotations

import cv2
import numpy as np

from isp.classical.color import rgb_to_ycbcr, srgb_encode, ycbcr_to_rgb

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def auto_exposure(lin: np.ndarray, target: float = 0.18, max_gain: float = 16.0,
                  min_gain: float = 0.25) -> tuple[np.ndarray, float]:
    """Scale linear RGB so the log-average luminance hits mid-grey (0.18).

    Why: the geometric mean is the standard 'key' of a scene and is robust to a few
    bright pixels. Gain is bounded (and limited so only the top 1% may clip) so
    black or saturated frames stay finite and highlights are not blown out.
    """
    lum = lin @ _LUMA
    gain = target / float(np.exp(np.mean(np.log(lum + 1e-4))))
    p99 = float(np.percentile(lum, 99))
    if p99 > 1e-3:
        gain = min(gain, 2.0 / p99)
    gain = float(np.clip(gain, min_gain, max_gain))
    return np.clip(lin * gain, 0, 1).astype(np.float32), gain


def srgb_gamma(lin: np.ndarray) -> np.ndarray:
    """Exact sRGB transfer curve (linear toe + 2.4 power)."""
    return srgb_encode(lin)


def gamma22(lin: np.ndarray) -> np.ndarray:
    """Plain 1/2.2 power law, the cheap approximation of sRGB."""
    return np.power(np.clip(lin, 0, 1), 1 / 2.2).astype(np.float32)


def s_curve(x: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Blend identity with smoothstep (3x^2 - 2x^3); strength 0..1.

    Why: one parameter, always monotonic, pins black and white, adds the mid-tone
    contrast that the plain gamma curve lacks.
    """
    x = np.clip(x, 0, 1).astype(np.float32)
    return (x + strength * (x * x * (3 - 2 * x) - x)).astype(np.float32)


def reinhard(lin: np.ndarray, white: float = 4.0) -> np.ndarray:
    """Extended Reinhard operator on luminance: L (1 + L/w^2) / (1 + L).

    Why: compresses HDR highlights smoothly; scaling RGB by the luminance ratio keeps hue.
    """
    lum = lin @ _LUMA
    mapped = lum * (1 + lum / white ** 2) / (1 + lum)
    scale = mapped / np.maximum(lum, 1e-6)
    return np.clip(lin * scale[..., None], 0, 1).astype(np.float32)


def clahe(rgb: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on luma of a display-referred image.

    Why: local contrast boost for flat, hazy or under-exposed shots; the clip limit
    stops it amplifying noise. Only the CLAHE change is added back to full-precision
    luma, so 8-bit quantisation does not leave banding.
    """
    ycc = rgb_to_ycbcr(np.clip(rgb, 0, 1))
    y8 = np.round(ycc[..., 0] * 255).astype(np.uint8)
    eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(y8)
    ycc[..., 0] = np.clip(ycc[..., 0] + (eq.astype(np.float32) - y8) / 255, 0, 1)
    return np.clip(ycbcr_to_rgb(ycc), 0, 1).astype(np.float32)
