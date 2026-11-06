"""Denoising and edge-preserving filtering (F-CV-4).

Sensor noise is Poisson-Gaussian: var(x) = a * x + b (a: shot, b: read noise).
"""
from __future__ import annotations

import cv2
import numpy as np

from isp.classical.bayer import pack_rggb, unpack_rggb
from isp.classical.color import rgb_to_ycbcr, ycbcr_to_rgb

_B = cv2.BORDER_REFLECT
# Immerkaer's Laplacian: kills smooth signal, passes noise with unit gain (sum of squares == 1).
_LAP = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32) / 6
_A_MIN = 1e-4  # keeps the variance-stabilising transform finite when shot noise is ~0


def estimate_noise(bayer: np.ndarray, bins: int = 12) -> tuple[float, float]:
    """Estimate Poisson-Gaussian (a, b) from one RGGB mosaic.

    Why: no calibration needed. A Laplacian high-pass removes the image, the robust
    MAD gives noise variance per brightness bin, and a line fit var = a*mean + b
    separates shot from read noise. Median-based, so edges barely bias it.
    """
    planes = pack_rggb(bayer.astype(np.float32))
    m_all, r_all = [], []
    for c in range(4):
        p = np.ascontiguousarray(planes[..., c])
        mean, res = cv2.blur(p, (5, 5)), cv2.filter2D(p, -1, _LAP, borderType=_B)
        keep = (mean > 0.002) & (mean < 0.95)  # clipped pixels understate the variance
        m_all.append(mean[keep]), r_all.append(res[keep])
    m, r = np.concatenate(m_all), np.concatenate(r_all)
    if m.size < 200:
        return 0.0, 0.0
    idx = np.digitize(m, np.quantile(m, np.linspace(0, 1, bins + 1))[1:-1])
    pts = [(m[idx == i].mean(), (1.4826 * np.median(np.abs(r[idx == i] - np.median(r[idx == i])))) ** 2)
           for i in range(bins) if (idx == i).sum() >= 50]
    return _fit_line(np.array(pts, np.float64))


def _fit_line(pts: np.ndarray) -> tuple[float, float]:
    """Least-squares var = a*mean + b with a, b >= 0."""
    if len(pts) == 0:
        return 0.0, 0.0
    mu, var = pts[:, 0], pts[:, 1]
    if len(pts) == 1 or np.ptp(mu) < 1e-9:
        return 0.0, float(var.mean())
    a, b = np.linalg.lstsq(np.stack([mu, np.ones_like(mu)], 1), var, rcond=None)[0]
    if a < 0:
        return 0.0, float(var.mean())
    if b < 0:
        return float((mu * var).sum() / (mu * mu).sum()), 0.0
    return float(a), float(b)


def anscombe(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Generalised Anscombe transform: makes Poisson-Gaussian noise ~unit-variance Gaussian.

    Why: guided/linear filters assume constant noise. After this transform one
    filter strength is right for dark and bright pixels alike.
    """
    a = max(a, _A_MIN)
    return 2.0 / a * np.sqrt(np.maximum(a * np.asarray(x, np.float64) + 0.375 * a * a + b, 0.0))


def inverse_anscombe(f: np.ndarray, a: float, b: float) -> np.ndarray:
    """Algebraic inverse of `anscombe`."""
    a = max(a, _A_MIN)
    return ((a * f / 2.0) ** 2 - 0.375 * a * a - b) / a


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter (box-filter form), O(N) regardless of radius.

    Why: edge-preserving like bilateral but linear-time and free of gradient reversals;
    the output is a local linear function of the guide, so edges in the guide survive.
    """
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=_B)  # noqa: E731
    g, s = guide.astype(np.float64), src.astype(np.float64)
    mg, ms = box(g), box(s)
    a = (box(g * s) - mg * ms) / (box(g * g) - mg * mg + eps)
    return (box(a) * g + box(ms - a * mg)).astype(src.dtype)


def denoise_raw(bayer: np.ndarray, a: float, b: float, strength: float = 1.0) -> np.ndarray:
    """VST + self-guided filter on each same-colour plane of the RAW mosaic, then inverse VST.

    Why: denoising before demosaicing is cheapest (1 channel) and avoids colour noise
    being spread by interpolation; the VST makes the filter strength signal independent.
    """
    if strength <= 0:
        return bayer.astype(np.float32)
    planes = pack_rggb(bayer.astype(np.float64))
    for c in range(4):
        f = anscombe(planes[..., c], a, b)
        planes[..., c] = inverse_anscombe(guided_filter(f, f, 2, 2.0 * strength ** 2), a, b)
    return np.clip(unpack_rggb(planes), 0, 1).astype(np.float32)


def denoise_luma_chroma(rgb: np.ndarray, sigma: float, strength: float = 1.0) -> np.ndarray:
    """Guided filtering in YCbCr: light on luma (detail), heavy on chroma (colour blotches).

    Why: the eye tolerates blurred colour far more than blurred brightness, so chroma
    noise can be removed aggressively, guided by luma so colour does not bleed over edges.
    """
    ycc = rgb_to_ycbcr(rgb)
    s = max(strength * sigma, 1e-4)
    y = ycc[..., 0]
    out = ycc.copy()
    out[..., 0] = guided_filter(y, y, 2, 2.0 * s * s)
    for c in (1, 2):
        out[..., c] = guided_filter(y, ycc[..., c], 6, (4 * s) ** 2)
    return np.clip(ycbcr_to_rgb(out), 0, 1).astype(np.float32)


def bilateral(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Bilateral filter: average neighbours that are close in space AND in colour.

    Why: the textbook edge-preserving smoother; kept as a comparison for the guided filter.
    """
    img = np.ascontiguousarray(rgb, np.float32)
    return np.clip(cv2.bilateralFilter(img, 7, max(3 * sigma, 1e-3), 3), 0, 1).astype(np.float32)


def nlm(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Non-local means (OpenCV, 8-bit): average patches that look alike anywhere nearby.

    Why: best quality of the classical filters on textures, but the slowest; it is the
    upper-quality reference against which the cheaper guided filter is judged.
    """
    u8 = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    h = float(max(sigma * 255, 1.0))
    out = cv2.fastNlMeansDenoisingColored(u8, None, h, h, 7, 21)
    return out.astype(np.float32) / 255
