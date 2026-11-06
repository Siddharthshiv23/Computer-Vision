"""Full-reference image metrics. pred and target are float (H, W, 3) sRGB in [0, 1]."""
from __future__ import annotations

import functools
import math

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from isp.types import InputError


class MetricUnavailable(RuntimeError):
    """A metric cannot be computed here. Never substitute a value for it."""


def _validated(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[2] != 3:
        raise InputError(f"Images must both be (H, W, 3); got {pred.shape} and {target.shape}.")
    for label, image in (("Prediction", pred), ("Target", target)):
        if not np.isfinite(image).all():
            raise InputError(f"{label} contains NaN or infinite pixels.")
        if image.min() < 0 or image.max() > 1:
            raise InputError(f"{label} must be in [0, 1]; got [{image.min():.3g}, {image.max():.3g}].")
    return pred, target


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB for peak 1. Identical images give infinity."""
    pred, target = _validated(pred, target)
    mse = float(np.mean((pred - target) ** 2))
    return math.inf if mse == 0 else -10.0 * math.log10(mse)


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean SSIM over channels with the settings of Wang et al. (2004)."""
    pred, target = _validated(pred, target)
    return float(
        structural_similarity(
            pred,
            target,
            data_range=1.0,
            channel_axis=2,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Per-pixel CIEDE2000 between Lab arrays with L, a, b on the last axis."""
    return deltaE_ciede2000(lab1, lab2)


def delta_e2000(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean CIEDE2000 after converting sRGB to CIE Lab (D65)."""
    pred, target = _validated(pred, target)
    return float(np.mean(ciede2000(rgb2lab(pred), rgb2lab(target))))


@functools.lru_cache(maxsize=1)
def _lpips_model():
    try:
        import lpips
    except ImportError as exc:
        raise MetricUnavailable("LPIPS needs the 'lpips' package: pip install lpips") from exc
    try:
        return lpips.LPIPS(net="alex", verbose=False).eval()
    except (OSError, RuntimeError) as exc:  # urllib's URLError is an OSError
        raise MetricUnavailable(
            "LPIPS weights are unavailable: the AlexNet backbone could not be loaded or downloaded. "
            f"Connect to the network or fill the torch hub cache, then rerun. Cause: {exc}"
        ) from exc


def lpips_alex(pred: np.ndarray, target: np.ndarray) -> float:
    """LPIPS v0.1 with AlexNet, on CPU."""
    import torch

    pred, target = _validated(pred, target)
    model = _lpips_model()

    def to_tensor(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)[None] * 2 - 1

    with torch.no_grad():
        return float(model(to_tensor(pred), to_tensor(target)).item())
