import numpy as np
import pytest

from isp.classical.bayer import mosaic_rggb
from isp.classical.denoise import (
    anscombe, bilateral, denoise_luma_chroma, denoise_raw, estimate_noise, guided_filter,
    inverse_anscombe, nlm)


def clean_bayer(h=96, w=96):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([0.2 + 0.6 * x / w, 0.2 + 0.6 * y / h, 0.5 + 0 * x], -1)
    return mosaic_rggb(rgb.astype(np.float32))


def noisy(clean, a, b, seed=1):
    rng = np.random.default_rng(seed)
    n = clean + rng.normal(size=clean.shape) * np.sqrt(a * clean + b)
    return np.clip(n, 0, 1).astype(np.float32)


def test_estimate_noise_recovers_parameters():
    clean = clean_bayer(192, 192)
    a, b = estimate_noise(noisy(clean, 0.01, 1e-4))
    assert 0.005 < a < 0.02
    # compare predicted variance at mid-grey rather than a and b separately
    assert a * 0.5 + b == pytest.approx(0.01 * 0.5 + 1e-4, rel=0.4)


def test_estimate_noise_clean_is_small_and_degenerate_frames_finite():
    a, b = estimate_noise(clean_bayer())
    assert a < 1e-3 and b < 1e-4
    for v in (0.0, 1.0):
        a, b = estimate_noise(np.full((32, 32), v, np.float32))
        assert np.isfinite([a, b]).all() and a >= 0 and b >= 0


def test_anscombe_roundtrip_and_stabilises_variance():
    x = np.linspace(0, 1, 50)
    assert np.allclose(inverse_anscombe(anscombe(x, 0.01, 1e-4), 0.01, 1e-4), x, atol=1e-6)
    rng = np.random.default_rng(0)
    sds = []
    for level in (0.1, 0.5, 0.9):
        n = level + rng.normal(size=20000) * np.sqrt(0.01 * level + 1e-4)
        sds.append(anscombe(n, 0.01, 1e-4).std())
    assert max(sds) / min(sds) < 1.15


def test_guided_filter_smooths_noise_keeps_step():
    rng = np.random.default_rng(0)
    img = np.zeros((64, 64), np.float32)
    img[:, 32:] = 1.0
    n = img + rng.normal(size=img.shape).astype(np.float32) * 0.1
    out = guided_filter(n, n, 4, 0.02)
    assert out[:, :24].std() < 0.5 * n[:, :24].std()
    assert out[:, 36:].mean() - out[:, :28].mean() > 0.9


def test_denoise_raw_improves_mse_and_is_bounded():
    clean = clean_bayer(128, 128)
    n = noisy(clean, 0.01, 1e-4)
    out = denoise_raw(n, 0.01, 1e-4, strength=1.0)
    assert out.shape == n.shape and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1
    assert np.mean((out - clean) ** 2) < 0.6 * np.mean((n - clean) ** 2)


def test_denoise_raw_black_and_white_finite():
    for v in (0.0, 1.0):
        out = denoise_raw(np.full((16, 16), v, np.float32), 0.0, 0.0, 1.0)
        assert np.isfinite(out).all()


@pytest.mark.parametrize("fn", [
    lambda x: denoise_luma_chroma(x, 0.05, 1.0), lambda x: bilateral(x, 0.05), lambda x: nlm(x, 0.05)])
def test_rgb_denoisers_reduce_noise(fn):
    rng = np.random.default_rng(0)
    clean = np.full((48, 48, 3), 0.5, np.float32)
    clean[:, 24:] = 0.3
    n = np.clip(clean + rng.normal(size=clean.shape) * 0.05, 0, 1).astype(np.float32)
    out = fn(n)
    assert out.shape == n.shape and out.dtype == np.float32
    assert np.mean((out - clean) ** 2) < 0.6 * np.mean((n - clean) ** 2)
