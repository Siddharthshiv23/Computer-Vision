import numpy as np
import pytest

from isp.classical.tone import auto_exposure, clahe, gamma22, reinhard, s_curve, srgb_gamma


def test_auto_exposure_brightens_dark_and_darkens_bright():
    _, g_dark = auto_exposure(np.full((16, 16, 3), 0.02, np.float32))
    _, g_bright = auto_exposure(np.full((16, 16, 3), 0.95, np.float32))
    assert g_dark > 1 > g_bright


def test_auto_exposure_hits_target_and_is_bounded():
    out, g = auto_exposure(np.full((16, 16, 3), 0.09, np.float32), target=0.18)
    assert g == pytest.approx(2.0, rel=0.05) and out.mean() == pytest.approx(0.18, rel=0.05)
    out, g = auto_exposure(np.zeros((8, 8, 3), np.float32))
    assert np.isfinite(out).all() and np.isfinite(g) and g <= 16
    out, g = auto_exposure(np.ones((8, 8, 3), np.float32))
    assert np.isfinite(out).all() and out.max() <= 1.0


def test_gamma_functions():
    x = np.linspace(0, 1, 11, dtype=np.float32)
    assert np.allclose(gamma22(x), x ** (1 / 2.2), atol=1e-6)
    assert srgb_gamma(np.float32(0.18)) == pytest.approx(0.4614, abs=1e-3)


def test_s_curve_monotonic_fixes_endpoints_and_adds_contrast():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    y = s_curve(x, 0.5)
    assert y[0] == 0 and y[-1] == pytest.approx(1.0) and np.all(np.diff(y) > 0)
    assert y[25] < x[25] and y[75] > x[75]
    assert np.array_equal(s_curve(x, 0.0), x)


def test_reinhard_compresses_highlights_keeps_hue():
    out = reinhard(np.array([[[4.0, 2.0, 1.0]]], np.float32))
    assert out.max() < 1.0 + 1e-6 and out[0, 0, 0] > out[0, 0, 1] > out[0, 0, 2]
    assert np.isfinite(reinhard(np.zeros((2, 2, 3), np.float32))).all()


def test_clahe_raises_local_contrast():
    rng = np.random.default_rng(0)
    x = (0.45 + 0.03 * rng.standard_normal((64, 64, 1)) * np.ones(3)).clip(0, 1).astype(np.float32)
    out = clahe(x, clip=3.0)
    assert out.shape == x.shape and out.dtype == np.float32
    assert out.std() > 1.5 * x.std()
