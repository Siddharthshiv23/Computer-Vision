"""W4 additions to the classical front end: a CCM off-switch and cheaper statistics (latency)."""
import cv2
import numpy as np

from isp.classical.pipeline import ClassicalConfig, front_end
from isp.types import RawImage


def _noisy_mosaic(h=1000, w=1000, a=0.004, b=2e-5, seed=0):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 25)
    clean = 0.02 + 0.5 * (base - base.min()) / np.ptp(base)
    noisy = rng.poisson(clean / a) * a + rng.normal(0, np.sqrt(b), clean.shape)
    return np.clip(noisy, 0, 1).astype(np.float32)


CCM = np.array([[1.6, -0.4, -0.2], [-0.3, 1.5, -0.2], [-0.1, -0.5, 1.6]], np.float32)


def test_ccm_switch_changes_colour_and_default_keeps_it():
    raw = RawImage(bayer=_noisy_mosaic(64, 64), ccm=CCM)
    on, _, _ = front_end(raw, False)
    off, stages, _ = front_end(raw, False, ClassicalConfig(ccm=False))
    assert np.abs(on - off).max() > 0.02
    assert np.array_equal(on, front_end(raw, False, ClassicalConfig(ccm=True))[0])


def test_fast_stats_matches_full_stats_on_a_large_frame():
    raw = RawImage(bayer=_noisy_mosaic())
    _, _, full = front_end(raw, False)
    rgb_f, _, fast = front_end(raw, False, ClassicalConfig(fast_stats=True))
    assert abs(fast["noise_a"] / full["noise_a"] - 1) < 0.15
    assert abs(fast["noise_b"] - full["noise_b"]) < 5e-5
    assert abs(fast["exposure"] / full["exposure"] - 1) < 0.02
    rgb_full, _, _ = front_end(raw, False)
    assert np.abs(rgb_f - rgb_full).mean() < 0.01  # only the AWB gains may move


def test_fast_stats_is_a_noop_on_small_frames():
    raw = RawImage(bayer=_noisy_mosaic(64, 64))
    a = front_end(raw, False)[0]
    b = front_end(raw, False, ClassicalConfig(fast_stats=True))[0]
    assert np.array_equal(a, b) or np.abs(a - b).max() < 0.02
