import numpy as np
import pytest

from isp.classical.white_balance import (
    apply_gains, fuse_awb, gray_edge, gray_world, shades_of_gray, white_patch)

ESTIMATORS = [gray_world, shades_of_gray, gray_edge, white_patch]


def cast_image(cast=(0.6, 1.0, 1.5), seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((64, 64, 3)).astype(np.float32) * 0.6 + 0.2
    base[..., :] = base.mean(-1, keepdims=True)  # neutral scene
    base[:8, :8] = 0.7                           # a white patch
    return (base * np.array(cast, np.float32)).astype(np.float32)


@pytest.mark.parametrize("fn", ESTIMATORS + [fuse_awb])
def test_neutralises_colour_cast(fn):
    img = cast_image()
    out = apply_gains(img, fn(img))
    means = out.reshape(-1, 3).mean(0)
    assert means.max() / means.min() < 1.1


@pytest.mark.parametrize("fn", ESTIMATORS + [fuse_awb])
def test_gains_have_green_one_and_survive_degenerate(fn):
    g = fn(cast_image())
    assert g[1] == pytest.approx(1.0)
    for v in (0.0, 1.0):
        g = fn(np.full((16, 16, 3), v, np.float32))
        assert np.isfinite(g).all() and (g > 0).all()


def test_fuse_ignores_clipped_white_patch():
    img = cast_image()
    img[:32] = 1.0  # large clipped region ruins white-patch
    g = fuse_awb(img)
    assert np.isfinite(g).all()
    assert g[0] > 1.2  # still detects the red-deficient cast from the rest


def test_apply_gains_clips():
    out = apply_gains(np.full((2, 2, 3), 0.9, np.float32), np.array([2, 1, 1], np.float32))
    assert out.max() <= 1.0
