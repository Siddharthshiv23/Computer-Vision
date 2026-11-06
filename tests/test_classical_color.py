import numpy as np
import pytest

from isp.classical.color import (
    DEFAULT_CCM, apply_ccm, lab_to_rgb, resolve_ccm, rgb_to_lab, rgb_to_xyz, rgb_to_ycbcr,
    srgb_decode, srgb_encode, xyz_to_rgb, ycbcr_to_rgb)
from isp.types import InputError


def rand_rgb(seed=0):
    return np.random.default_rng(seed).random((8, 8, 3)).astype(np.float32)


def test_srgb_roundtrip_and_known_points():
    x = rand_rgb()
    assert np.allclose(srgb_decode(srgb_encode(x)), x, atol=1e-5)
    assert srgb_encode(np.float32(0.0)) == 0 and srgb_encode(np.float32(1.0)) == pytest.approx(1.0)
    assert srgb_encode(np.float32(0.18)) == pytest.approx(0.4614, abs=1e-3)


def test_ycbcr_roundtrip_and_neutral_chroma():
    x = rand_rgb()
    assert np.allclose(ycbcr_to_rgb(rgb_to_ycbcr(x)), x, atol=1e-5)
    gray = rgb_to_ycbcr(np.full((2, 2, 3), 0.4, np.float32))
    assert np.allclose(gray[..., 1:], 0.5, atol=1e-5) and np.allclose(gray[..., 0], 0.4, atol=1e-5)


def test_xyz_white_is_d65():
    xyz = rgb_to_xyz(np.ones((1, 1, 3), np.float32))[0, 0]
    assert np.allclose(xyz, [0.9505, 1.0, 1.089], atol=2e-3)
    x = rand_rgb()
    assert np.allclose(xyz_to_rgb(rgb_to_xyz(x)), x, atol=1e-4)


def test_lab_white_and_roundtrip():
    lab = rgb_to_lab(np.ones((1, 1, 3), np.float32))[0, 0]
    assert lab[0] == pytest.approx(100, abs=0.1) and abs(lab[1]) < 0.2 and abs(lab[2]) < 0.2
    x = rand_rgb()
    assert np.allclose(lab_to_rgb(rgb_to_lab(x)), x, atol=1e-3)


def test_default_ccm_preserves_white():
    assert np.allclose(DEFAULT_CCM.sum(1), 1.0, atol=1e-6)
    out = apply_ccm(np.full((2, 2, 3), 0.5, np.float32), DEFAULT_CCM)
    assert np.allclose(out, 0.5, atol=1e-5)


def test_resolve_ccm():
    assert np.array_equal(resolve_ccm(None), DEFAULT_CCM)
    m = np.array([[2, -1, 0], [0, 2, -1], [-1, 0, 2.0]])  # rows sum to 1
    assert np.allclose(resolve_ccm(m), m)
    assert np.allclose(resolve_ccm(m * 2).sum(1), 1.0)  # rows normalised to sum 1
    with pytest.raises(InputError):
        resolve_ccm(np.eye(4))
    with pytest.raises(InputError):
        resolve_ccm(np.full((3, 3), np.nan))


def test_apply_ccm_clips():
    out = apply_ccm(rand_rgb(), DEFAULT_CCM)
    assert out.min() >= 0 and out.max() <= 1
