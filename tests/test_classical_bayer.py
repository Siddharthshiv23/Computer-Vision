import numpy as np
import pytest

from isp.classical.bayer import mosaic_rggb, normalize_cfa, pack_rggb, unpack_rggb
from isp.types import InputError


def test_pack_unpack_roundtrip():
    x = np.random.default_rng(0).random((8, 6)).astype(np.float32)
    assert np.array_equal(unpack_rggb(pack_rggb(x)), x)
    assert pack_rggb(x).shape == (4, 3, 4)


def test_mosaic_samples_rggb_sites():
    rgb = np.zeros((4, 4, 3), np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.1, 0.5, 0.9
    m = mosaic_rggb(rgb)
    assert m[0, 0] == pytest.approx(0.1) and m[0, 1] == pytest.approx(0.5)
    assert m[1, 0] == pytest.approx(0.5) and m[1, 1] == pytest.approx(0.9)


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_normalize_cfa_gives_rggb(pattern):
    rgb = np.zeros((10, 10, 3), np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.2, 0.5, 0.8
    layout = {"RGGB": [[0, 1], [1, 2]], "BGGR": [[2, 1], [1, 0]],
              "GRBG": [[1, 0], [2, 1]], "GBRG": [[1, 2], [0, 1]]}[pattern]
    ch = np.tile(np.array(layout), (5, 5))
    mosaic = np.take_along_axis(rgb, ch[..., None], axis=2)[..., 0]
    out = normalize_cfa(mosaic, pattern, 0.0, 1.0)
    assert out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0
    assert out[0, 0] == pytest.approx(0.2) and out[1, 1] == pytest.approx(0.8)
    assert out[0, 1] == pytest.approx(0.5)


def test_normalize_cfa_levels_and_clip():
    m = np.array([[100, 200], [300, 4000]], np.float32)
    out = normalize_cfa(m, "RGGB", 100.0, 1100.0)
    assert out.min() == 0.0 and out.max() == 1.0
    assert out[0, 1] == pytest.approx(0.1)


def test_normalize_cfa_rejects_bad_input():
    with pytest.raises(InputError):
        normalize_cfa(np.zeros((4, 4, 3)), "RGGB", 0, 1)
    with pytest.raises(InputError):
        normalize_cfa(np.zeros((4, 4)), "XTRN", 0, 1)
    with pytest.raises(InputError):
        normalize_cfa(np.zeros((4, 4)), "RGGB", 5, 5)
