import numpy as np
import pytest

from isp.classical.bayer import mosaic_rggb
from isp.classical.demosaic import demosaic, demosaic_bilinear, demosaic_malvar, demosaic_opencv_ea


def smooth_rgb(h=64, w=64):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.5 + 0.3 * np.sin(x / 15)
    g = 0.5 + 0.3 * np.cos(y / 12)
    b = 0.3 + 0.4 * (x + y) / (h + w)
    return np.clip(np.stack([r, g, b], -1), 0, 1).astype(np.float32)


def psnr_inner(a, b, m=6):
    d = (a[m:-m, m:-m] - b[m:-m, m:-m]) ** 2
    return -10 * np.log10(d.mean() + 1e-12)


@pytest.mark.parametrize("fn", [demosaic_bilinear, demosaic_malvar, demosaic_opencv_ea])
def test_flat_colour_recovered(fn):
    rgb = np.empty((16, 16, 3), np.float32)
    rgb[...] = (0.2, 0.5, 0.8)
    out = fn(mosaic_rggb(rgb))
    assert out.shape == (16, 16, 3) and out.dtype == np.float32
    assert np.allclose(out[4:-4, 4:-4], rgb[4:-4, 4:-4], atol=0.01)


@pytest.mark.parametrize("fn", [demosaic_bilinear, demosaic_malvar, demosaic_opencv_ea])
def test_smooth_image_accurate(fn):
    rgb = smooth_rgb()
    assert psnr_inner(fn(mosaic_rggb(rgb)), rgb) > 35


def test_malvar_beats_bilinear_on_edges():
    # Malvar assumes channels are correlated (true of real edges): a brightness step, same hue.
    rgb = smooth_rgb()
    rgb[:, 31:] *= 0.4
    m = mosaic_rggb(rgb)
    assert psnr_inner(demosaic_malvar(m), rgb) > psnr_inner(demosaic_bilinear(m), rgb)


def test_dispatch_and_unknown():
    m = mosaic_rggb(smooth_rgb(16, 16))
    assert np.array_equal(demosaic(m, "malvar"), demosaic_malvar(m))
    with pytest.raises(ValueError):
        demosaic(m, "nope")


def test_black_and_white_frames_finite():
    for v in (0.0, 1.0):
        out = demosaic_malvar(np.full((8, 8), v, np.float32))
        assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
