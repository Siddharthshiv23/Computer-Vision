import numpy as np
import torch

from isp.learned.inputs import estimate_bayer_sigma, liteisp_input, pack_bayer, rawnet_input
from isp.types import RawImage


def test_pack_bayer_rggb_order():
    bayer = np.arange(16, dtype=np.float32).reshape(4, 4)
    packed = pack_bayer(bayer)
    assert packed.shape == (4, 2, 2)
    assert packed[0, 0, 0] == bayer[0, 0] and packed[1, 0, 0] == bayer[0, 1]
    assert packed[2, 0, 0] == bayer[1, 0] and packed[3, 1, 1] == bayer[3, 3]


def test_sigma_estimate_tracks_noise_and_is_finite_on_flat_frames():
    rng = np.random.default_rng(0)
    clean = np.full((64, 64), 0.3, np.float32)
    s_lo = estimate_bayer_sigma(clean + rng.normal(0, 0.01, clean.shape).astype(np.float32))
    s_hi = estimate_bayer_sigma(clean + rng.normal(0, 0.05, clean.shape).astype(np.float32))
    assert 0.005 < s_lo < 0.02 and s_hi > 3 * s_lo
    assert estimate_bayer_sigma(np.zeros((8, 8), np.float32)) == 0.0


def test_rawnet_input_shape():
    raw = RawImage(bayer=np.random.rand(16, 24).astype(np.float32))
    x = rawnet_input(raw)
    assert x.shape == (5, 8, 12) and x.dtype == torch.float32 and torch.isfinite(x).all()


def test_liteisp_input_shape_and_nonneg_sigma():
    rgb = np.random.rand(10, 12, 3).astype(np.float32)
    x = liteisp_input(rgb, noise_a=0.01, noise_b=1e-4)
    assert x.shape == (4, 10, 12)
    assert (x[3] >= 0).all() and torch.allclose(x[:3], torch.from_numpy(rgb).permute(2, 0, 1))


def test_liteisp_sigma_grows_with_exposure_gain():
    """Gain amplifies noise, so a dark frame pushed 8x must report more noise than one left at 1x."""
    rgb = np.full((8, 8, 3), 0.5, np.float32)
    base = liteisp_input(rgb, 0.01, 1e-4, exposure=1.0)[3].mean()
    gained = liteisp_input(rgb, 0.01, 1e-4, exposure=8.0)[3].mean()
    assert gained > 2 * base
