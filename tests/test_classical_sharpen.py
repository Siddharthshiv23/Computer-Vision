import cv2
import numpy as np

from isp.classical.sharpen import noise_aware_unsharp


def test_sharpens_edges():
    img = np.full((32, 32, 3), 0.3, np.float32)
    img[:, 16:] = 0.7
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    out = noise_aware_unsharp(img, 0.001, 1.0)
    assert np.abs(np.diff(out[16, :, 0])).max() > np.abs(np.diff(img[16, :, 0])).max()


def test_does_not_amplify_noise():
    rng = np.random.default_rng(0)
    n = np.clip(0.5 + 0.02 * rng.standard_normal((64, 64, 3)), 0, 1).astype(np.float32)
    naive = noise_aware_unsharp(n, 0.0, 1.0)   # no coring
    aware = noise_aware_unsharp(n, 0.02, 1.0)  # coring at the true sigma
    assert aware.std() < naive.std()
    assert aware.std() < 1.2 * n.std()


def test_zero_amount_identity_and_range():
    x = np.random.default_rng(1).random((16, 16, 3)).astype(np.float32)
    assert np.allclose(noise_aware_unsharp(x, 0.01, 0.0), x)
    out = noise_aware_unsharp(x, 0.01, 3.0)
    assert out.min() >= 0 and out.max() <= 1 and out.dtype == np.float32
