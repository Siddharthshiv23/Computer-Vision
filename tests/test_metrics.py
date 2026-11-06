"""Metric correctness against analytic or published reference values."""
import builtins
import math

import numpy as np
import pytest

from isp.eval import metrics
from isp.eval.metrics import ciede2000, delta_e2000, lpips_alex, psnr, ssim
from isp.types import InputError

# Sharma, Wu, Dalal (2005), "The CIEDE2000 color-difference formula", Table 1.
SHARMA_PAIRS = [
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461),
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    (50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977),
    (50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030),
    (50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731),
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082),
]


@pytest.fixture
def image():
    return np.random.default_rng(0).random((64, 64, 3), dtype=np.float32)


def flat(value, shape=(32, 32, 3)):
    return np.full(shape, value, dtype=np.float32)


# ---- PSNR ----

def test_psnr_identical_is_infinite(image):
    assert psnr(image, image) == math.inf


@pytest.mark.parametrize("offset, expected", [(0.1, 20.0), (0.01, 40.0), (0.5, 20 * math.log10(2))])
def test_psnr_constant_offset_matches_closed_form(offset, expected):
    # MSE = offset^2 on a peak-1 signal, so PSNR = -20 log10(offset).
    base = flat(0.25)
    assert psnr(base + np.float32(offset), base) == pytest.approx(expected, abs=1e-4)


def test_psnr_does_not_lose_precision_on_float32():
    a = flat(0.5)
    b = a + np.float32(1e-4)
    assert psnr(a, b) == pytest.approx(80.0, abs=0.01)


# ---- SSIM ----

def test_ssim_identical_is_one(image):
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-9)


def test_ssim_flat_images_match_luminance_term():
    # Two flat images have zero variance, so only the luminance term survives:
    # SSIM = (2ab + C1) / (a^2 + b^2 + C1), C1 = (0.01 * 1)^2 (Wang et al., 2004).
    a, b, c1 = 0.2, 0.6, 0.01**2
    expected = (2 * a * b + c1) / (a * a + b * b + c1)
    assert ssim(flat(a), flat(b)) == pytest.approx(expected, abs=1e-4)


def test_ssim_uses_unit_data_range_not_dtype_range(image):
    # A dtype-derived range would make this nearly 1.
    assert ssim(image, 1 - image) < 0.2


# ---- CIEDE2000 ----

@pytest.mark.parametrize("pair", SHARMA_PAIRS, ids=[f"pair{i + 1}" for i in range(34)])
def test_ciede2000_matches_sharma_table(pair):
    lab1, lab2, expected = np.array(pair[:3]), np.array(pair[3:6]), pair[6]
    assert ciede2000(lab1, lab2) == pytest.approx(expected, abs=5e-5)


def test_ciede2000_is_vectorised_over_pixels():
    table = np.array(SHARMA_PAIRS)
    got = ciede2000(table[:, :3], table[:, 3:6])
    assert got.shape == (34,)
    np.testing.assert_allclose(got, table[:, 6], atol=5e-5)


def test_delta_e2000_identical_is_zero(image):
    assert delta_e2000(image, image) == pytest.approx(0.0, abs=1e-9)


def test_delta_e2000_white_vs_black_is_100():
    # sRGB white is L*=100 and black L*=0, both neutral. SL at the mean L=50 is 1, so dE = 100.
    assert delta_e2000(flat(1.0), flat(0.0)) == pytest.approx(100.0, abs=1e-2)


def test_delta_e2000_srgb_is_converted_with_d65_white():
    # Equal sRGB neutrals must stay neutral in Lab; a D50 conversion would leave a colour cast,
    # so grey vs. a slightly different grey is a pure lightness difference.
    def l_star(v):  # IEC 61966-2-1 decode, then CIE 1976 L*
        return 116 * ((v + 0.055) / 1.055) ** (2.4 / 3) - 16

    l1, l2 = l_star(0.5), l_star(0.6)
    mean_l = (l1 + l2) / 2
    sl = 1 + 0.015 * (mean_l - 50) ** 2 / math.sqrt(20 + (mean_l - 50) ** 2)
    assert delta_e2000(flat(0.5), flat(0.6)) == pytest.approx((l2 - l1) / sl, abs=1e-2)


# ---- LPIPS ----

def test_lpips_identical_is_zero(image):
    assert lpips_alex(image, image) == pytest.approx(0.0, abs=1e-6)


def test_lpips_grows_with_distortion(image):
    rng = np.random.default_rng(1)
    mild = np.clip(image + 0.02 * rng.standard_normal(image.shape), 0, 1).astype(np.float32)
    strong = np.clip(image + 0.3 * rng.standard_normal(image.shape), 0, 1).astype(np.float32)
    assert 0 < lpips_alex(mild, image) < lpips_alex(strong, image)


def test_lpips_matches_upstream_model_on_minus_one_to_one_inputs(image):
    # Upstream publishes no numeric reference, so parity with its documented [-1, 1] input
    # convention is the check. Feeding [0, 1] straight in would fail this.
    import lpips
    import torch

    other = np.ascontiguousarray(image[::-1])
    to_tensor = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None] * 2 - 1  # noqa: E731
    with torch.no_grad():
        expected = lpips.LPIPS(net="alex", verbose=False)(to_tensor(image), to_tensor(other)).item()
    assert lpips_alex(image, other) == pytest.approx(expected, rel=1e-5)


def test_lpips_missing_package_raises_clear_error(monkeypatch, image):
    metrics._lpips_model.cache_clear()
    real_import = builtins.__import__

    def no_lpips(name, *args, **kwargs):
        if name == "lpips":
            raise ImportError("no lpips")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_lpips)
    with pytest.raises(metrics.MetricUnavailable, match="LPIPS"):
        lpips_alex(image, image)
    metrics._lpips_model.cache_clear()


def test_lpips_weight_download_failure_raises_clear_error(monkeypatch, image):
    import lpips

    metrics._lpips_model.cache_clear()

    def offline(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(lpips, "LPIPS", offline)
    with pytest.raises(metrics.MetricUnavailable, match="weights"):
        lpips_alex(image, image)
    metrics._lpips_model.cache_clear()


# ---- shared validation ----

@pytest.mark.parametrize("fn", [psnr, ssim, delta_e2000, lpips_alex])
def test_shape_mismatch_raises_input_error(fn):
    with pytest.raises(InputError):
        fn(flat(0.5, (32, 32, 3)), flat(0.5, (32, 34, 3)))


@pytest.mark.parametrize("fn", [psnr, ssim, delta_e2000, lpips_alex])
def test_non_finite_pixels_raise_input_error(fn):
    bad = flat(0.5)
    bad[0, 0, 0] = np.nan
    with pytest.raises(InputError):
        fn(bad, flat(0.5))


@pytest.mark.parametrize("fn", [psnr, ssim, delta_e2000, lpips_alex])
def test_out_of_range_pixels_raise_input_error(fn):
    with pytest.raises(InputError):
        fn(flat(255.0), flat(0.5))


def test_metrics_return_python_floats(image):
    other = 1 - image
    for fn in (psnr, ssim, delta_e2000):
        assert type(fn(image, other)) is float
