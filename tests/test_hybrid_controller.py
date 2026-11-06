import math

import pytest

from isp.hybrid.controller import AdaptiveController
from isp.types import InputError

# Raw-domain (a, b): shot-noise slope and read-noise variance. Typical low/high ISO values.
CLEAN = dict(noise_a=0.0002, noise_b=1e-6, exposure=1.0)
NOISY = dict(noise_a=0.01, noise_b=2e-4, exposure=4.0)


def test_decision_has_all_fields():
    d = AdaptiveController().decide(**NOISY, megapixels=1.0)
    assert set(d) >= {"denoise_strength", "sharpen_amount", "model", "tiling", "noise_sigma"}
    assert d["model"] in {"student", "teacher", "classical"}
    assert set(d["tiling"]) == {"enabled", "tile", "overlap"}
    assert 0.0 <= d["denoise_strength"] <= 1.0


def test_noisier_input_gets_more_denoising_and_less_sharpening():
    c = AdaptiveController()
    clean, noisy = c.decide(**CLEAN, megapixels=1.0), c.decide(**NOISY, megapixels=1.0)
    assert noisy["noise_sigma"] > clean["noise_sigma"]
    assert noisy["denoise_strength"] >= clean["denoise_strength"]
    assert noisy["sharpen_amount"] < clean["sharpen_amount"]


def test_noise_sigma_uses_output_sigma_from_the_classical_module():
    from isp.classical.pipeline import output_sigma

    d = AdaptiveController().decide(**NOISY, megapixels=1.0)
    assert d["noise_sigma"] == pytest.approx(output_sigma(**NOISY))


def test_large_images_are_tiled_small_are_not():
    c = AdaptiveController()
    assert not c.decide(**NOISY, megapixels=1.0)["tiling"]["enabled"]
    big = c.decide(**NOISY, megapixels=12.0)["tiling"]
    assert big["enabled"] and big["overlap"] > 0 and big["tile"] > 2 * big["overlap"]


def test_model_choice_clean_uses_classical_extreme_noise_uses_teacher_when_small():
    c = AdaptiveController()
    assert c.decide(**CLEAN, megapixels=1.0)["model"] == "classical"
    extreme = dict(noise_a=0.05, noise_b=2e-3, exposure=8.0)
    assert c.decide(**extreme, megapixels=1.0)["model"] == "teacher"
    assert c.decide(**extreme, megapixels=12.0)["model"] == "student"  # teacher too slow when tiled
    assert c.decide(**NOISY, megapixels=1.0)["model"] == "student"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_unusable_noise_estimate_falls_back_to_classical(bad):
    d = AdaptiveController().decide(noise_a=bad, noise_b=1e-4, exposure=1.0, megapixels=1.0)
    assert d["model"] == "classical"
    assert math.isfinite(d["sharpen_amount"]) and 0 <= d["denoise_strength"] <= 1


def test_rejects_non_positive_size():
    with pytest.raises(InputError):
        AdaptiveController().decide(**NOISY, megapixels=0.0)


def test_thresholds_are_configurable():
    strict = AdaptiveController(clean_sigma=1.0)  # everything counts as clean
    assert strict.decide(**NOISY, megapixels=1.0)["model"] == "classical"
