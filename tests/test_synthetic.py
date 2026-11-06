"""Synthetic sensor model. Expected values are derived by hand from the Brooks et al. 2019 formulas."""
from dataclasses import replace

import numpy as np
import pytest

from isp.data.synthetic import (
    TEST_CONDITIONS as CONFIG,
    SensorCondition,
    load_test_conditions,
    sample_condition,
    synthesize,
    unprocess_jpeg,
)
from isp.eval import harness
from isp.types import InputError, RawImage

IDENTITY = np.eye(3)
NEUTRAL = np.array([0.5, 1.0, 0.6])


def _cond(**kw):
    base = dict(iso_a=0.0, iso_b=0.0, ev=0.0, illuminant=NEUTRAL, ccm=IDENTITY)
    return SensorCondition(**{**base, **kw})


def _flat(value, h=64, w=64):
    return np.full((h, w, 3), value, np.float32)


def _gradient(h=64, w=96):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([x / (w - 1), y / (h - 1), 1 - x / (w - 1)], axis=-1) * 0.8 + 0.1


def test_mid_grey_lands_on_hand_computed_bayer_values():
    # inverse smoothstep(0.5) = 0.5; 0.5 ** 2.2 = 0.21764; one stop down; RGGB sites scaled by illuminant.
    raw, _ = synthesize(_flat(0.5), _cond(ev=-1.0))
    assert raw.bayer[0, 0] == pytest.approx(0.21764 * 0.5 * 0.5, abs=2e-4)
    assert raw.bayer[0, 1] == pytest.approx(0.21764 * 1.0 * 0.5, abs=2e-4)
    assert raw.bayer[1, 0] == pytest.approx(0.21764 * 1.0 * 0.5, abs=2e-4)
    assert raw.bayer[1, 1] == pytest.approx(0.21764 * 0.6 * 0.5, abs=2e-4)


def test_target_is_the_clean_input_at_normal_exposure():
    rgb = _gradient()
    _, target = synthesize(rgb, _cond(ev=-3.0, iso_a=0.01, iso_b=1e-4))
    np.testing.assert_allclose(target, rgb, atol=1e-6)
    assert target.dtype == np.float32


def test_true_forward_isp_recovers_the_input():
    """Undo every step with an ideal ISP (no demosaic: sample the sites each channel sits on)."""
    rng = np.random.default_rng(1)
    cond = replace(sample_condition(rng), iso_a=0.0, iso_b=0.0, ev=-1.0)
    rgb = _flat(0.0, 64, 64) + np.array([0.3, 0.5, 0.7], np.float32)  # flat colour: every site sees the same
    raw, _ = synthesize(rgb, cond)
    cam = np.array([raw.bayer[0, 0], raw.bayer[0, 1], raw.bayer[1, 1]]) / 2.0**cond.ev / cond.illuminant
    lin = cond.ccm @ cam
    smooth = np.clip(lin, 0, 1) ** (1 / 2.2)
    srgb = 3 * smooth**2 - 2 * smooth**3
    np.testing.assert_allclose(srgb, [0.3, 0.5, 0.7], atol=0.01)


def test_output_obeys_the_rawimage_contract():
    raw, target = synthesize(_gradient(), sample_condition(np.random.default_rng(0)))
    assert isinstance(raw, RawImage)
    assert raw.bayer.dtype == np.float32 and raw.bayer.shape == (64, 96)
    assert raw.bayer.min() >= 0 and raw.bayer.max() <= 1
    assert target.shape == (64, 96, 3)


def test_bayer_is_quantised_to_12_bits():
    raw, _ = synthesize(_gradient(), _cond(iso_a=0.001, iso_b=1e-5))
    codes = raw.bayer.astype(np.float64) * 4095
    np.testing.assert_allclose(codes, np.round(codes), atol=1e-3)


def test_noise_variance_follows_poisson_gaussian_model():
    a, b = 0.004, 2e-4
    raw, _ = synthesize(_flat(0.5, 512, 512), _cond(iso_a=a, iso_b=b, illuminant=np.ones(3)))
    x = 0.5**2.2
    assert raw.bayer.mean() == pytest.approx(x, rel=0.01)
    assert raw.bayer.var() == pytest.approx(a * x + b, rel=0.05)


def test_shot_noise_is_signal_dependent():
    dark, _ = synthesize(_flat(0.1, 256, 256), _cond(iso_a=0.01, illuminant=np.ones(3)))
    bright, _ = synthesize(_flat(0.7, 256, 256), _cond(iso_a=0.01, illuminant=np.ones(3)))
    assert bright.bayer.var() > 3 * dark.bayer.var()


def test_same_condition_seed_reproduces_the_same_noise():
    cond = _cond(iso_a=0.01, iso_b=1e-4, seed=5)
    first, _ = synthesize(_gradient(), cond)
    second, _ = synthesize(_gradient(), cond)
    np.testing.assert_array_equal(first.bayer, second.bayer)


def test_different_noise_seed_gives_different_noise():
    a, _ = synthesize(_gradient(), _cond(iso_a=0.01, iso_b=1e-4, seed=1))
    b, _ = synthesize(_gradient(), _cond(iso_a=0.01, iso_b=1e-4, seed=2))
    assert not np.array_equal(a.bayer, b.bayer)


def test_numpy_global_seed_controls_noise_when_the_condition_has_no_seed():
    cond = _cond(iso_a=0.01, iso_b=1e-4)
    np.random.seed(1)
    first, _ = synthesize(_gradient(), cond)
    np.random.seed(1)
    again, _ = synthesize(_gradient(), cond)
    np.random.seed(2)
    other, _ = synthesize(_gradient(), cond)
    np.testing.assert_array_equal(first.bayer, again.bayer)
    assert not np.array_equal(first.bayer, other.bayer)


def test_evaluation_harness_gets_different_noise_per_image_and_condition():
    # W5 calls the real synthesize through these helpers. Identical images must not share a noise field.
    config = harness.load_config(CONFIG)
    grey = _flat(0.5, 64, 64)
    samples = harness.make_samples({"a": grey, "b": grey}, config, harness.synthesize_fixed)
    assert not np.array_equal(samples[0].raw.bayer, samples[1].raw.bayer)
    again = harness.make_samples({"a": grey, "b": grey}, config, harness.synthesize_fixed)
    np.testing.assert_array_equal(samples[0].raw.bayer, again[0].raw.bayer)
    noisier = harness.make_samples({"a": grey}, config, harness.synthesize_fixed, iso="iso_6400")
    assert noisier[0].raw.bayer.std() > samples[0].raw.bayer.std()


@pytest.mark.parametrize("level", ["ev_-1", "ev_-2", "ev_-3"])
def test_underexposed_target_stays_the_clean_normally_exposed_image(level):
    config = harness.load_config(CONFIG)
    rgb = _gradient()
    sample = harness.make_samples({"a": rgb}, config, harness.synthesize_fixed, ev=level)[0]
    normal = harness.make_samples({"a": rgb}, config, harness.synthesize_fixed)[0]
    np.testing.assert_array_equal(sample.target, rgb)
    assert sample.raw.bayer.mean() < normal.raw.bayer.mean()


def test_training_range_covers_every_test_condition():
    # A model must not meet the headline condition for the first time at test.
    config = harness.load_config(CONFIG)
    rng = np.random.default_rng(8)
    conds = [sample_condition(rng) for _ in range(4000)]
    a = [c.iso_a for c in conds]
    red = [c.illuminant[0] for c in conds]
    blue = [c.illuminant[2] for c in conds]
    for level in config["axes"]["iso"].values():
        assert min(a) <= level["iso_a"] <= max(a)
    for level in config["axes"]["cast"].values():
        r, g, b = level["illuminant"]
        assert min(red) <= r / g <= max(red)
        assert min(blue) <= b / g <= max(blue)


def test_highlights_stay_bright_under_wb_gains():
    # Brooks' safe gains: pure white must not turn the illuminant's colour before exposure.
    raw, _ = synthesize(_flat(1.0), _cond())
    assert raw.bayer[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert raw.bayer[1, 1] == pytest.approx(1.0, abs=1e-3)


def test_exposure_also_dims_highlights():
    raw, _ = synthesize(_flat(1.0), _cond(ev=-3.0))
    assert raw.bayer.max() == pytest.approx(0.125, abs=1e-3)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_black_and_saturated_frames_stay_finite(value):
    raw, target = synthesize(_flat(value), sample_condition(np.random.default_rng(3)))
    assert np.isfinite(raw.bayer).all() and np.isfinite(target).all()


def test_ground_truth_is_kept_out_of_the_as_shot_fields():
    # White balance errors are part of the task, so the pipeline must estimate the illuminant itself.
    cond = sample_condition(np.random.default_rng(4))
    raw, _ = synthesize(_gradient(), cond)
    assert raw.wb_gains is None
    np.testing.assert_allclose(raw.ccm, cond.ccm, rtol=1e-6)
    assert raw.meta["truth"]["noise_a"] == cond.iso_a
    assert raw.meta["truth"]["ev"] == cond.ev


def test_uint8_input_is_scaled():
    raw_u8, target = synthesize(np.full((64, 64, 3), 255, np.uint8), _cond())
    assert target.max() == 1.0
    assert raw_u8.bayer.max() == pytest.approx(1.0, abs=1e-3)


def test_odd_sizes_are_cropped_even_for_raw_and_target_alike():
    raw, target = synthesize(_flat(0.5, 65, 67), _cond())
    assert raw.bayer.shape == (64, 66)
    assert target.shape == (64, 66, 3)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((16, 64, 3), np.float32),  # tiny
        np.zeros((64, 64), np.float32),  # no channels
        np.zeros((64, 64, 4), np.float32),  # RGBA
        np.full((64, 64, 3), np.nan, np.float32),  # NaN
    ],
)
def test_bad_arrays_raise_input_error(bad):
    with pytest.raises(InputError):
        synthesize(bad, _cond())
    with pytest.raises(InputError):
        unprocess_jpeg(bad)


def test_sample_condition_stays_in_documented_ranges():
    rng = np.random.default_rng(5)
    conds = [sample_condition(rng) for _ in range(500)]
    evs = np.array([c.ev for c in conds])
    assert evs.min() >= -3 and evs.max() <= 0 and evs.min() < -2.8 and evs.max() > -0.2
    a = np.array([c.iso_a for c in conds])
    assert a.min() >= 1e-4 and a.max() <= 0.02
    assert all(c.iso_b > 0 for c in conds)
    assert all(c.illuminant[1] == 1.0 for c in conds)


def test_sampled_ccms_keep_white_white_and_vary():
    rng = np.random.default_rng(6)
    conds = [sample_condition(rng) for _ in range(20)]
    for c in conds:
        np.testing.assert_allclose(c.ccm @ np.ones(3), np.ones(3), atol=1e-6)
    assert np.std([c.ccm[0, 0] for c in conds]) > 0.01


def test_sampling_is_seeded():
    a = sample_condition(np.random.default_rng(7))
    b = sample_condition(np.random.default_rng(7))
    assert a.iso_a == b.iso_a and a.ev == b.ev and a.seed == b.seed
    np.testing.assert_array_equal(a.ccm, b.ccm)


def test_unprocess_jpeg_is_deterministic_and_noise_free():
    rgb = _gradient()
    a, b = unprocess_jpeg(rgb), unprocess_jpeg(rgb)
    np.testing.assert_array_equal(a.bayer, b.bayer)
    flat = unprocess_jpeg(_flat(0.5))
    assert flat.bayer[0::2, 0::2].std() < 1e-6


def test_unprocess_jpeg_gives_gains_and_ccm_that_restore_the_colour():
    rgb = _flat(0.0) + np.array([0.3, 0.5, 0.7], np.float32)
    raw = unprocess_jpeg(rgb)
    cam = np.array([raw.bayer[0, 0], raw.bayer[0, 1], raw.bayer[1, 1]]) * raw.wb_gains
    smooth = np.clip(raw.ccm @ cam, 0, 1) ** (1 / 2.2)
    np.testing.assert_allclose(3 * smooth**2 - 2 * smooth**3, [0.3, 0.5, 0.7], atol=0.01)


def test_test_conditions_file_covers_the_prd_sweeps():
    conds = load_test_conditions()
    assert "standard" in conds
    assert {conds[f"ev_{n}"].ev for n in ("0", "-1", "-2", "-3")} == {0.0, -1.0, -2.0, -3.0}
    assert len({conds[k].iso_a for k in conds if k.startswith("iso_")}) >= 3
    assert len({tuple(conds[k].illuminant) for k in ("neutral", "warm", "cool", "green")}) == 4


def test_a_named_level_changes_one_axis_and_keeps_the_standard_elsewhere():
    conds = load_test_conditions()
    assert conds["ev_-3"].ev == -3.0
    assert conds["ev_-3"].iso_a == conds["standard"].iso_a
    np.testing.assert_array_equal(conds["warm"].illuminant, [1.3, 1.0, 0.75])
    assert conds["warm"].ev == conds["standard"].ev


def test_loaded_condition_matches_the_evaluation_harness():
    config = harness.load_config(CONFIG)
    fields = harness.condition_fields(config, ev="ev_-2")
    cond = load_test_conditions()["ev_-2"]
    assert (cond.iso_a, cond.iso_b, cond.ev) == (fields["iso_a"], fields["iso_b"], fields["ev"])
    np.testing.assert_array_equal(cond.ccm, fields["ccm"])


def test_test_conditions_are_identical_on_every_load():
    a, b = load_test_conditions(), load_test_conditions()
    for name in a:
        np.testing.assert_array_equal(a[name].ccm, b[name].ccm)
        assert a[name].seed == b[name].seed
