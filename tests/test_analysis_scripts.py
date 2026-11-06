"""robustness.py, ablation.py and failures.py against fake pipelines."""
import json

import numpy as np
import pytest

from isp.eval.harness import run_pipelines
from isp.types import InputError
from scripts import ablation, failures, robustness
from tests.eval_fakes import CONFIG, FAST_METRICS, FakePipeline, fake_synth, make_sample, trio


def images(n=2):
    rng = np.random.default_rng(3)
    return {f"kodim{i:02d}": rng.uniform(0.2, 0.8, (32, 32, 3)).astype(np.float32) for i in range(n)}


# ---- robustness ----

def test_sweep_covers_the_full_iso_ev_cast_grid():
    out = robustness.sweep(trio(), images(), CONFIG, fake_synth, FAST_METRICS)
    assert len(out["grid"]) == 2 * 3 * 2
    assert {(g["iso"], g["ev"], g["cast"]) for g in out["grid"]} == {
        (i, e, c) for i in ("low", "high") for e in ("ev0", "ev-1", "ev-3") for c in ("neutral", "warm")
    }


def test_sweep_gives_each_pipeline_identical_input_per_condition():
    pipes = trio()
    robustness.sweep(pipes, images(), CONFIG, fake_synth, FAST_METRICS)
    assert len(pipes[0].seen) == 2 * 12
    for a, b, c in zip(*(p.seen for p in pipes)):
        assert a.tobytes() == b.tobytes() == c.tobytes()


def test_curves_vary_one_axis_with_the_others_at_standard():
    out = robustness.sweep(trio(), images(), CONFIG, fake_synth, FAST_METRICS)
    curves = robustness.curves(out, CONFIG)
    assert curves["ev"]["levels"] == ["ev0", "ev-1", "ev-3"]
    hybrid_psnr = curves["ev"]["hybrid"]["psnr"]
    assert len(hybrid_psnr) == 3
    standard = next(
        g for g in out["grid"] if (g["iso"], g["ev"], g["cast"]) == ("low", "ev-3", "neutral")
    )
    assert hybrid_psnr[2] == pytest.approx(standard["pipelines"]["hybrid"]["psnr"])


def test_curves_hold_other_axes_at_a_standard_that_is_not_first_in_the_grid():
    config = {**CONFIG, "standard": {"iso": "high", "ev": "ev0", "cast": "warm"}}
    out = robustness.sweep(trio(), images(), config, fake_synth, FAST_METRICS)
    expected = [
        next(g for g in out["grid"] if (g["iso"], g["ev"], g["cast"]) == ("high", ev, "warm"))
        for ev in ("ev0", "ev-1", "ev-3")
    ]
    got = robustness.curves(out, config)["ev"]["hybrid"]["psnr"]
    assert got == [g["pipelines"]["hybrid"]["psnr"] for g in expected]


def test_darker_exposure_lowers_measured_psnr_in_the_fake_world():
    out = robustness.sweep(trio(), images(), CONFIG, fake_synth, FAST_METRICS)
    series = robustness.curves(out, CONFIG)["ev"]["hybrid"]["psnr"]
    assert series[0] > series[1] > series[2]


def test_robustness_writes_json_and_curve_plot(tmp_path):
    robustness.write_outputs(robustness.sweep(trio(), images(), CONFIG, fake_synth, FAST_METRICS), CONFIG, tmp_path)
    data = json.loads((tmp_path / "robustness.json").read_text(encoding="utf-8"))
    assert data["curves"]["iso"]["levels"] == ["low", "high"]
    assert (tmp_path / "robustness_curves.png").stat().st_size > 5000


# ---- ablation ----

def biased_factory(toggles):
    """Each toggle costs a little quality, so deltas are non-zero."""
    return FakePipeline("hybrid", bias=0.01 + 0.02 * len(toggles))


def test_ablation_covers_every_requested_toggle():
    keys = {key for variant in ablation.VARIANTS.values() for key in variant}
    assert keys == {"colour_head", "sharpen", "ccm", "awb", "noise_map", "raw_denoise"}
    awb_methods = {v["awb"] for v in ablation.VARIANTS.values() if "awb" in v}
    assert len(awb_methods) >= 2


def test_ablation_passes_each_toggle_to_the_factory():
    seen = []

    def factory(toggles):
        seen.append(toggles)
        return biased_factory(toggles)

    ablation.run(factory, [make_sample()], FAST_METRICS)
    assert seen[0] == {}
    assert {"colour_head": False} in seen
    assert {"raw_denoise": True} in seen


def test_ablation_deltas_are_against_the_full_pipeline():
    out = ablation.run(biased_factory, [make_sample()], FAST_METRICS)
    variant = out["variants"]["no_sharpen"]
    assert variant["delta"]["psnr"] == pytest.approx(variant["mean"]["psnr"] - out["full"]["mean"]["psnr"])
    assert variant["delta"]["psnr"] < 0


def test_a_toggle_with_no_effect_is_flagged_not_reported_as_a_zero_delta():
    def ignores_toggles(toggles):
        return FakePipeline("hybrid", bias=0.03 if "sharpen" in toggles else 0.01)

    out = ablation.run(ignores_toggles, [make_sample()], FAST_METRICS)
    assert out["variants"]["no_sharpen"]["no_effect"] is False
    assert out["variants"]["no_ccm"]["no_effect"] is True


def test_a_factory_that_rejects_a_toggle_stops_the_run():
    def strict(toggles):
        if "ccm" in toggles:
            raise TypeError("unexpected toggle ccm")
        return biased_factory(toggles)

    with pytest.raises(TypeError):
        ablation.run(strict, [make_sample()], FAST_METRICS)


def test_ablation_writes_json_and_plot(tmp_path):
    ablation.write_outputs(ablation.run(biased_factory, [make_sample()], FAST_METRICS), tmp_path)
    data = json.loads((tmp_path / "ablation.json").read_text(encoding="utf-8"))
    assert "no_colour_head" in data["variants"]
    assert (tmp_path / "ablation.png").stat().st_size > 5000


# ---- failures ----

def per_image():
    return {
        "a": {"psnr": 30.0, "delta_e2000": 2.0},
        "b": {"psnr": 20.0, "delta_e2000": 9.0},
        "c": {"psnr": 25.0, "delta_e2000": 1.0},
        "d": {"psnr": None, "delta_e2000": 0.0},  # null = infinite PSNR in strict JSON
    }


def test_rank_puts_lowest_psnr_first():
    assert failures.rank_worst(per_image(), "psnr") == ["b", "c", "a", "d"]


def test_rank_puts_highest_delta_e_first():
    assert failures.rank_worst(per_image(), "delta_e2000") == ["b", "a", "c", "d"]


def gallery_inputs(n=4):
    samples = {f"s{i}": make_sample(f"s{i}", i, size=32) for i in range(n)}
    pipes = trio()
    return pipes, samples, run_pipelines(pipes, list(samples.values()), FAST_METRICS)


def test_gallery_has_at_least_three_cases_per_pipeline(tmp_path):
    pipes, samples, results = gallery_inputs()
    cases = failures.build_gallery(pipes, samples, results, tmp_path, metric="psnr", count=3)
    for pipe in pipes:
        assert sum(c["pipeline"] == pipe.name for c in cases) == 3
        assert (tmp_path / f"{pipe.name}_worst.png").stat().st_size > 5000


def test_full_ranking_is_saved_per_pipeline(tmp_path):
    pipes, samples, results = gallery_inputs()
    failures.build_gallery(pipes, samples, results, tmp_path, metric="psnr", count=3)
    ranking = json.loads((tmp_path / "ranking.json").read_text(encoding="utf-8"))
    assert set(ranking) == {"classical", "learned", "hybrid"}
    assert sorted(ranking["hybrid"]) == sorted(samples)


def test_fewer_than_three_cases_is_refused(tmp_path):
    pipes, samples, results = gallery_inputs()
    with pytest.raises(InputError, match="at least 3"):
        failures.build_gallery(pipes, samples, results, tmp_path, metric="psnr", count=2)


def test_too_few_images_is_refused(tmp_path):
    pipes, samples, results = gallery_inputs(n=2)
    with pytest.raises(InputError, match="at least 3"):
        failures.build_gallery(pipes, samples, results, tmp_path, metric="psnr", count=3)


def test_notes_stub_has_a_blank_cause_for_each_case(tmp_path):
    pipes, samples, results = gallery_inputs()
    cases = failures.build_gallery(pipes, samples, results, tmp_path, metric="psnr", count=3)
    path = failures.write_notes(cases, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.count(failures.CAUSE_STUB) == len(cases)
    for case in cases:
        assert f"### {case['id']}" in text


def test_rerun_keeps_human_notes_and_adds_new_cases(tmp_path):
    cases = [{"id": "hybrid-1-a", "pipeline": "hybrid", "image": "a", "metric": "psnr", "value": 1.0}]
    path = failures.write_notes(cases, tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace(failures.CAUSE_STUB, "Cause: clipped sky"), encoding="utf-8")
    extra = {"id": "hybrid-2-b", "pipeline": "hybrid", "image": "b", "metric": "psnr", "value": 2.0}
    text = failures.write_notes(cases + [extra], tmp_path).read_text(encoding="utf-8")
    assert "Cause: clipped sky" in text
    assert text.count("### hybrid-1-a") == 1
    assert "### hybrid-2-b" in text
