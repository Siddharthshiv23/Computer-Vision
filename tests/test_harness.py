import hashlib
import json
import math
import time

import numpy as np
import pytest

from isp.eval import harness
from isp.eval.harness import (
    condition_fields,
    derive_seed,
    make_samples,
    measure_cost,
    model_stats,
    run_pipelines,
    tile_to_megapixel,
    write_json,
)
from isp.types import InputError, PipelineResult, RawImage
from tests.eval_fakes import CONFIG, FAST_METRICS, FakePipeline, fake_synth, make_sample, trio


def test_every_pipeline_sees_byte_identical_input():
    pipes = trio()
    samples = [make_sample("a", 1), make_sample("b", 2)]
    run_pipelines(pipes, samples, FAST_METRICS)
    for first, second, third in zip(pipes[0].seen, pipes[1].seen, pipes[2].seen):
        assert first.tobytes() == second.tobytes() == third.tobytes()


def test_a_pipeline_that_mutates_its_input_cannot_affect_the_next_one():
    vandal, victim = FakePipeline("classical", mutate_input=True), FakePipeline("hybrid")
    sample = make_sample()
    original = sample.raw.bayer.copy()
    run_pipelines([vandal, victim], [sample], FAST_METRICS)
    np.testing.assert_array_equal(victim.seen[0], original)


def test_means_are_over_per_image_scores():
    out = run_pipelines([FakePipeline("classical", 0.1)], [make_sample("a", 1), make_sample("b", 2)], FAST_METRICS)
    entry = out["classical"]
    for name in FAST_METRICS:
        scores = [entry["per_image"][i][name] for i in ("a", "b")]
        assert entry["mean"][name] == pytest.approx(sum(scores) / 2)


def test_metrics_keep_the_required_order():
    out = run_pipelines([FakePipeline("classical", 0.1)], [make_sample()], FAST_METRICS)
    assert list(out["classical"]["mean"]) == ["psnr", "ssim", "delta_e2000", "lpips"]


def test_nan_output_fails_instead_of_being_averaged():
    class Broken:
        name = "hybrid"

        def run(self, raw):
            return PipelineResult(rgb=np.full((64, 64, 3), np.nan, dtype=np.float32))

    with pytest.raises(InputError):
        run_pipelines([Broken()], [make_sample()], FAST_METRICS)


def test_fallbacks_are_counted_per_pipeline():
    pipe = FakePipeline("hybrid", info={"fallback": True})
    out = run_pipelines([pipe, FakePipeline("classical")], [make_sample("a"), make_sample("b")], FAST_METRICS)
    assert out["hybrid"]["fallbacks"] == 2
    assert out["classical"]["fallbacks"] == 0


def test_sanitised_nonfinite_network_output_is_counted():
    # LearnedISP replaces NaN before returning and only reports it through info.
    pipe = FakePipeline("learned", info={"nonfinite_output": True})
    assert run_pipelines([pipe], [make_sample()], FAST_METRICS)["learned"]["nonfinite_outputs"] == 1


def test_make_samples_refuses_a_synthesiser_that_ignores_the_seed():
    def unseeded(rgb, fields, seed):
        return fake_synth(rgb, {"iso_a": 0.1}, int(rng.integers(1 << 31)))

    rng = np.random.default_rng()
    rgb = {"a": np.full((16, 16, 3), 0.5, dtype=np.float32)}
    with pytest.raises(InputError, match="reproducible"):
        make_samples(rgb, CONFIG, unseeded)


def test_make_samples_is_identical_across_calls():
    rgb = {"a": np.full((16, 16, 3), 0.5, dtype=np.float32)}
    first = make_samples(rgb, CONFIG, fake_synth, iso="high")
    second = make_samples(rgb, CONFIG, fake_synth, iso="high")
    assert first[0].raw.bayer.tobytes() == second[0].raw.bayer.tobytes()


def test_condition_fields_merge_base_and_each_axis():
    fields = condition_fields(CONFIG, iso="high", ev="ev-1", cast="warm")
    assert fields == {"iso_a": 0.1, "ev": -1.0, "illuminant": [1.4, 1.0, 0.7]}


def test_condition_fields_default_to_standard():
    assert condition_fields(CONFIG)["ev"] == 0.0


def test_unknown_axis_level_is_rejected():
    with pytest.raises(InputError, match="ev-9"):
        condition_fields(CONFIG, ev="ev-9")


def test_seed_is_stable_and_depends_on_every_part():
    assert derive_seed(7, "kodim01", "low/ev0") == derive_seed(7, "kodim01", "low/ev0")
    assert derive_seed(7, "kodim01", "low/ev0") != derive_seed(7, "kodim02", "low/ev0")
    assert derive_seed(7, "kodim01", "low/ev0") != derive_seed(8, "kodim01", "low/ev0")
    # Pinned to sha256, because Python's hash() is salted per process and would not reproduce.
    digest = hashlib.sha256(b"7/kodim01/low/ev0").digest()
    assert derive_seed(7, "kodim01", "low/ev0") == int.from_bytes(digest[:8], "big")


def test_tile_to_megapixel_reaches_one_megapixel_and_keeps_cfa_phase():
    bayer = np.zeros((512, 768), dtype=np.float32)
    bayer[0::2, 0::2] = 1.0  # mark the R sites
    out = tile_to_megapixel(RawImage(bayer=bayer))
    assert out.bayer.shape == (1024, 1024)
    assert out.bayer[0::2, 0::2].min() == 1.0
    assert out.bayer[1::2, :].max() == 0.0


def test_measure_cost_reports_median_latency():
    class Slow(FakePipeline):
        def run(self, raw):
            time.sleep(0.03)
            return super().run(raw)

    cost = measure_cost(Slow("x"), make_sample().raw, repeats=3, cpus=None)
    assert 25 <= cost["latency_ms"] < 500
    assert cost["repeats"] == 3


def test_measure_cost_sees_a_transient_allocation():
    class Hungry(FakePipeline):
        def run(self, raw):
            block = np.ones(200 * 1024 * 1024 // 8)  # about 200 MB, touched, then freed
            time.sleep(0.1)
            del block
            return super().run(raw)

    cost = measure_cost(Hungry("x"), make_sample().raw, repeats=1, cpus=None)
    assert cost["peak_rss_delta_mb"] > 120


def test_measure_cost_restores_cpu_affinity():
    psutil = pytest.importorskip("psutil")
    before = psutil.Process().cpu_affinity()
    cost = measure_cost(FakePipeline("x"), make_sample().raw, repeats=1, cpus=2)
    assert psutil.Process().cpu_affinity() == before
    assert cost["cpus"] == 2


def test_model_stats_none_is_a_model_free_pipeline():
    assert model_stats(None) == {"params": 0, "size_bytes": 0}


def test_model_stats_counts_torch_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "m.pt"
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
    torch.save({"model": "rawnet", "width": 4, "state": net.state_dict(), "val_loss": 0.1}, path)
    stats = model_stats(path)
    # conv 3*4*9 + 4 bias, BN weight and bias 4 + 4. Running stats are buffers, not parameters.
    assert stats["params"] == 108 + 4 + 8
    assert stats["size_bytes"] == path.stat().st_size


def test_model_stats_counts_onnx_without_quantisation_scales(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    weight = helper.make_tensor("conv.weight", TensorProto.INT8, [2, 5], [0] * 10)
    scale = helper.make_tensor("conv.weight_scale", TensorProto.FLOAT, [2], [1.0, 1.0])
    zero = helper.make_tensor("conv.weight_zero_point", TensorProto.INT8, [2], [0, 0])
    graph = helper.make_graph([], "g", [], [], [weight, scale, zero])
    path = tmp_path / "m.onnx"
    onnx.save(helper.make_model(graph), path)
    assert model_stats(path)["params"] == 10


def test_model_stats_missing_file_is_an_error(tmp_path):
    with pytest.raises(InputError, match="nope.onnx"):
        model_stats(tmp_path / "nope.onnx")


def test_write_json_is_strict_json_with_infinity_as_null(tmp_path):
    path = tmp_path / "sub" / "out.json"
    write_json(path, {"psnr": math.inf, "ok": 1.5})
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert json.loads(text) == {"psnr": None, "ok": 1.5}


def test_load_kodak_requires_all_24_images(tmp_path):
    with pytest.raises(InputError, match="Kodak24"):
        harness.load_kodak(tmp_path)
