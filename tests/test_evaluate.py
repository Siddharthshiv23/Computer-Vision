import json
import re

import pytest

from isp.eval.harness import run_pipelines
from scripts import evaluate
from tests.eval_fakes import FAST_METRICS, make_sample, trio


def scores(psnr=30.0, ssim=0.8, de=10.0, lp=0.2):
    return {"psnr": psnr, "ssim": ssim, "delta_e2000": de, "lpips": lp}


def table(classical, hybrid, learned=None):
    make = lambda mean: {"mean": mean, "per_image": {}, "fallbacks": 0}  # noqa: E731
    return {"classical": make(classical), "learned": make(learned or classical), "hybrid": make(hybrid)}


def costs(params=150_000, size=400_000, latency=1000.0):
    def entry(p, s):
        return {"params": p, "size_bytes": s, "latency_ms": latency, "peak_rss_mb": 300.0}

    return {"classical": entry(0, 0), "learned": entry(params, size), "hybrid": entry(params, size)}


def status(checks, key):
    return next(c["status"] for c in checks if c["id"] == key)


def test_deltas_are_against_classical():
    deltas = evaluate.deltas(table(scores(), scores(psnr=34.5, de=6.0)))
    assert deltas["hybrid"]["psnr"] == pytest.approx(4.5)
    assert deltas["hybrid"]["delta_e2000"] == pytest.approx(-4.0)
    assert "classical" not in deltas


@pytest.mark.parametrize(
    "key, hybrid, expected",
    [
        ("psnr", scores(psnr=33.0), "met"),  # exactly baseline + 3 dB
        ("psnr", scores(psnr=32.99), "missed"),
        ("ssim", scores(ssim=0.85), "met"),
        ("ssim", scores(ssim=0.84), "missed"),
        ("delta_e2000", scores(de=7.0), "met"),  # exactly 0.7 x baseline
        ("delta_e2000", scores(de=7.01), "missed"),
        ("lpips", scores(lp=0.14), "met"),
        ("lpips", scores(lp=0.15), "missed"),
    ],
)
def test_quality_targets_follow_prd_thresholds(key, hybrid, expected):
    checks = evaluate.check_targets(table(scores(), hybrid), costs())
    assert status(checks, key) == expected


@pytest.mark.parametrize(
    "key, kwargs, expected",
    [
        ("params", {"params": 199_999}, "met"),
        ("params", {"params": 200_000}, "missed"),  # the PRD bound is strict
        ("size", {"size": 500_000}, "met"),  # the PRD bound is inclusive
        ("size", {"size": 500_001}, "missed"),
        ("latency", {"latency": 1499.0}, "met"),
        ("latency", {"latency": 1500.0}, "missed"),
    ],
)
def test_efficiency_targets_follow_prd_thresholds(key, kwargs, expected):
    assert status(evaluate.check_targets(table(scores(), scores()), costs(**kwargs)), key) == expected


def test_unmeasured_values_are_never_marked_met():
    measured = costs()
    measured["hybrid"]["latency_ms"] = None
    checks = evaluate.check_targets(table(scores(), scores()), measured)
    assert status(checks, "latency") == "not measured"


def test_app_response_target_is_listed_as_not_measured_here():
    assert status(evaluate.check_targets(table(scores(), scores()), costs()), "app_response") == "not measured"


def test_infinite_psnr_counts_as_measured():
    checks = evaluate.check_targets(table(scores(), scores(psnr=float("inf"))), costs())
    assert status(checks, "psnr") == "met"


def report(tmp_path, **kwargs):
    pipes = trio()
    results = run_pipelines(pipes, [make_sample("a", 1), make_sample("b", 2)], FAST_METRICS)
    return evaluate.write_outputs(results, costs(), tmp_path, provenance={"note": "test"}, **kwargs)


def test_first_line_is_usns(tmp_path):
    path = report(tmp_path, usns="1AB23CS001,1AB23CS002")
    assert path.read_text(encoding="utf-8").splitlines()[0] == "1AB23CS001,1AB23CS002"


def test_first_line_is_a_placeholder_until_supplied(tmp_path):
    path = report(tmp_path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == evaluate.USN_PLACEHOLDER


def test_metric_rows_appear_in_required_order(tmp_path):
    text = report(tmp_path).read_text(encoding="utf-8")
    table_text = text.split("## Metrics")[1].split("##")[0]
    labels = ["PSNR", "SSIM", "DeltaE2000", "LPIPS", "Params", "Size", "Latency", "Peak memory"]
    positions = [table_text.index(label) for label in labels]
    assert positions == sorted(positions)


def test_report_has_a_column_per_pipeline_and_delta(tmp_path):
    header = next(
        line for line in report(tmp_path).read_text(encoding="utf-8").splitlines() if line.startswith("| Metric")
    )
    for column in ("Classical", "Learned", "Hybrid", "Learned - Classical", "Hybrid - Classical"):
        assert column in header


def test_report_lists_target_status_from_measured_values(tmp_path):
    text = report(tmp_path).read_text(encoding="utf-8")
    section = text.split("## PRD section 8 targets")[1]
    assert re.search(r"\b(met|missed)\b", section)
    assert "not measured" in section  # app response


def test_metrics_json_round_trips_with_per_image_scores(tmp_path):
    report(tmp_path)
    data = json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert set(data["pipelines"]) == {"classical", "learned", "hybrid"}
    assert set(data["pipelines"]["hybrid"]["per_image"]) == {"a", "b"}
    assert data["provenance"] == {"note": "test"}
    assert data["targets"] and data["deltas"]["hybrid"]


def test_fallbacks_are_disclosed_in_the_report(tmp_path):
    results = run_pipelines(trio(), [make_sample()], FAST_METRICS)
    results["hybrid"]["fallbacks"] = 1
    path = evaluate.write_outputs(results, costs(), tmp_path, provenance={})
    assert "fell back" in path.read_text(encoding="utf-8")


def test_report_defaults_to_the_txx_placeholder_name(tmp_path):
    assert report(tmp_path).name == "eval_P2_TXX.md"


def test_team_id_names_the_report(tmp_path):
    assert report(tmp_path, team_id="T07").name == "eval_P2_T07.md"


def test_missing_pipeline_is_refused(tmp_path):
    results = run_pipelines(trio()[:2], [make_sample()], FAST_METRICS)
    with pytest.raises(ValueError, match="hybrid"):
        evaluate.write_outputs(results, costs(), tmp_path, provenance={})
