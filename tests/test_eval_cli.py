"""Each script's main() end to end: real config file, fake pipelines and images, output in tmp."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from isp.eval import harness
from scripts import ablation, evaluate, failures, robustness
from tests.eval_fakes import FAST_METRICS, FakePipeline, fake_synth, trio

REPO = Path(__file__).resolve().parents[1]
COMMON = ["--learned-ckpt", "unused.pt", "--hybrid-backend", "unused.onnx"]


@pytest.fixture
def wired(monkeypatch, tmp_path):
    rng = np.random.default_rng(5)
    images = {f"kodim{i:02d}": rng.uniform(0.2, 0.8, (64, 64, 3)).astype(np.float32) for i in range(1, 5)}
    monkeypatch.setattr(harness, "load_kodak", lambda root: images)
    monkeypatch.setattr(harness, "synthesize_fixed", fake_synth)
    monkeypatch.setattr(harness, "build_pipelines", lambda ckpt, backend: trio())
    monkeypatch.setattr(harness, "model_stats", lambda path: {"params": 0, "size_bytes": 0})
    monkeypatch.setitem(harness.METRICS, "lpips", FAST_METRICS["lpips"])
    return ["--root", str(tmp_path), *COMMON]


def test_evaluate_main_writes_report_and_json(wired, tmp_path):
    evaluate.main([*wired, "--repeats", "1"])
    data = json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert len(data["pipelines"]["hybrid"]["per_image"]) == 4
    assert data["provenance"]["condition"] == {"iso": "iso_400", "ev": "ev_0", "cast": "neutral"}
    assert data["costs"]["hybrid"]["cpus"] is not None
    assert (tmp_path / "eval_P2_TXX.md").read_text(encoding="utf-8").startswith(evaluate.USN_PLACEHOLDER)


def test_evaluate_main_team_id_and_usns_options(wired, tmp_path):
    evaluate.main([*wired, "--repeats", "1", "--team-id", "T07", "--usns", "1AB23CS001,1AB23CS002"])
    assert (tmp_path / "eval_P2_T07.md").read_text(encoding="utf-8").splitlines()[0] == "1AB23CS001,1AB23CS002"


def test_robustness_main_uses_the_real_grid(wired, tmp_path):
    robustness.main([*wired, "--images", "1"])
    data = json.loads((tmp_path / "results" / "robustness.json").read_text(encoding="utf-8"))
    assert len(data["grid"]) == 4 * 4 * 4
    assert data["images"] == ["kodim01"]


def test_ablation_main_writes_outputs(wired, tmp_path, monkeypatch):
    monkeypatch.setattr(ablation, "hybrid_factory", lambda backend: lambda t: FakePipeline("hybrid", 0.01 * len(t)))
    ablation.main(wired)
    assert (tmp_path / "results" / "ablation.json").exists()


def test_failures_main_reads_metrics_json(wired, tmp_path):
    evaluate.main([*wired, "--repeats", "1"])
    failures.main(wired)
    out = tmp_path / "results" / "failures"
    assert (out / "hybrid_worst.png").exists()
    assert (out / "failure_notes.md").exists()


def test_failures_main_without_metrics_says_what_to_run(wired):
    with pytest.raises(SystemExit, match="evaluate.py"):
        failures.main(wired)


def test_real_config_has_standard_levels_on_every_axis():
    config = harness.load_config(REPO / "configs" / "test_conditions.yaml")
    assert set(config["standard"]) == set(config["axes"]) == {"iso", "ev", "cast"}
    fields = harness.condition_fields(config)
    assert set(fields) == {"ccm", "iso_a", "iso_b", "ev", "illuminant"}


@pytest.mark.parametrize("script", ["evaluate", "robustness", "ablation", "failures"])
def test_script_runs_from_the_command_line(script):
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / f"{script}.py"), "--help"], capture_output=True, text=True, cwd=REPO.parent
    )
    assert done.returncode == 0, done.stderr
