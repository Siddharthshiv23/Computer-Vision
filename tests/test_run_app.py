"""scripts/run_app.py demo: real outputs only, and the user's file survives."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from app import adapter
from app.adapter import Slot
from isp.types import PipelineResult, RawImage

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_app", ROOT / "scripts" / "run_app.py")
run_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_app)


class Grey:
    def run(self, raw):
        return PipelineResult(rgb=np.full((*raw.bayer.shape, 3), 0.5, np.float32), timings_ms={"tone": 1.0})


def wire(monkeypatch, slots, loader=True):
    def load(data, filename=None):
        return RawImage(bayer=np.zeros((64, 64), np.float32))

    monkeypatch.setattr(adapter, "load_slots", lambda: slots)
    monkeypatch.setattr(adapter, "load_loader", lambda: (load, "") if loader else (None, "Not available yet."))
    monkeypatch.setattr(adapter, "load_metrics", lambda: (None, "Not available yet."))


def test_demo_writes_one_png_per_finished_pipeline_and_keeps_the_input(monkeypatch, tmp_path):
    wire(monkeypatch, [Slot("hybrid", "Hybrid", Grey(), ""), Slot("learned", "Learned", None, "Not available yet.")])
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x")
    out = tmp_path / "out"
    assert run_app.demo(str(photo), out) == 0
    assert photo.read_bytes() == b"x"
    assert [p.name for p in out.iterdir()] == ["hybrid.png"]


def test_demo_with_nothing_available_fails_and_writes_nothing(monkeypatch, tmp_path, capsys):
    wire(monkeypatch, [Slot("hybrid", "Hybrid", None, "Not available yet.")], loader=False)
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x")
    out = tmp_path / "out"
    assert run_app.demo(str(photo), out) == 1
    assert not out.exists()
    assert "not available yet" in capsys.readouterr().out.lower()


def test_demo_without_input_lists_availability_and_says_how_to_run(monkeypatch, tmp_path, capsys):
    wire(monkeypatch, [Slot("hybrid", "Hybrid", Grey(), "")])
    assert run_app.demo(None, tmp_path / "out") == 2
    printed = capsys.readouterr().out
    assert "hybrid" in printed.lower() and "--input" in printed
