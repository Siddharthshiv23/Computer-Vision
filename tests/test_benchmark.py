import json

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("psutil")

from scripts import benchmark  # noqa: E402


def test_benchmark_on_untrained_tiny_model_writes_marked_results(tmp_path):
    out = tmp_path / "benchmark.json"
    benchmark.main(["--width", "8", "--sizes", "64x64,96x128", "--repeats", "1", "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["threads"] == 2
    assert data["weights"] == "untrained"
    assert {"fp32_onnx", "int8_onnx"} <= set(data["variants"])
    for v in data["variants"].values():
        assert v["weights"] == "untrained"
        assert v["params"] > 0 and 0 < v["size_mb"] < 0.5 and v["peak_rss_mb"] > 10
        for lat in v["latency_ms"].values():
            assert lat["total"] > 0 and {"front_end", "network", "sharpen"} <= set(lat)
            assert lat["total"] >= lat["network"]
    assert data["variants"]["int8_onnx"]["size_mb"] < data["variants"]["fp32_onnx"]["size_mb"]


def test_target_verdict_is_derived_from_measurements(tmp_path):
    out = tmp_path / "b.json"
    benchmark.main(["--width", "8", "--sizes", "1mp=64x64", "--repeats", "1", "--out", str(out)])
    data = json.loads(out.read_text())
    lat = data["variants"]["int8_onnx"]["latency_ms"]["1mp"]["total"]
    assert data["variants"]["int8_onnx"]["meets_1mp_target"] == (lat < data["target_ms"])


def test_rejects_bad_size_spec(tmp_path):
    with pytest.raises(SystemExit):
        benchmark.main(["--sizes", "banana", "--out", str(tmp_path / "x.json")])
