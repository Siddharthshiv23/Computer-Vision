"""Plumbing shared by the evaluation scripts: conditions, inputs, scoring and cost measurement."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import psutil
import yaml
from PIL import Image

from isp.eval.metrics import delta_e2000, lpips_alex, psnr, ssim
from isp.types import InputError, Pipeline, RawImage

REPO = Path(__file__).resolve().parents[2]

# Insertion order is the reporting order the brief requires.
METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "psnr": psnr,
    "ssim": ssim,
    "delta_e2000": delta_e2000,
    "lpips": lpips_alex,
}
HIGHER_IS_BETTER = {"psnr", "ssim"}


@dataclass
class Sample:
    image_id: str
    raw: RawImage
    target: np.ndarray


def run_pipelines(
    pipelines: Sequence[Pipeline], samples: Iterable[Sample], metric_fns: Mapping[str, Callable] = METRICS
) -> dict[str, dict[str, Any]]:
    """Score every pipeline on the same samples. Each run gets its own copy of the input."""
    out = {p.name: {"per_image": {}, "fallbacks": 0, "nonfinite_outputs": 0} for p in pipelines}
    for sample in samples:
        for pipeline in pipelines:
            result = pipeline.run(copy.deepcopy(sample.raw))
            entry = out[pipeline.name]
            entry["per_image"][sample.image_id] = {
                name: fn(result.rgb, sample.target) for name, fn in metric_fns.items()
            }
            entry["fallbacks"] += bool(result.info.get("fallback"))
            entry["nonfinite_outputs"] += bool(result.info.get("nonfinite_output"))
    for entry in out.values():
        scores = list(entry["per_image"].values())
        entry["mean"] = {name: sum(s[name] for s in scores) / len(scores) for name in metric_fns}
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _levels(config: Mapping[str, Any], overrides: Mapping[str, str]) -> dict[str, str]:
    levels = {axis: overrides.get(axis, config["standard"][axis]) for axis in config["axes"]}
    for axis, label in levels.items():
        if label not in config["axes"][axis]:
            raise InputError(f"Unknown {axis} level {label!r}; expected one of {list(config['axes'][axis])}.")
    return levels


def condition_fields(config: Mapping[str, Any], **overrides: str) -> dict[str, Any]:
    """SensorCondition fields for one level per axis; unnamed axes stay at the standard level."""
    fields = dict(config["base"])
    for axis, label in _levels(config, overrides).items():
        fields.update(config["axes"][axis][label])
    return fields


def derive_seed(*parts: object) -> int:
    """Stable across processes, unlike hash()."""
    digest = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def synthesize_fixed(rgb: np.ndarray, fields: Mapping[str, Any], seed: int) -> tuple[RawImage, np.ndarray]:
    """Real sensor model, with the noise seeded through numpy's global generator."""
    from isp.data.synthetic import SensorCondition, synthesize

    arrays = {k: np.asarray(v, np.float32) if isinstance(v, list) else v for k, v in fields.items()}
    np.random.seed(seed % 2**32)
    return synthesize(rgb, SensorCondition(**arrays))


def make_samples(
    images: Mapping[str, np.ndarray], config: Mapping[str, Any], synth: Callable, **overrides: str
) -> list[Sample]:
    """One fixed degraded input per image for the chosen condition."""
    fields = condition_fields(config, **overrides)
    tag = "/".join(_levels(config, overrides).values())
    samples = []
    for image_id, rgb in images.items():
        seed = derive_seed(config["seed"], image_id, tag)
        raw, target = synth(rgb, fields, seed)
        if not samples and synth(rgb, fields, seed)[0].bayer.tobytes() != raw.bayer.tobytes():
            raise InputError("The sensor model is not reproducible from a seed, so test inputs would not be fixed.")
        samples.append(Sample(image_id, raw, target))
    return samples


def load_kodak(root: str | Path) -> dict[str, np.ndarray]:
    paths = sorted(Path(root).rglob("kodim*.png"))
    if len(paths) != 24:
        raise InputError(f"Kodak24 needs 24 kodim*.png files under {root}; found {len(paths)}.")
    return {p.stem: np.asarray(Image.open(p).convert("RGB"), np.float32) / 255 for p in paths}


def tile_to_megapixel(raw: RawImage, side: int = 1024) -> RawImage:
    """Wrap the mosaic out to side x side. Even input sizes keep the RGGB phase."""
    pad = [(0, max(0, side - n)) for n in raw.bayer.shape]
    return dataclasses.replace(raw, bayer=np.pad(raw.bayer, pad, mode="wrap")[:side, :side])


def _set_cpus(process: psutil.Process, cpus: int | None) -> tuple[list[int] | None, int | None]:
    """Pin to the first `cpus` CPUs. Returns the previous affinity and torch thread count."""
    if cpus is None or not hasattr(process, "cpu_affinity"):
        return None, None
    previous = process.cpu_affinity()
    process.cpu_affinity(previous[:cpus])
    torch = sys.modules.get("torch")
    threads = torch.get_num_threads() if torch else None
    if torch:
        torch.set_num_threads(min(cpus, len(previous)))
    return previous, threads


def measure_cost(pipeline: Pipeline, raw: RawImage, repeats: int = 5, cpus: int | None = 2) -> dict[str, Any]:
    """Median latency and peak process RSS. Peak comes from a sampler, as RSS has no per-call peak."""
    process = psutil.Process()
    previous, threads = _set_cpus(process, cpus)
    baseline = peak = process.memory_info().rss
    done = threading.Event()

    def sample_rss() -> None:
        nonlocal peak
        while not done.wait(0.005):
            peak = max(peak, process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    try:
        pipeline.run(copy.deepcopy(raw))  # warm-up, excluded from latency but not from peak memory
        times = []
        for _ in range(repeats):
            run_input = copy.deepcopy(raw)
            start = time.perf_counter()
            pipeline.run(run_input)
            times.append((time.perf_counter() - start) * 1000)
        pinned = len(process.cpu_affinity()) if previous is not None else None
    finally:
        done.set()
        sampler.join()
        if previous is not None:
            process.cpu_affinity(previous)
        if threads is not None:
            sys.modules["torch"].set_num_threads(threads)
    return {
        "latency_ms": statistics.median(times),
        "repeats": repeats,
        "cpus": pinned,
        "input_shape": list(raw.bayer.shape),
        "peak_rss_mb": peak / 1e6,
        "peak_rss_delta_mb": (peak - baseline) / 1e6,
    }


def model_stats(path: str | Path | None) -> dict[str, int]:
    """Parameter count and on-disk size read from the model file. None means no model."""
    if path is None:
        return {"params": 0, "size_bytes": 0}
    path = Path(path)
    if not path.is_file():
        raise InputError(f"Model file not found: {path}")
    if path.suffix == ".onnx":
        import onnx

        tensors = onnx.load(str(path)).graph.initializer
        counts = [
            math.prod(t.dims) for t in tensors if not t.name.endswith(("_scale", "_zero_point"))
        ]
    elif path.suffix == ".pt":
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)["state"]
        buffers = ("running_mean", "running_var", "num_batches_tracked")
        counts = [t.numel() for name, t in state.items() if not name.endswith(buffers)]
    else:
        raise InputError(f"Expected a .pt or .onnx model, got {path.name}.")
    return {"params": sum(counts), "size_bytes": path.stat().st_size}


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, data: Any) -> None:
    """Strict JSON. A non-finite score, such as the PSNR of identical images, becomes null."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_strict(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_pipelines(learned_ckpt: str, hybrid_backend: str) -> list[Pipeline]:
    """The three real pipelines. Imported here so the harness works before they exist."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # evaluation is CPU only
    from isp.classical.pipeline import ClassicalISP
    from isp.hybrid.pipeline import HybridISP
    from isp.learned.pipeline import LearnedISP

    return [ClassicalISP(), LearnedISP(learned_ckpt), HybridISP(hybrid_backend)]


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=REPO, help="output root; results go in <root>/results")
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "test_conditions.yaml")
    parser.add_argument("--kodak", type=Path, default=REPO / "data" / "kodak24")
    parser.add_argument("--learned-ckpt", required=True, help="RawNet .pt checkpoint")
    parser.add_argument("--hybrid-backend", required=True, help="hybrid .pt or .onnx backend")
    return parser
