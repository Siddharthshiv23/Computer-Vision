"""Benchmark the hybrid pipeline: params, MB on disk, peak RSS and median latency (F-OPT-4).

  python -m scripts.benchmark                       # untrained student at the default width (latency only)
  python -m scripts.benchmark --ckpt runs/student/best.pt --onnx-int8 weights/student_int8.onnx

Latency does not depend on the weight values, so an untrained model is fine for timing, but the JSON
marks it "untrained" and carries no quality number. Every variant is timed in a fresh subprocess so
its peak RSS is its own. Threads are limited to 2 (torch, OpenCV, BLAS and ONNX Runtime) because the
target is a 2 vCPU machine; note that is a thread limit on this host, not the same CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREADS = int(os.environ.get("ISP_BENCH_THREADS", "2"))
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = str(THREADS)  # must be set before numpy / torch are imported

import numpy as np  # noqa: E402

sys.path.insert(0, str(ROOT))

TARGET_MS = 1500.0
DEFAULT_SIZES = "512=512x512,1mp=1000x1000,12mp=4000x3000"
_KEYS = ("front_end", "input_prep", "network", "sharpen")


def parse_sizes(spec: str) -> list[tuple[str, int, int]]:
    """'label=WxH,WxH,...' -> [(label, w, h)]; both dimensions must be even (Bayer)."""
    out = []
    for tok in spec.split(","):
        label, _, dims = tok.rpartition("=")
        try:
            w, h = (int(v) for v in dims.lower().split("x"))
        except ValueError:
            raise argparse.ArgumentTypeError(f"bad size {tok!r}; use WxH or label=WxH") from None
        if w < 16 or h < 16 or w % 2 or h % 2:
            raise argparse.ArgumentTypeError(f"size {tok!r} must be even and at least 16 px")
        out.append((label or dims, w, h))
    return out


def synthetic_mosaic(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Smooth structured scene plus Poisson-Gaussian-like noise; content only matters for realism."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    clean = 0.05 + 0.3 * (0.5 + 0.5 * np.sin(xx / 37.0) * np.cos(yy / 53.0))
    noisy = clean + rng.normal(0, 1, clean.shape).astype(np.float32) * np.sqrt(0.004 * clean + 2e-5)
    return np.clip(noisy, 0, 1).astype(np.float32)


def peak_rss_mb() -> float:
    import psutil

    info = psutil.Process().memory_info()
    return float(getattr(info, "peak_wset", info.rss)) / 2**20  # peak_wset exists on Windows only


def run_worker(spec: dict) -> dict:
    """Time one backend over all sizes. Runs inside its own subprocess."""
    import cv2
    import torch

    torch.set_num_threads(spec["threads"])
    cv2.setNumThreads(spec["threads"])
    from isp.hybrid.controller import AdaptiveController
    from isp.hybrid.pipeline import HybridISP
    from isp.types import RawImage

    pipe = HybridISP(spec["backend"], controller=AdaptiveController(clean_sigma=0.0), threads=spec["threads"])
    result = {}
    for label, w, h in spec["sizes"]:
        raw = RawImage(bayer=synthetic_mosaic(w, h))
        pipe.run(raw)  # warm-up: allocator, ORT arena, caches
        runs = [pipe.run(raw) for _ in range(spec["repeats"])]
        result[label] = {"width": w, "height": h, "runs": len(runs), "tiles": runs[0].info["tiles"],
                         "fallback": any(r.info["fallback"] for r in runs),
                         "total": statistics.median(r.timings_ms["total"] for r in runs),
                         **{k: statistics.median(r.timings_ms.get(k, 0.0) for r in runs) for k in _KEYS}}
    return {"latency_ms": result, "peak_rss_mb": peak_rss_mb()}


def _measure(name: str, backend: str, args, weights: str, params: int | None) -> dict:
    spec = {"backend": backend, "sizes": args.sizes, "repeats": args.repeats, "threads": args.threads}
    env = {**os.environ, "ISP_BENCH_THREADS": str(args.threads)}
    proc = subprocess.run([sys.executable, "-m", "scripts.benchmark", "--worker", json.dumps(spec)], cwd=ROOT,
                          env=env, capture_output=True, text=True)
    if proc.returncode:
        raise SystemExit(f"benchmark worker for {name} failed:\n{proc.stderr[-2000:]}")
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    res.update(weights=weights, params=params, size_mb=Path(backend).stat().st_size / 2**20, backend=Path(backend).name)
    one_mp = res["latency_ms"].get("1mp")
    res["meets_1mp_target"] = None if one_mp is None else one_mp["total"] < TARGET_MS
    return res


def _calibration(n: int = 6) -> list[np.ndarray]:
    """Real network inputs: the classical front end applied to synthetic mosaics."""
    from isp.classical.pipeline import front_end
    from isp.learned.inputs import liteisp_input
    from isp.types import RawImage

    out = []
    for seed in range(n):
        rgb, _, m = front_end(RawImage(bayer=synthetic_mosaic(256, 256, seed)), False)
        out.append(liteisp_input(rgb, m["noise_a"], m["noise_b"], m["exposure"])[None].numpy())
    return out


def _source(args, tmp: Path) -> tuple[str | None, str, int | None]:
    """(checkpoint path, weights status, param count)."""
    from isp.learned.checkpoint import build_model, load_checkpoint, save_checkpoint
    from isp.models.liteisp import STUDENT_WIDTH

    if args.ckpt:
        model, meta = load_checkpoint(args.ckpt, expect="liteisp")
        status = args.weights if args.weights != "auto" else ("trained" if "epoch" in meta else "untrained")
    elif args.onnx_fp32 and args.onnx_int8:
        return None, args.weights if args.weights != "auto" else "unknown", None
    else:
        width = args.width or STUDENT_WIDTH
        model, status = build_model("liteisp", width), "untrained"
        args.ckpt = str(tmp / "untrained.pt")
        save_checkpoint(args.ckpt, model, "liteisp", width)
    return args.ckpt, status, sum(p.numel() for p in model.parameters())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--ckpt", help="LiteISPNet .pt; omit to benchmark an untrained model")
    ap.add_argument("--width", type=int, help="width of the untrained model (default: STUDENT_WIDTH)")
    ap.add_argument("--onnx-fp32", help="pre-exported FP32 ONNX (else exported from the checkpoint)")
    ap.add_argument("--onnx-int8", help="pre-quantised INT8 ONNX (else quantised here)")
    ap.add_argument("--weights", choices=["auto", "trained", "untrained"], default="auto")
    ap.add_argument("--sizes", type=parse_sizes, default=parse_sizes(DEFAULT_SIZES))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark.json"))
    args = ap.parse_args(argv)
    if args.worker:
        print(json.dumps(run_worker(json.loads(args.worker))))
        return
    from isp.optim import export_onnx, quantize_int8

    with tempfile.TemporaryDirectory() as td:
        ckpt, weights, params = _source(args, Path(td))
        fp32 = args.onnx_fp32 or export_onnx(ckpt, str(Path(td) / "student_fp32.onnx"))
        int8 = args.onnx_int8 or quantize_int8(fp32, str(Path(td) / "student_int8.onnx"), _calibration())
        variants = {"fp32_onnx": _measure("fp32_onnx", fp32, args, weights, params),
                    "int8_onnx": _measure("int8_onnx", int8, args, weights, params)}
    report = {"threads": args.threads, "weights": weights, "target_ms": TARGET_MS, "created": time.strftime("%F %T"),
              "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "python": platform.python_version()},
              "note": "Latency is host-dependent; threads are limited, the CPU is not a 2 vCPU part.",
              "variants": variants}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {"size_mb": round(v["size_mb"], 3), **{s: round(r["total"]) for s, r in v["latency_ms"].items()}}
                      for k, v in variants.items()}))


if __name__ == "__main__":
    main()
