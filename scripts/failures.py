"""Rank the worst test images per pipeline and save a failure gallery with notes for people to fill in.

Reads results/metrics.json, so run scripts/evaluate.py first.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from isp.eval import harness  # noqa: E402
from isp.eval.harness import Sample  # noqa: E402
from isp.types import InputError, Pipeline  # noqa: E402

MIN_CASES = 3
CAUSE_STUB = "Cause: _TODO: a person writes this after looking at the image._"


def rank_worst(per_image: Mapping[str, Mapping[str, float | None]], metric: str) -> list[str]:
    """Image ids, worst first. A null score is an infinite PSNR read back from JSON."""
    if metric in harness.HIGHER_IS_BETTER:
        return sorted(per_image, key=lambda i: math.inf if per_image[i][metric] is None else per_image[i][metric])
    return sorted(per_image, key=lambda i: per_image[i][metric], reverse=True)


def _save_sheet(rows: list[tuple[dict, np.ndarray, np.ndarray]], path: Path) -> None:
    """One row per case: reference, output, absolute error on a shared scale."""
    errors = [np.abs(out - ref).mean(axis=2) for _, ref, out in rows]
    top = max(float(e.max()) for e in errors) or 1.0
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.6 * len(rows)), squeeze=False, layout="constrained")
    for line, (case, ref, out), error in zip(axes, rows, errors):
        line[0].imshow(ref)
        line[1].imshow(np.clip(out, 0, 1))
        heat = line[2].imshow(error, cmap="Blues", vmin=0, vmax=top)
        line[0].set_ylabel(f"{case['image']}\n{case['metric']} {case['value']:.3f}", fontsize=9)
        for ax, title in zip(line, ("Reference", case["pipeline"], "Absolute error")):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=9)
    fig.colorbar(heat, ax=axes[:, 2], shrink=0.6, label="Mean absolute error")
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def build_gallery(
    pipelines: Sequence[Pipeline],
    samples: Mapping[str, Sample],
    results: Mapping[str, Any],
    out_dir: Path,
    metric: str = "psnr",
    count: int = MIN_CASES,
) -> list[dict[str, Any]]:
    if count < MIN_CASES or len(samples) < MIN_CASES:
        raise InputError(f"The gallery needs at least {MIN_CASES} cases from at least {MIN_CASES} images.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rankings = {p.name: rank_worst(results[p.name]["per_image"], metric) for p in pipelines}
    harness.write_json(out_dir / "ranking.json", rankings)
    cases = []
    for pipeline in pipelines:
        rows = []
        for rank, image in enumerate(rankings[pipeline.name][:count], start=1):
            value = results[pipeline.name]["per_image"][image][metric]
            case = {
                "id": f"{pipeline.name}-{rank}-{image}",
                "pipeline": pipeline.name,
                "image": image,
                "metric": metric,
                "value": math.inf if value is None else value,
            }
            output = pipeline.run(copy.deepcopy(samples[image].raw)).rgb
            rows.append((case, samples[image].target, output))
            cases.append(case)
        _save_sheet(rows, out_dir / f"{pipeline.name}_worst.png")
    return cases


def write_notes(cases: Sequence[Mapping[str, Any]], out_dir: Path) -> Path:
    """Add a stub for each new case. Notes people have already written are left alone."""
    path = out_dir / "failure_notes.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Failure cases\n"
    for case in cases:
        if f"### {case['id']}\n" not in text:
            text += (
                f"\n### {case['id']}\n\n"
                f"Image {case['image']} on {case['pipeline']}: {case['metric']} = {case['value']:.3f}. "
                f"See `{case['pipeline']}_worst.png`.\n\n{CAUSE_STUB}\n"
            )
    path.write_text(text, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = harness.common_parser(__doc__.splitlines()[0])
    parser.add_argument("--metric", default="psnr", choices=list(harness.METRICS))
    parser.add_argument("--count", type=int, default=MIN_CASES, help="cases per pipeline")
    args = parser.parse_args(argv)

    metrics_path = args.root / "results" / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"{metrics_path} not found. Run scripts/evaluate.py first.")
    results = json.loads(metrics_path.read_text(encoding="utf-8"))["pipelines"]

    config = harness.load_config(args.config)
    samples = harness.make_samples(harness.load_kodak(args.kodak), config, harness.synthesize_fixed)
    pipelines = harness.build_pipelines(args.learned_ckpt, args.hybrid_backend)
    out_dir = args.root / "results" / "failures"
    cases = build_gallery(pipelines, {s.image_id: s for s in samples}, results, out_dir, args.metric, args.count)
    print(f"wrote {write_notes(cases, out_dir)}")


if __name__ == "__main__":
    main()
