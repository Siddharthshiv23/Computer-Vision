"""Run classical, learned and hybrid on identical Kodak24 inputs and write the results.

Writes results/metrics.json and eval_P2_<TeamID>.md (F-EV-1, F-HY-4).
"""
from __future__ import annotations

import math
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isp.eval import harness  # noqa: E402

USN_PLACEHOLDER = "USN_TBD"
TEAM_PLACEHOLDER = "TXX"  # the T?? of the brief is not a legal Windows file name
REQUIRED = ("classical", "learned", "hybrid")
TOLERANCE = 1e-9  # so a value exactly on a threshold is not lost to float rounding
MAX_PARAMS = 200_000
MAX_SIZE_BYTES = 500_000  # PRD 0.5 MB, decimal
MAX_LATENCY_MS = 1500.0


def report_name(team_id: str) -> str:
    return f"eval_P2_{team_id}.md"


def deltas(results: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    base = results["classical"]["mean"]
    return {
        name: {metric: entry["mean"][metric] - base[metric] for metric in base}
        for name, entry in results.items()
        if name != "classical"
    }


def _check(key: str, label: str, required: str, value: float | None, passes: Callable[[float], bool]) -> dict:
    status = "not measured" if value is None else "met" if passes(value) else "missed"
    return {"id": key, "metric": label, "required": required, "measured": value, "status": status}


def check_targets(results: Mapping[str, Any], costs: Mapping[str, Any]) -> list[dict]:
    """PRD section 8, judged on the hybrid pipeline's measured values."""
    base, hybrid, cost = results["classical"]["mean"], results["hybrid"]["mean"], costs["hybrid"]
    psnr_min, ssim_min = base["psnr"] + 3, base["ssim"] + 0.05
    de_max, lpips_max = 0.7 * base["delta_e2000"], 0.7 * base["lpips"]
    return [
        _check("psnr", "PSNR", f">= {psnr_min:.2f} dB (baseline + 3)", hybrid["psnr"],
               lambda v: v >= psnr_min - TOLERANCE),
        _check("ssim", "SSIM", f">= {ssim_min:.4f} (baseline + 0.05)", hybrid["ssim"],
               lambda v: v >= ssim_min - TOLERANCE),
        _check("delta_e2000", "DeltaE2000", f"<= {de_max:.3f} (0.7 x baseline)", hybrid["delta_e2000"],
               lambda v: v <= de_max + TOLERANCE),
        _check("lpips", "LPIPS", f"<= {lpips_max:.4f} (0.7 x baseline)", hybrid["lpips"],
               lambda v: v <= lpips_max + TOLERANCE),
        _check("params", "Params", f"< {MAX_PARAMS}", cost.get("params"), lambda v: v < MAX_PARAMS),
        _check("size", "Size (bytes)", f"<= {MAX_SIZE_BYTES}", cost.get("size_bytes"),
               lambda v: v <= MAX_SIZE_BYTES),
        _check("latency", "Latency 1 MP (ms)", f"< {MAX_LATENCY_MS:.0f}", cost.get("latency_ms"),
               lambda v: v < MAX_LATENCY_MS),
        _check("app_response", "App response (s)", "< 10, measured on the deployed app", None, lambda v: False),
    ]


def _fmt(value: float | None, digits: int) -> str:
    if value is None:
        return "not measured"
    return "inf" if math.isinf(value) else f"{value:.{digits}f}"


def _signed(value: float | None, digits: int) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return "inf" if math.isinf(value) else f"{value:+.{digits}f}"


def _metric_rows(results: Mapping[str, Any], costs: Mapping[str, Any]) -> list[str]:
    def size_mb(cost: Mapping[str, Any]) -> float | None:
        return None if cost.get("size_bytes") is None else cost["size_bytes"] / 1e6

    rows = [
        (label, lambda name, key=key: results[name]["mean"][key], digits)
        for label, key, digits in (
            ("PSNR (dB)", "psnr", 2),
            ("SSIM", "ssim", 4),
            ("DeltaE2000", "delta_e2000", 3),
            ("LPIPS (AlexNet)", "lpips", 4),
        )
    ] + [
        ("Params", lambda name: costs[name].get("params"), 0),
        ("Size (MB)", lambda name: size_mb(costs[name]), 3),
        ("Latency 1 MP (ms)", lambda name: costs[name].get("latency_ms"), 1),
        ("Peak memory (MB)", lambda name: costs[name].get("peak_rss_mb"), 1),
    ]
    lines = []
    for label, get, digits in rows:
        classical, learned, hybrid = (get(name) for name in REQUIRED)
        differences = [None if v is None or classical is None else v - classical for v in (learned, hybrid)]
        cells = [_fmt(v, digits) for v in (classical, learned, hybrid)]
        cells += [_signed(v, digits) for v in differences]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def render_report(results: Mapping, costs: Mapping, targets: list[dict], provenance: Mapping, usns: str, team: str) -> str:
    lines = [
        usns,
        "",
        f"# Evaluation: Problem P2, Team {team}",
        "",
        "All values are measured by `scripts/evaluate.py`; `results/metrics.json` holds per-image scores.",
        "Peak memory is whole-process RSS, so it includes the Python and runtime footprint.",
        "",
        "## Metrics",
        "",
        "| Metric | Classical | Learned | Hybrid | Learned - Classical | Hybrid - Classical |",
        "|---|---|---|---|---|---|",
        *_metric_rows(results, costs),
        "",
        "## PRD section 8 targets",
        "",
        "| Metric | Required | Measured | Status |",
        "|---|---|---|---|",
    ]
    for t in targets:
        measured = t["measured"]
        shown = "not measured" if measured is None else "inf" if math.isinf(measured) else f"{measured:.4g}"
        lines.append(f"| {t['metric']} | {t['required']} | {shown} | {t['status']} |")
    notes = []
    for name in REQUIRED:
        total = len(results[name]["per_image"])
        if results[name].get("fallbacks"):
            notes.append(f"- {name} fell back to classical on {results[name]['fallbacks']} of {total} images.")
        if results[name].get("nonfinite_outputs"):
            notes.append(f"- {name} produced non-finite output on {results[name]['nonfinite_outputs']} of {total} images.")
    if notes:
        lines += ["", "## Disclosures", "", *notes]
    lines += ["", "## Provenance", ""] + [f"- {key}: {value}" for key, value in provenance.items()]
    return "\n".join(lines) + "\n"


def write_outputs(
    results: Mapping,
    costs: Mapping,
    root: Path,
    provenance: Mapping,
    usns: str = USN_PLACEHOLDER,
    team_id: str = TEAM_PLACEHOLDER,
) -> Path:
    missing = [name for name in REQUIRED if name not in results]
    if missing:
        raise ValueError(f"Results are missing pipelines: {', '.join(missing)}")
    targets = check_targets(results, costs)
    harness.write_json(
        root / "results" / "metrics.json",
        {"provenance": provenance, "pipelines": results, "costs": costs, "deltas": deltas(results), "targets": targets},
    )
    path = root / report_name(team_id)
    path.write_text(render_report(results, costs, targets, provenance, usns, team_id), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = harness.common_parser(__doc__.splitlines()[0])
    parser.add_argument("--usns", default=USN_PLACEHOLDER, help="comma-separated USNs for the first line")
    parser.add_argument("--team-id", default=TEAM_PLACEHOLDER, help="names eval_P2_<team-id>.md")
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per pipeline")
    parser.add_argument("--latency-cpus", type=int, default=2, help="CPUs to pin to while timing")
    args = parser.parse_args(argv)

    config = harness.load_config(args.config)
    samples = harness.make_samples(harness.load_kodak(args.kodak), config, harness.synthesize_fixed)
    pipelines = harness.build_pipelines(args.learned_ckpt, args.hybrid_backend)
    models = {"classical": None, "learned": args.learned_ckpt, "hybrid": args.hybrid_backend}

    # Costs go first, so peak memory does not include the LPIPS network.
    timing_input = harness.tile_to_megapixel(samples[0].raw)
    costs = {
        p.name: harness.measure_cost(p, timing_input, args.repeats, args.latency_cpus) | harness.model_stats(models[p.name])
        for p in pipelines
    }
    results = harness.run_pipelines(pipelines, samples)
    provenance = {
        "dataset": "Kodak24",
        "images": len(samples),
        "condition": config["standard"],
        "seed": config["seed"],
        "learned_ckpt": str(args.learned_ckpt),
        "hybrid_backend": str(args.hybrid_backend),
        "machine": f"{platform.platform()}, {platform.processor()}",
    }
    print(f"wrote {write_outputs(results, costs, args.root, provenance, args.usns, args.team_id)}")


if __name__ == "__main__":
    main()
