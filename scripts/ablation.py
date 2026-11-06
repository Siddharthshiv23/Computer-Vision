"""Ablate the hybrid pipeline one toggle at a time and report deltas against the full pipeline.

Kodak24 results here are for reporting only. Choosing a configuration from them would break the
rule that the test set is never used for selection.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from isp.eval import harness  # noqa: E402
from isp.eval.harness import Sample  # noqa: E402
from isp.types import Pipeline  # noqa: E402

# Each variant changes one thing from the full pipeline. The full pipeline leaves raw_denoise off
# because the network does the denoising (PRD section 6), so that toggle turns it on.
VARIANTS: dict[str, dict[str, Any]] = {
    "no_colour_head": {"colour_head": False},
    "no_sharpen": {"sharpen": False},
    "no_ccm": {"ccm": False},
    "awb_gray_world": {"awb": "gray_world"},
    "awb_shades_of_gray": {"awb": "shades_of_gray"},
    "awb_gray_edge": {"awb": "gray_edge"},
    "awb_white_patch": {"awb": "white_patch"},
    "no_noise_map": {"noise_map": False},
    "raw_denoise": {"raw_denoise": True},
}


def hybrid_factory(backend: str) -> Callable[[dict[str, Any]], Pipeline]:
    """Needs HybridISP(backend, ablation=toggles). Fails loudly until W4 provides that keyword."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # evaluation is CPU only

    def factory(toggles: dict[str, Any]) -> Pipeline:
        from isp.hybrid.pipeline import HybridISP

        return HybridISP(backend, ablation=toggles) if toggles else HybridISP(backend)

    return factory


def run(
    factory: Callable[[dict[str, Any]], Pipeline],
    samples: Sequence[Sample],
    metric_fns: Mapping[str, Callable] = harness.METRICS,
) -> dict[str, Any]:
    def score(toggles: dict[str, Any]) -> dict[str, Any]:
        pipeline = factory(toggles)
        return harness.run_pipelines([pipeline], samples, metric_fns)[pipeline.name]

    full = score({})
    variants = {}
    for name, toggles in VARIANTS.items():
        entry = score(toggles)
        variants[name] = {
            "toggles": toggles,
            "mean": entry["mean"],
            "delta": {m: entry["mean"][m] - full["mean"][m] for m in full["mean"]},
            "fallbacks": entry["fallbacks"],
            # Every score matching on every image means the toggle changed nothing, which
            # points at an unwired toggle rather than at a stage that does not matter.
            "no_effect": entry["per_image"] == full["per_image"],
        }
    return {"images": len(samples), "full": {"mean": full["mean"], "fallbacks": full["fallbacks"]}, "variants": variants}


def plot(out: Mapping[str, Any], path: Path) -> None:
    names = list(out["variants"])[::-1]
    metrics = list(out["full"]["mean"])
    labels = [f"{n} (no effect)" if out["variants"][n]["no_effect"] else n for n in names]
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(2.9 * len(metrics), 0.32 * len(names) + 1.2), sharey=True, layout="constrained"
    )
    titles = {"psnr": "PSNR (dB)", "ssim": "SSIM", "delta_e2000": "DeltaE2000", "lpips": "LPIPS"}
    for ax, metric in zip(axes, metrics):
        better = "higher" if metric in harness.HIGHER_IS_BETTER else "lower"
        ax.barh(labels, [out["variants"][n]["delta"][metric] for n in names], color="#2a78d6", height=0.6)
        ax.axvline(0, color="#52514e", linewidth=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  # small deltas have long tick labels
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(length=0, labelsize=8, colors="#52514e")
        ax.set_title(f"{titles.get(metric, metric)}, {better} is better", fontsize=9, color="#0b0b0b")
    fig.suptitle("Change from the full hybrid pipeline", fontsize=11, color="#0b0b0b")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def write_outputs(out: Mapping[str, Any], results_dir: Path) -> None:
    harness.write_json(results_dir / "ablation.json", out)
    plot(out, results_dir / "ablation.png")


def main(argv: list[str] | None = None) -> None:
    args = harness.common_parser(__doc__.splitlines()[0]).parse_args(argv)
    config = harness.load_config(args.config)
    samples = harness.make_samples(harness.load_kodak(args.kodak), config, harness.synthesize_fixed)
    write_outputs(run(hybrid_factory(args.hybrid_backend), samples), args.root / "results")


if __name__ == "__main__":
    main()
