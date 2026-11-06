"""ISO x EV x colour-cast sweep with degradation curves for each pipeline (F-EV-3)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from isp.eval import harness  # noqa: E402
from isp.types import Pipeline  # noqa: E402

# Identity follows the pipeline: fixed hue plus a marker shape, so it never rests on colour alone.
STYLE = {
    "classical": {"color": "#2a78d6", "marker": "o"},
    "learned": {"color": "#eb6834", "marker": "s"},
    "hybrid": {"color": "#1baf7a", "marker": "^"},
}
AXIS_TITLES = {"iso": "ISO", "ev": "Exposure", "cast": "Colour cast"}
METRIC_LABELS = {
    "psnr": "PSNR (dB), higher is better",
    "ssim": "SSIM, higher is better",
    "delta_e2000": "DeltaE2000, lower is better",
    "lpips": "LPIPS, lower is better",
}


def sweep(
    pipelines: Sequence[Pipeline],
    images: Mapping[str, Any],
    config: Mapping[str, Any],
    synth: Callable,
    metric_fns: Mapping[str, Callable] = harness.METRICS,
) -> dict[str, Any]:
    """Score every pipeline at every grid point. Inputs are shared within a grid point."""
    grid = []
    axes = config["axes"]
    for iso, ev, cast in itertools.product(axes["iso"], axes["ev"], axes["cast"]):
        samples = harness.make_samples(images, config, synth, iso=iso, ev=ev, cast=cast)
        results = harness.run_pipelines(pipelines, samples, metric_fns)
        grid.append({
            "iso": iso,
            "ev": ev,
            "cast": cast,
            "pipelines": {name: entry["mean"] for name, entry in results.items()},
            "fallbacks": {name: entry["fallbacks"] for name, entry in results.items()},
        })
    return {"images": list(images), "grid": grid}


def curves(out: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """One curve per axis: that axis varies, the other two stay at their standard level."""
    standard = config["standard"]
    result = {}
    for axis, levels in config["axes"].items():
        points = [
            next(g for g in out["grid"] if all(g[a] == (level if a == axis else standard[a]) for a in standard))
            for level in levels
        ]
        entry: dict[str, Any] = {"levels": list(levels)}
        for name in points[0]["pipelines"]:
            entry[name] = {m: [p["pipelines"][name][m] for p in points] for m in points[0]["pipelines"][name]}
        result[axis] = entry
    return result


def plot_curves(data: Mapping[str, Any], path: Path) -> None:
    axes_names = list(data)
    levels = data[axes_names[0]]
    names = [k for k in levels if k != "levels"]
    metrics = list(levels[names[0]])
    fig, grid = plt.subplots(
        len(metrics), len(axes_names), figsize=(3.6 * len(axes_names), 2.5 * len(metrics)),
        sharey="row", squeeze=False, layout="constrained",
    )
    for row, metric in enumerate(metrics):
        for col, axis in enumerate(axes_names):
            ax = grid[row][col]
            labels = data[axis]["levels"]
            for name in names:
                style = STYLE.get(name, {})
                ax.plot(labels, data[axis][name][metric], label=name, linewidth=2, markersize=7,
                        markeredgecolor="white", markeredgewidth=1, **style)
            ax.margins(y=0.1)  # keeps markers on the extremes from being clipped
            ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(length=0, labelsize=8, colors="#52514e")
            if row == 0:
                ax.set_title(AXIS_TITLES.get(axis, axis), fontsize=11, color="#0b0b0b")
            if col == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=8, color="#52514e")
    handles, labels = grid[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(names), frameon=False, fontsize=9)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def write_outputs(out: Mapping[str, Any], config: Mapping[str, Any], results_dir: Path) -> None:
    data = curves(out, config)
    harness.write_json(results_dir / "robustness.json", {**out, "curves": data})
    plot_curves(data, results_dir / "robustness_curves.png")


def main(argv: list[str] | None = None) -> None:
    parser = harness.common_parser(__doc__.splitlines()[0])
    parser.add_argument("--images", type=int, help="use only the first N images; the output records which")
    args = parser.parse_args(argv)

    config = harness.load_config(args.config)
    images = harness.load_kodak(args.kodak)
    images = dict(list(images.items())[: args.images])
    pipelines = harness.build_pipelines(args.learned_ckpt, args.hybrid_backend)
    write_outputs(sweep(pipelines, images, config, harness.synthesize_fixed), config, args.root / "results")


if __name__ == "__main__":
    main()
