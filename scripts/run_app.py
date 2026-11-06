"""Same commands as the Makefile, for machines without make.

    python scripts/run_app.py app
    python scripts/run_app.py demo --input photo.jpg
    python scripts/run_app.py test
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def demo(input_path: str | None, out_dir: Path) -> int:
    """Run every available pipeline on one file and save real outputs only."""
    from PIL import Image

    from app import adapter, processing

    slots = adapter.load_slots()
    for slot in slots:
        print(f"{slot.name}: {'ready' if slot.pipeline else slot.reason}")
    if input_path is None:
        print("Pass an image with --input PATH (make demo INPUT=PATH).")
        return 2
    if not Path(input_path).is_file():
        print(f"No such file: {input_path}")
        return 2
    load, load_reason = adapter.load_loader()
    metrics, metrics_reason = adapter.load_metrics()
    with tempfile.TemporaryDirectory() as scratch:
        # develop() deletes what it reads, which must never be the user's own file.
        copy = shutil.copy(input_path, scratch)
        *_, report = processing.develop(
            copy, None, slots=slots, load=load, load_reason=load_reason, metrics=metrics, metrics_reason=metrics_reason
        )
    if report.error:
        print(report.error)
        return 1
    if report.note:
        print(report.note)
    written = 0
    for outcome in report.outcomes:
        if outcome.rgb is None:
            print(f"{outcome.name}: {outcome.message}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(outcome.rgb).save(out_dir / f"{outcome.name}.png")
        written += 1
        print(f"{outcome.name}: {outcome.total_ms:.0f} ms, wrote {out_dir / (outcome.name + '.png')}")
    return 0 if written else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["app", "demo", "test"])
    parser.add_argument("--input", help="image for the demo command")
    args = parser.parse_args()
    if args.command == "app":
        return subprocess.call([sys.executable, "-m", "app.app"], cwd=ROOT)
    if args.command == "test":
        return subprocess.call([sys.executable, "-m", "pytest"], cwd=ROOT)
    return demo(args.input, ROOT / "runs" / "demo")


if __name__ == "__main__":
    sys.exit(main())
