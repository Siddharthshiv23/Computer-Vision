"""Train every model in order: LiteISPNet teacher, distilled student, RawNet baseline.

Safe to re-run: a stage with a last.pt resumes from it, and a finished stage exits at once.
Windows is asked not to sleep while this runs; the request ends with the process.
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake() -> None:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def stages(args) -> list[list[str]]:
    teacher = str(Path(args.runs_dir) / "teacher" / "best.pt")
    return [
        ["--model", "liteisp", "--name", "teacher", "--width", str(args.teacher_width), "--epochs", str(args.teacher_epochs)],
        ["--model", "liteisp", "--name", "student", "--teacher", teacher, "--epochs", str(args.student_epochs)],
        ["--model", "rawnet", "--name", "rawnet", "--epochs", str(args.rawnet_epochs)],
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-epochs", type=int, default=600)
    p.add_argument("--student-epochs", type=int, default=800)
    p.add_argument("--rawnet-epochs", type=int, default=600)
    p.add_argument("--teacher-width", type=int, default=56)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args()
    keep_awake()
    for stage in stages(args):
        name = stage[stage.index("--name") + 1]
        cmd = [sys.executable, "-m", "scripts.train", *stage, "--workers", str(args.workers), "--runs-dir", args.runs_dir]
        if (Path(args.runs_dir) / name / "last.pt").is_file():
            cmd.append("--resume")
        print(f"[{time.strftime('%H:%M:%S')}] start {name}: {' '.join(cmd[1:])}", flush=True)
        code = subprocess.call(cmd)
        print(f"[{time.strftime('%H:%M:%S')}] end {name}: exit {code}", flush=True)
        if code:
            sys.exit(code)


if __name__ == "__main__":
    main()
