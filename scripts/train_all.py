"""Train every model in order: LiteISPNet teacher, distilled student, RawNet baseline.

Safe to re-run: a stage with a last.pt resumes from it, and a finished stage exits at once.
Windows is asked not to sleep while this runs; the request ends with the process.
"""
from __future__ import annotations

import argparse
import ctypes
import os
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
    p.add_argument("--teacher-epochs", type=int, default=38)
    p.add_argument("--student-epochs", type=int, default=58)
    p.add_argument("--rawnet-epochs", type=int, default=40)
    p.add_argument("--teacher-width", type=int, default=56)
    p.add_argument("--workers", type=int, default=8)  # measured: 12 is slower than 8 on this 16-thread CPU
    p.add_argument("--repeats", type=int, default=32)
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args()
    keep_awake()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for stage in stages(args):
        name = stage[stage.index("--name") + 1]
        cmd = [sys.executable, "-m", "scripts.train", *stage, "--workers", str(args.workers),
               "--repeats", str(args.repeats), "--runs-dir", args.runs_dir]
        if (Path(args.runs_dir) / name / "last.pt").is_file():
            cmd.append("--resume")
        print(f"[{time.strftime('%H:%M:%S')}] start {name}: {' '.join(cmd[1:])}", flush=True)
        code = subprocess.call(cmd, env=env)
        print(f"[{time.strftime('%H:%M:%S')}] end {name}: exit {code}", flush=True)
        if code:
            sys.exit(code)


if __name__ == "__main__":
    main()
