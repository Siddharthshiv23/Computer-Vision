"""The only place the app touches isp pipelines, IO and metrics.

Everything is resolved by name so the app starts while modules are still landing.
Anything missing comes back as None plus a reason to show the user; nothing is
ever substituted.
"""
from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from isp.types import Pipeline

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
NOT_YET = "Not available yet."
METRIC_NAMES = ("psnr", "ssim", "delta_e2000")

# name, label, module, class, weights env var, default weights file
PIPELINES = (
    ("classical", "Classical", "isp.classical.pipeline", "ClassicalISP", None, None),
    ("learned", "Learned", "isp.learned.pipeline", "LearnedISP", "ISP_LEARNED_WEIGHTS", "learned.onnx"),
    ("hybrid", "Hybrid", "isp.hybrid.pipeline", "HybridISP", "ISP_HYBRID_WEIGHTS", "hybrid_int8.onnx"),
)


@dataclass(frozen=True)
class Slot:
    name: str
    label: str
    pipeline: Pipeline | None
    reason: str


def resolve(module: str, attr: str) -> tuple[Any, str]:
    """Return (object, "") or (None, reason)."""
    try:
        found = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and (module == exc.name or module.startswith(exc.name + ".")):
            return None, NOT_YET
        return None, f"Needs the Python package {exc.name}, which is not installed."
    except Exception:
        log.exception("importing %s failed", module)
        return None, "Could not start: this module failed to load."
    obj = getattr(found, attr, None)
    return (obj, "") if obj is not None else (None, NOT_YET)


def _load_slot(name: str, label: str, module: str, attr: str, env: str | None, default: str | None) -> Slot:
    cls, reason = resolve(module, attr)
    if cls is None:
        return Slot(name, label, None, reason)
    args: tuple[str, ...] = ()
    if env:
        weights = Path(os.environ.get(env) or ROOT / "weights" / default)
        if not weights.is_file():
            return Slot(name, label, None, f"{NOT_YET} Model weights have not been added.")
        args = (str(weights),)
    try:
        return Slot(name, label, cls(*args), "")
    except Exception:
        log.exception("constructing %s failed", name)
        return Slot(name, label, None, "Could not start: the model failed to load.")


def load_slots() -> list[Slot]:
    return [_load_slot(*spec) for spec in PIPELINES]


def load_loader() -> tuple[Callable | None, str]:
    fn, reason = resolve("isp.io", "load_input")
    return fn, f"Image loading: {reason}" if reason else ""


def load_metrics() -> tuple[dict[str, Callable] | None, str]:
    fns = {}
    for name in METRIC_NAMES:
        fn, reason = resolve("isp.eval.metrics", name)
        if fn is None:
            return None, f"Quality metrics: {reason}"
        fns[name] = fn
    return fns, ""
