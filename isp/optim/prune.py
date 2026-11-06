"""L1 magnitude pruning sweep (F-OPT-3)."""
from __future__ import annotations

import copy
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune

from isp.learned.checkpoint import load_checkpoint
from isp.types import InputError


def _prunable(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def prune_model(model: nn.Module, ratio: float) -> nn.Module:
    """Copy of `model` with the smallest `ratio` of conv/linear weights (by |w|, one global
    threshold) set to zero. Masks are folded into the weights, so the result trains and exports normally.

    Why global rather than per layer: layers differ in how much they can lose, and a global L1
    threshold lets the small, redundant ones absorb most of the pruning. Biases are kept.
    Note this is unstructured: zeros do not shrink a dense ONNX file or speed up dense CPU kernels.
    """
    if not 0.0 <= ratio < 1.0:
        raise InputError(f"Pruning ratio must be in [0, 1), got {ratio}.")
    pruned = copy.deepcopy(model)
    layers = _prunable(pruned)
    if ratio > 0:
        prune.global_unstructured([(m, "weight") for m in layers], pruning_method=prune.L1Unstructured,
                                  amount=ratio)
        for m in layers:
            prune.remove(m, "weight")
    return pruned


def _sparsity(model: nn.Module) -> float:
    w = [m.weight for m in _prunable(model)]
    return float(sum((x == 0).sum().item() for x in w) / sum(x.numel() for x in w))


def _drift(base: nn.Module, model: nn.Module, probe: Sequence[np.ndarray]) -> float:
    """Mean |output change| versus the unpruned model on probe inputs. A cheap sanity signal, NOT quality."""
    with torch.no_grad():
        diffs = [(model(torch.from_numpy(x)) - base(torch.from_numpy(x))).abs().mean().item() for x in probe]
    return float(np.mean(diffs))


def _default_probe(in_ch: int) -> list[np.ndarray]:
    g = torch.Generator().manual_seed(0)
    return [torch.rand(1, in_ch, 64, 64, generator=g).numpy()]


def prune_sweep(ckpt: str, ratios: Sequence[float], probe: Sequence[np.ndarray] | None = None,
                score: Callable[[nn.Module], float] | None = None) -> list[dict]:
    """Prune a checkpoint at each ratio and report one row per ratio.

    Row: ratio, sparsity (measured fraction of zero weights), params (unchanged, dense count),
    output_drift (vs the unpruned model on `probe`), and `score` if a `score(model) -> float`
    callable is given (e.g. validation PSNR; supplied by the caller because it needs the dataset).
    Without `score` the sweep says how much the output moves, which is not a quality measure.
    """
    model, meta = load_checkpoint(ckpt)
    in_ch = model.head[0].in_channels if hasattr(model, "head") else 5
    probe = probe if probe is not None else _default_probe(in_ch)
    params = sum(p.numel() for p in model.parameters())
    rows = []
    for ratio in ratios:
        pruned = prune_model(model, ratio)
        row = {"ratio": float(ratio), "sparsity": _sparsity(pruned), "params": params,
               "output_drift": _drift(model, pruned, probe)}
        if score is not None:
            row["score"] = float(score(pruned))
        rows.append(row)
    return rows
