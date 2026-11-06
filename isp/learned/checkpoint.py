"""Checkpoint format shared by train.py, LearnedISP and (W4) the hybrid pipeline / ONNX export."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from isp.models.liteisp import STUDENT_WIDTH, LiteISPNet
from isp.models.rawnet import RAWNET_WIDTH, RawNet
from isp.types import InputError

DEFAULT_WIDTH = {"liteisp": STUDENT_WIDTH, "rawnet": RAWNET_WIDTH}


def build_model(name: str, width: int | None = None) -> nn.Module:
    if name not in DEFAULT_WIDTH:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(DEFAULT_WIDTH)}")
    width = width or DEFAULT_WIDTH[name]
    return LiteISPNet(width) if name == "liteisp" else RawNet(width)


def save_checkpoint(path, model: nn.Module, name: str, width: int, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": name, "width": width, "state": model.state_dict(), **extra}, path)


def load_checkpoint(path, expect: str | None = None) -> tuple[nn.Module, dict]:
    """Returns (eval-mode model on CPU, metadata). Raises InputError for missing/incompatible files."""
    if not Path(path).is_file():
        raise InputError(f"Model file not found: {path}")
    meta = torch.load(path, map_location="cpu", weights_only=False)
    if expect and meta.get("model") != expect:
        raise InputError(f"Expected a {expect} checkpoint but {path} holds {meta.get('model')!r}.")
    model = build_model(meta["model"], meta["width"])
    model.load_state_dict(meta["state"])
    return model.eval(), meta
