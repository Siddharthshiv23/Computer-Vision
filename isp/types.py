"""Shared types. Every workstream codes against these; change them only via the lead."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class InputError(ValueError):
    """Bad user input. The message is shown to the user, so keep it friendly."""


@dataclass
class RawImage:
    """Normalised Bayer mosaic.

    bayer: float32 (H, W), even H and W, RGGB, black level removed, in [0, 1].
    wb_gains: optional as-shot (r, g, b) gains. ccm: optional 3x3 camera RGB -> linear sRGB.
    """

    bayer: np.ndarray
    wb_gains: np.ndarray | None = None
    ccm: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """rgb: float32 (H, W, 3) sRGB in [0, 1]. stages: name -> displayable image, in order."""

    rgb: np.ndarray
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


class Pipeline(Protocol):
    name: str

    def run(self, raw: RawImage) -> PipelineResult: ...
