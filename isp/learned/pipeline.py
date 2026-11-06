"""LearnedISP: the learned-only baseline (packed Bayer -> RawNet -> sRGB). Implements isp.types.Pipeline."""
from __future__ import annotations

import time

import numpy as np
import torch

from isp.learned.checkpoint import load_checkpoint
from isp.learned.inputs import rawnet_input
from isp.types import PipelineResult, RawImage


class LearnedISP:
    name = "learned"

    def __init__(self, ckpt: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        model, self.meta = load_checkpoint(ckpt, expect="rawnet")
        self.model = model.to(self.device)

    @torch.no_grad()
    def run(self, raw: RawImage) -> PipelineResult:
        t0 = time.perf_counter()
        x = rawnet_input(raw).unsqueeze(0).to(self.device)
        y = self.model(x)[0].permute(1, 2, 0).cpu().numpy()
        finite = bool(np.isfinite(y).all())
        rgb = np.clip(np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        ms = (time.perf_counter() - t0) * 1000
        return PipelineResult(rgb=rgb, stages={"output": rgb}, timings_ms={"network": ms},
                              info={"nonfinite_output": not finite})
