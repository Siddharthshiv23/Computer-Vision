"""LearnedISP: the learned-only baseline (packed Bayer -> RawNet -> sRGB). Implements isp.types.Pipeline."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from isp.learned.checkpoint import load_checkpoint
from isp.learned.inputs import rawnet_input
from isp.types import InputError, PipelineResult, RawImage


class LearnedISP:
    name = "learned"

    def __init__(self, ckpt: str, device: str = "cpu") -> None:
        """ckpt is a .pt checkpoint or an exported .onnx; deployment ships only the .onnx."""
        self.device = torch.device(device)
        self.model = self.session = None
        if str(ckpt).endswith(".onnx"):
            import onnxruntime as ort

            if not Path(ckpt).is_file():
                raise InputError(f"Model file not found: {ckpt}")
            self.session = ort.InferenceSession(str(ckpt), providers=["CPUExecutionProvider"])
            self.meta = {"model": "rawnet", "format": "onnx"}
        else:
            model, self.meta = load_checkpoint(ckpt, expect="rawnet")
            self.model = model.to(self.device)

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> np.ndarray:
        """(1, 5, H/2, W/2) -> (H, W, 3)."""
        if self.session is not None:
            name = self.session.get_inputs()[0].name
            return self.session.run(None, {name: x.numpy()})[0][0].transpose(1, 2, 0)
        return self.model(x.to(self.device))[0].permute(1, 2, 0).cpu().numpy()

    def run(self, raw: RawImage) -> PipelineResult:
        t0 = time.perf_counter()
        y = self._forward(rawnet_input(raw).unsqueeze(0))
        finite = bool(np.isfinite(y).all())
        rgb = np.clip(np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        ms = (time.perf_counter() - t0) * 1000
        return PipelineResult(rgb=rgb, stages={"output": rgb}, timings_ms={"network": ms},
                              info={"nonfinite_output": not finite})
