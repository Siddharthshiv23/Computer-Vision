"""Network backends for the hybrid pipeline: a .pt checkpoint (PyTorch) or an .onnx file (ONNX Runtime)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from isp.learned.checkpoint import load_checkpoint
from isp.types import InputError

LITEISP_IN_CH = 4


class Backend:
    """Callable (1, 4, h, w) float32 -> (3, h, w) float32. Hides which runtime is underneath."""

    def __init__(self, path: str, threads: int | None = None, colour_head: bool = True) -> None:
        self.path = str(path)
        self.session = self.model = None
        if self.path.endswith(".onnx"):
            self._load_onnx(threads, colour_head)
        else:
            self._load_torch(colour_head)

    def _load_onnx(self, threads: int | None, colour_head: bool) -> None:
        import onnxruntime as ort

        if not Path(self.path).is_file():
            raise InputError(f"Model file not found: {self.path}")
        if not colour_head:
            raise InputError("The colour_head ablation needs a .pt checkpoint: it is fused into the ONNX graph.")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(self.path, opts, providers=["CPUExecutionProvider"])
        shape = self.session.get_inputs()[0].shape
        if shape[1] != LITEISP_IN_CH:
            raise InputError(f"{self.path} is not a LiteISPNet model (expected 4 input channels, got {shape[1]}).")

    def _load_torch(self, colour_head: bool) -> None:
        self.model, _ = load_checkpoint(self.path, expect="liteisp")
        if not colour_head:  # zeroing the last layer makes the matrix the identity and the bias zero
            for p in self.model.colour_head[-1].parameters():
                torch.nn.init.zeros_(p)

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.session is not None:
            return self.session.run(None, {self.session.get_inputs()[0].name: x})[0][0]
        return self.model(torch.from_numpy(x))[0].numpy()
