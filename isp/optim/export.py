"""ONNX export, FP16 conversion and INT8 static post-training quantisation (F-OPT-2, F-OPT-3)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from isp.learned.checkpoint import load_checkpoint
from isp.types import InputError

_IN_CH = {"liteisp": 4, "rawnet": 5}
_INPUT, _OUTPUT = "x", "y"
_OPSET = 17  # the opset the models were validated against (tests/test_export.py)


def export_onnx(ckpt: str, out: str, fp16: bool = False) -> str:
    """Export a checkpoint to ONNX with dynamic height and width. Returns `out`.

    Why dynamic axes: one file serves 512 px previews, 1 MP and tiled 12 MP. fp16=True stores the
    weights in half precision (half the file) but keeps float32 inputs and outputs, so callers
    never change; compute on CPU still runs in float32 after ORT's own up-cast, so this is a size
    optimisation, not a speed one.
    """
    model, meta = load_checkpoint(ckpt)
    dummy = torch.rand(1, _IN_CH[meta["model"]], 64, 64)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(out), opset_version=_OPSET, input_names=[_INPUT],
                      output_names=[_OUTPUT], dynamo=False,
                      dynamic_axes={_INPUT: {2: "h", 3: "w"}, _OUTPUT: {2: "h", 3: "w"}})
    if fp16:
        _to_fp16(out)
    return str(out)


def _to_fp16(path: str) -> None:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    converted = convert_float_to_float16(onnx.load(path), keep_io_types=True)
    onnx.save(converted, path)


class _Calibration:
    """Feeds calibration arrays to ORT's quantiser one at a time."""

    def __init__(self, name: str, arrays: list[np.ndarray]) -> None:
        self._iter = iter([{name: a} for a in arrays])

    def get_next(self):
        return next(self._iter, None)


def quantize_int8(onnx_in: str, onnx_out: str, calib: Iterable[np.ndarray]) -> str:
    """Static INT8 PTQ (QDQ format, per-channel int8 weights, uint8 activations). Returns `onnx_out`.

    calib: float32 arrays shaped (1, C, H, W) drawn from the real input distribution; they set the
    activation ranges, so synthetic noise in place of real images would give poor ranges.
    Why per-channel weights: the depthwise 3x3 convs have very different scales per channel, and one
    scale for the whole tensor crushes the small ones.
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in calib]
    if not arrays:
        raise InputError("INT8 calibration needs at least one representative input.")
    import onnxruntime as ort

    name = ort.InferenceSession(onnx_in, providers=["CPUExecutionProvider"]).get_inputs()[0].name
    Path(onnx_out).parent.mkdir(parents=True, exist_ok=True)
    quantize_static(onnx_in, onnx_out, _calibration_reader(name, arrays), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    return str(onnx_out)


def _calibration_reader(name: str, arrays: list[np.ndarray]):
    """CalibrationDataReader subclass built lazily so importing this module needs no quantiser."""
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(_Calibration, CalibrationDataReader):
        pass

    return Reader(name, arrays)
