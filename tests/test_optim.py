"""ONNX export, INT8 PTQ, FP16 and pruning, all on an UNTRAINED student: these test the machinery, not quality."""
import os

import numpy as np
import pytest
import torch
from torch import nn

from isp.learned.checkpoint import build_model, save_checkpoint
from isp.models.liteisp import STUDENT_WIDTH, LiteISPNet

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from isp.optim import export_onnx, prune_model, prune_sweep, quantize_int8  # noqa: E402


def _active_student(width=STUDENT_WIDTH, seed=0):
    """Zero-init heads make a fresh LiteISPNet an identity map, which would hide quantisation error."""
    torch.manual_seed(seed)
    net = LiteISPNet(width)
    for layer in (net.out, net.colour_head[-1]):
        nn.init.normal_(layer.weight, std=0.05)
        nn.init.normal_(layer.bias, std=0.01)
    return net.eval()


def _images(n, h=96, w=128, seed=0):
    """Smooth random 'sRGB + sigma' inputs resembling the real network input."""
    rng = np.random.default_rng(seed)
    import cv2

    out = []
    for _ in range(n):
        rgb = np.stack([cv2.GaussianBlur(rng.random((h, w), dtype=np.float32), (0, 0), 3) for _ in range(3)])
        rgb = (rgb - rgb.min()) / np.ptp(rgb)
        sigma = np.full((1, h, w), 0.02, np.float32)
        out.append(np.concatenate([rgb, sigma])[None].astype(np.float32))
    return out


@pytest.fixture(scope="module")
def student_ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "student.pt"
    save_checkpoint(path, _active_student(), "liteisp", STUDENT_WIDTH)
    return str(path)


def _run(path, x):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})[0]


def test_export_onnx_matches_torch_and_accepts_any_size(student_ckpt, tmp_path):
    out = export_onnx(student_ckpt, str(tmp_path / "s.onnx"))
    net = _active_student()
    for x in _images(1, 64, 96) + _images(1, 50, 70, seed=1):
        with torch.no_grad():
            want = net(torch.from_numpy(x)).numpy()
        assert np.abs(_run(out, x) - want).max() < 1e-4


def test_int8_is_small_and_close_to_fp32(student_ckpt, tmp_path):
    fp32 = export_onnx(student_ckpt, str(tmp_path / "s.onnx"))
    int8 = quantize_int8(fp32, str(tmp_path / "s_int8.onnx"), _images(8, seed=10))
    assert os.path.getsize(int8) < 0.5 * 1024 * 1024
    assert os.path.getsize(int8) < 0.5 * os.path.getsize(fp32)
    x = _images(1, 96, 128, seed=99)[0]
    err = np.abs(_run(int8, x) - _run(fp32, x))
    assert err.mean() < 0.01 and err.max() < 0.05  # outputs live in about [0, 1]


def test_quantize_rejects_empty_calibration(student_ckpt, tmp_path):
    from isp.types import InputError

    fp32 = export_onnx(student_ckpt, str(tmp_path / "s.onnx"))
    with pytest.raises(InputError):
        quantize_int8(fp32, str(tmp_path / "q.onnx"), [])


def test_fp16_export_is_half_size_with_float_io(student_ckpt, tmp_path):
    fp32 = export_onnx(student_ckpt, str(tmp_path / "s.onnx"))
    fp16 = export_onnx(student_ckpt, str(tmp_path / "s16.onnx"), fp16=True)
    assert os.path.getsize(fp16) < 0.6 * os.path.getsize(fp32)
    x = _images(1, 64, 64)[0]
    got = _run(fp16, x)
    assert got.dtype == np.float32
    assert np.abs(got - _run(fp32, x)).max() < 0.02


def test_export_missing_checkpoint_is_input_error(tmp_path):
    from isp.types import InputError

    with pytest.raises(InputError):
        export_onnx(str(tmp_path / "nope.pt"), str(tmp_path / "x.onnx"))


def test_prune_model_zeroes_requested_fraction_of_conv_weights():
    net = _active_student()
    pruned = prune_model(net, 0.5)
    w = torch.cat([m.weight.flatten() for m in pruned.modules() if isinstance(m, (nn.Conv2d, nn.Linear))])
    assert abs((w == 0).float().mean().item() - 0.5) < 0.02
    assert not any(hasattr(m, "weight_mask") for m in pruned.modules())  # masks folded in
    assert (net.head[0].weight == 0).float().mean() < 0.01  # original untouched


def test_prune_sweep_reports_monotonic_sparsity_and_drift(student_ckpt):
    rows = prune_sweep(student_ckpt, [0.0, 0.3, 0.6], probe=_images(2))
    assert [r["ratio"] for r in rows] == [0.0, 0.3, 0.6]
    assert rows[0]["output_drift"] == pytest.approx(0.0, abs=1e-7)
    assert rows[0]["sparsity"] < rows[1]["sparsity"] < rows[2]["sparsity"]
    assert rows[2]["output_drift"] >= rows[1]["output_drift"] > 0
    assert all(r["params"] == rows[0]["params"] for r in rows)


def test_prune_sweep_uses_custom_score_function(student_ckpt):
    rows = prune_sweep(student_ckpt, [0.2], score=lambda model: 42.0)
    assert rows[0]["score"] == 42.0
