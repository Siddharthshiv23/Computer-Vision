"""HybridISP with UNTRAINED tiny networks. These check plumbing and invariants, never image quality."""
import numpy as np
import pytest
import torch
from torch import nn

from isp.classical.pipeline import ClassicalISP
from isp.hybrid.controller import AdaptiveController
from isp.hybrid.pipeline import HybridISP
from isp.learned.checkpoint import build_model, save_checkpoint
from isp.types import InputError, RawImage

ALWAYS_NETWORK = AdaptiveController(clean_sigma=0.0)  # tests must exercise the network on any frame


def _mosaic(h=96, w=128, a=0.01, b=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    clean = 0.05 + 0.3 * (0.5 + 0.5 * np.sin(xx / 9.0) * np.cos(yy / 13.0))
    noisy = rng.poisson(clean / a) * a + rng.normal(0, np.sqrt(b), clean.shape)
    return np.clip(noisy, 0, 1).astype(np.float32)


def _raw(**kw):
    return RawImage(bayer=_mosaic(**kw))


def _make_ckpt(path, width=8, colour_head=True, nan=False, seed=0):
    torch.manual_seed(seed)
    net = build_model("liteisp", width)
    for layer in (net.out, net.colour_head[-1]):
        nn.init.normal_(layer.weight, std=0.05 if (colour_head or layer is net.out) else 0.0)
        nn.init.normal_(layer.bias, std=0.01 if colour_head or layer is net.out else 0.0)
    if nan:
        net.out.bias.data[:] = float("nan")
    save_checkpoint(path, net.eval(), "liteisp", width)
    return str(path)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return _make_ckpt(tmp_path_factory.mktemp("w") / "student.pt")


@pytest.fixture(scope="module")
def onnx_path(ckpt, tmp_path_factory):
    from isp.optim import export_onnx

    return export_onnx(ckpt, str(tmp_path_factory.mktemp("o") / "student.onnx"))


def _hybrid(backend, **kw):
    kw.setdefault("controller", ALWAYS_NETWORK)
    return HybridISP(backend, **kw)


def test_basic_contract_and_stage_timings(ckpt):
    pipe = _hybrid(ckpt)
    res = pipe.run(_raw())
    assert pipe.name == "hybrid"
    assert res.rgb.shape == (96, 128, 3) and res.rgb.dtype == np.float32
    assert res.rgb.min() >= 0 and res.rgb.max() <= 1 and np.isfinite(res.rgb).all()
    assert {"front_end", "network", "sharpen", "total"} <= set(res.timings_ms)
    assert all(v >= 0 for v in res.timings_ms.values())
    assert {"demosaic", "network", "output"} <= set(res.stages)
    assert res.info["fallback"] is False and res.info["model"] == "student"
    for key in ("noise_a", "noise_b", "exposure", "decision"):
        assert key in res.info


def test_onnx_backend_matches_pt_backend(ckpt, onnx_path):
    raw = _raw()
    assert np.abs(_hybrid(ckpt).run(raw).rgb - _hybrid(onnx_path).run(raw).rgb).max() < 2e-3


def test_exposure_is_passed_to_the_noise_map(ckpt, monkeypatch):
    seen = []
    import isp.hybrid.pipeline as hp

    real = hp.liteisp_input
    monkeypatch.setattr(hp, "liteisp_input", lambda rgb, a, b, e=1.0: (seen.append(e), real(rgb, a, b, e))[1])
    res = _hybrid(ckpt).run(_raw())
    assert seen == [res.info["exposure"]]


def test_nan_network_output_falls_back_to_classical(tmp_path):
    bad = _make_ckpt(tmp_path / "bad.pt", nan=True)
    raw = _raw()
    res = _hybrid(bad).run(raw)
    assert res.info["fallback"] is True and "non-finite" in res.info["fallback_reason"]
    assert np.isfinite(res.rgb).all()
    assert np.array_equal(res.rgb, ClassicalISP().run(raw).rgb)


def test_clean_frame_uses_classical_path_without_calling_network(ckpt):
    res = HybridISP(ckpt).run(RawImage(bayer=np.full((64, 64), 0.2, np.float32)))
    assert res.info["model"] == "classical" and res.info["fallback"] is False
    assert res.timings_ms.get("network", 0.0) == 0.0


def test_teacher_is_used_only_when_available_and_noise_is_extreme(ckpt, tmp_path):
    teacher = _make_ckpt(tmp_path / "teacher.pt", width=12, seed=1)
    noisy = _raw(a=0.05, b=2e-3)
    ctl = AdaptiveController(clean_sigma=0.0, extreme_sigma=0.0)
    assert _hybrid(ckpt, controller=ctl, teacher=teacher).run(noisy).info["model"] == "teacher"
    assert _hybrid(ckpt, controller=ctl).run(noisy).info["model"] == "student"  # no teacher supplied


def test_rejects_odd_and_non_finite_mosaics(ckpt):
    pipe = _hybrid(ckpt)
    with pytest.raises(InputError):
        pipe.run(RawImage(bayer=np.zeros((33, 32), np.float32)))
    bad = _mosaic()
    bad[2, 2] = np.nan
    with pytest.raises(InputError):
        pipe.run(RawImage(bayer=bad))


def test_missing_or_wrong_backend_is_input_error(tmp_path):
    with pytest.raises(InputError):
        HybridISP(str(tmp_path / "nope.pt"))
    raw_ckpt = tmp_path / "rawnet.pt"
    save_checkpoint(raw_ckpt, build_model("rawnet", 8), "rawnet", 8)
    with pytest.raises(InputError):
        HybridISP(str(raw_ckpt))


# ---- tiling (F-HY-2) -------------------------------------------------------------------------

def _tiled(tile, overlap):
    return AdaptiveController(clean_sigma=0.0, tile_above_mp=0.0, tile=tile, overlap=overlap)


def test_tiled_output_matches_untiled_closely(ckpt):
    raw = _raw(h=192, w=256)
    whole = _hybrid(ckpt).run(raw)
    tiled = _hybrid(ckpt, controller=_tiled(128, 64)).run(raw)
    assert tiled.info["tiles"] > 1 and whole.info["tiles"] == 1
    diff = np.abs(tiled.rgb - whole.rgb)
    assert diff.mean() < 5e-4 and diff.max() < 5e-3


def test_tiling_is_seamless_with_a_conv_only_network(tmp_path):
    """Without the global colour head the only tile effect is border context, which the overlap must hide."""
    ck = _make_ckpt(tmp_path / "conv.pt", colour_head=False)
    raw = _raw(h=192, w=256)
    whole = _hybrid(ck).run(raw).rgb
    tiled = _hybrid(ck, controller=_tiled(96, 64)).run(raw).rgb
    assert np.abs(tiled - whole).max() < 2e-3


def test_tiling_handles_sizes_that_do_not_divide_evenly(ckpt):
    res = _hybrid(ckpt, controller=_tiled(96, 32)).run(_raw(h=130, w=202))
    assert res.rgb.shape == (130, 202, 3) and np.isfinite(res.rgb).all()


# ---- ablation toggles (used by scripts/ablation.py) --------------------------------------------

@pytest.mark.parametrize("toggles", [
    {"sharpen": False}, {"ccm": False}, {"awb": "gray_world"}, {"noise_map": False},
    {"raw_denoise": True}, {"colour_head": False},
])
def test_each_ablation_changes_the_output(ckpt, toggles):
    raw = RawImage(bayer=_mosaic(a=0.02), ccm=np.array([[1.6, -0.4, -0.2], [-0.3, 1.5, -0.2], [-0.1, -0.5, 1.6]],
                                                         np.float32))
    full = _hybrid(ckpt).run(raw).rgb
    ablated = _hybrid(ckpt, ablation=toggles).run(raw).rgb
    assert np.abs(full - ablated).max() > 1e-4


def test_empty_or_none_ablation_equals_default(ckpt):
    raw = _raw()
    base = _hybrid(ckpt).run(raw).rgb
    assert np.array_equal(base, _hybrid(ckpt, ablation={}).run(raw).rgb)
    assert np.array_equal(base, _hybrid(ckpt, ablation=None).run(raw).rgb)


def test_unknown_ablation_key_is_rejected(ckpt):
    with pytest.raises(InputError, match="nonsense"):
        _hybrid(ckpt, ablation={"nonsense": True})


def test_colour_head_ablation_needs_a_pt_backend(onnx_path):
    with pytest.raises(InputError, match="colour_head"):
        _hybrid(onnx_path, ablation={"colour_head": False})
