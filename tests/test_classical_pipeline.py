import numpy as np
import pytest

from isp.classical.bayer import mosaic_rggb
from isp.classical.pipeline import ClassicalConfig, ClassicalISP, front_end
from isp.types import InputError, PipelineResult, RawImage

STAGES = {"raw", "denoised_raw", "demosaic", "white_balance", "ccm", "exposure", "tone"}


def scene(h=64, w=64):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([0.1 + 0.3 * x / w, 0.1 + 0.3 * y / h, 0.08 + 0 * x], -1)
    return rgb.astype(np.float32)


def raw_of(rgb, **kw):
    return RawImage(bayer=mosaic_rggb(rgb), **kw)


def test_run_fills_result():
    isp = ClassicalISP()
    res = isp.run(raw_of(scene()))
    assert isinstance(res, PipelineResult) and isp.name == "classical"
    assert res.rgb.shape == (64, 64, 3) and res.rgb.dtype == np.float32
    assert 0 <= res.rgb.min() and res.rgb.max() <= 1
    assert STAGES <= set(res.stages) and "sharpen" in res.stages
    for img in res.stages.values():
        assert img.ndim == 3 and np.isfinite(img).all()
    assert set(res.timings_ms) >= (STAGES - {"raw"}) | {"noise_estimation", "sharpen", "total"}
    assert all(v >= 0 for v in res.timings_ms.values())
    assert {"noise_a", "noise_b", "exposure"} <= set(res.info)


def test_front_end_contract():
    rgb, stages, meta = front_end(raw_of(scene()), denoise=False)
    assert rgb.shape == (64, 64, 3) and set(meta) == {"noise_a", "noise_b", "exposure"}
    assert "denoised_raw" not in stages


def test_uses_raw_ccm_when_given():
    ccm = np.eye(3)
    a = ClassicalISP().run(raw_of(scene(), ccm=ccm)).rgb
    b = ClassicalISP().run(raw_of(scene())).rgb
    assert not np.allclose(a, b)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_black_and_saturated_frames_have_no_nan(value):
    res = ClassicalISP().run(RawImage(bayer=np.full((32, 32), value, np.float32)))
    assert np.isfinite(res.rgb).all() and 0 <= res.rgb.min() and res.rgb.max() <= 1
    assert all(np.isfinite(v) for v in res.info.values() if isinstance(v, float))


def test_noisy_dark_image_is_cleaner_with_denoise():
    rng = np.random.default_rng(0)
    clean = mosaic_rggb(scene(96, 96))
    n = np.clip(clean + rng.normal(size=clean.shape) * np.sqrt(0.005 * clean + 5e-5), 0, 1)
    n = n.astype(np.float32)
    ref = ClassicalISP().run(RawImage(bayer=clean)).rgb
    on = ClassicalISP(ClassicalConfig(denoise=True)).run(RawImage(bayer=n)).rgb
    off = ClassicalISP(ClassicalConfig(denoise=False)).run(RawImage(bayer=n)).rgb
    assert np.mean((on - ref) ** 2) < np.mean((off - ref) ** 2)


def test_rejects_bad_input():
    with pytest.raises(InputError):
        ClassicalISP().run(RawImage(bayer=np.zeros((8, 8, 3), np.float32)))


def test_default_tone_inverts_the_reference_rendering():
    """The baseline must render like the targets, or it loses PSNR to a tone choice and not to its image quality."""
    from isp.classical.pipeline import ClassicalConfig
    from isp.classical.tone import gamma22, s_curve

    x = np.linspace(0.02, 0.98, 49, dtype=np.float32)
    lin = (0.5 - np.sin(np.arcsin(1.0 - 2.0 * x) / 3.0)) ** 2.2  # the unprocessing in isp.data.synthetic
    cfg = ClassicalConfig()
    assert cfg.gamma == "gamma22"
    assert np.abs(s_curve(gamma22(lin), cfg.s_curve) - x).max() < 1e-3


def test_as_shot_white_balance_is_used_when_the_file_has_it():
    from isp.classical.pipeline import ClassicalConfig, _estimate_gains
    from isp.types import RawImage

    rgb = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    shot = RawImage(bayer=np.zeros((32, 32), np.float32), wb_gains=np.array([2.0, 1.0, 1.5], np.float32))
    assert np.allclose(_estimate_gains(rgb, shot, ClassicalConfig().awb), [2.0, 1.0, 1.5])
    blind = RawImage(bayer=np.zeros((32, 32), np.float32))
    assert np.isfinite(_estimate_gains(rgb, blind, ClassicalConfig().awb)).all()
