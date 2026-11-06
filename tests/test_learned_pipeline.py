import numpy as np
import pytest

from isp.learned.checkpoint import build_model, save_checkpoint
from isp.learned.pipeline import LearnedISP
from isp.types import InputError, RawImage


def test_learned_isp_runs_on_raw(tmp_path):
    ckpt = tmp_path / "rawnet.pt"
    save_checkpoint(ckpt, build_model("rawnet", 8), "rawnet", 8)
    pipe = LearnedISP(str(ckpt))
    res = pipe.run(RawImage(bayer=np.random.rand(32, 48).astype(np.float32)))
    assert pipe.name == "learned"
    assert res.rgb.shape == (32, 48, 3) and res.rgb.dtype == np.float32
    assert 0.0 <= res.rgb.min() and res.rgb.max() <= 1.0
    assert "network" in res.timings_ms


def test_learned_isp_rejects_wrong_model_kind(tmp_path):
    ckpt = tmp_path / "lite.pt"
    save_checkpoint(ckpt, build_model("liteisp", 8), "liteisp", 8)
    with pytest.raises(InputError):
        LearnedISP(str(ckpt))


def test_learned_isp_missing_checkpoint():
    with pytest.raises(InputError):
        LearnedISP("does/not/exist.pt")


def test_learned_isp_tolerates_black_frame(tmp_path):
    ckpt = tmp_path / "rawnet.pt"
    save_checkpoint(ckpt, build_model("rawnet", 8), "rawnet", 8)
    res = LearnedISP(str(ckpt)).run(RawImage(bayer=np.zeros((16, 16), np.float32)))
    assert np.isfinite(res.rgb).all()
