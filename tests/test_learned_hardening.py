"""Review findings for LearnedISP / load_checkpoint: bad input must raise InputError, never a raw error."""
import numpy as np
import pytest
import torch

from isp.learned.checkpoint import build_model, load_checkpoint, save_checkpoint
from isp.learned.pipeline import LearnedISP
from isp.types import InputError, RawImage


class Evil:  # module level so torch.save can pickle it
    pass


def _rawnet_ckpt(tmp_path):
    ckpt = tmp_path / "rawnet.pt"
    save_checkpoint(ckpt, build_model("rawnet", 8), "rawnet", 8)
    return str(ckpt)


@pytest.mark.parametrize("shape", [(33, 32), (32, 33), (31, 31)])
def test_rejects_odd_sized_mosaic(tmp_path, shape):
    with pytest.raises(InputError, match="even"):
        LearnedISP(_rawnet_ckpt(tmp_path)).run(RawImage(bayer=np.zeros(shape, np.float32)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_mosaic(tmp_path, bad):
    bayer = np.full((16, 16), 0.2, np.float32)
    bayer[3, 4] = bad
    with pytest.raises(InputError, match="finite"):
        LearnedISP(_rawnet_ckpt(tmp_path)).run(RawImage(bayer=bayer))


def test_clips_out_of_range_input(tmp_path):
    pipe = LearnedISP(_rawnet_ckpt(tmp_path))
    base = np.random.default_rng(0).random((16, 16), dtype=np.float32) * 2 - 0.5
    got = pipe.run(RawImage(bayer=base)).rgb
    want = pipe.run(RawImage(bayer=np.clip(base, 0, 1))).rgb
    assert np.array_equal(got, want)


def test_load_checkpoint_refuses_pickled_objects(tmp_path):
    path = tmp_path / "evil.pt"
    torch.save({"model": "rawnet", "width": 8, "state": {}, "obj": Evil()}, path)
    with pytest.raises(InputError):
        load_checkpoint(path)


def test_load_checkpoint_garbage_file_is_input_error(tmp_path):
    path = tmp_path / "junk.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(InputError):
        load_checkpoint(path)
