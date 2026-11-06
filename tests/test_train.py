import csv

import numpy as np
import pytest
import torch

from isp.types import RawImage
from scripts import train


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, split, crop, seed, n=4):
        self.crop, self.n, self.seed = crop, n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        target = rng.random((self.crop, self.crop, 3), dtype=np.float32)
        bayer = rng.random((self.crop, self.crop), dtype=np.float32)
        return RawImage(bayer=bayer), target


def fake_front_end(raw, denoise):
    h, w = raw.bayer.shape
    rgb = np.repeat(raw.bayer[..., None], 3, axis=2)
    return rgb, {}, {"noise_a": 0.01, "noise_b": 1e-4, "exposure": 1.0}


def make_factory(seen):
    def factory(split, crop, seed):
        seen.append(split)
        return FakeDataset(split, crop, seed)
    return factory


def run(tmp_path, *extra, model="rawnet", name="t"):
    seen = []
    argv = ["--model", model, "--name", name, "--epochs", "2", "--batch-size", "2", "--crop", "32",
            "--width", "8", "--runs-dir", str(tmp_path), "--seed", "1", *extra]
    train.main(argv, dataset_factory=make_factory(seen), front_end=fake_front_end)
    return seen


def test_rawnet_trains_and_writes_artifacts(tmp_path):
    seen = run(tmp_path)
    assert (tmp_path / "t" / "best.pt").exists()
    rows = list(csv.DictReader(open(tmp_path / "t" / "log.csv")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert {"train_loss", "val_loss", "val_psnr"} <= set(rows[0])
    assert set(seen) == {"train", "val"}  # never touches the test split


def test_liteisp_trains(tmp_path):
    run(tmp_path, model="liteisp")
    assert (tmp_path / "t" / "best.pt").exists()


def test_best_checkpoint_is_lowest_val_loss(tmp_path):
    run(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "t" / "log.csv")))
    best = torch.load(tmp_path / "t" / "best.pt", weights_only=False)
    assert best["val_loss"] == pytest.approx(min(float(r["val_loss"]) for r in rows), rel=1e-5)


def test_seeded_runs_are_reproducible(tmp_path):
    run(tmp_path, name="a")
    run(tmp_path, name="b")
    a = open(tmp_path / "a" / "log.csv").read()
    b = open(tmp_path / "b" / "log.csv").read()
    assert a == b


def test_resume_continues_from_last_epoch(tmp_path):
    run(tmp_path)  # epochs 1-2
    argv = ["--model", "rawnet", "--name", "t", "--epochs", "3", "--batch-size", "2", "--crop", "32",
            "--width", "8", "--runs-dir", str(tmp_path), "--seed", "1", "--resume"]
    train.main(argv, dataset_factory=make_factory([]), front_end=fake_front_end)
    rows = list(csv.DictReader(open(tmp_path / "t" / "log.csv")))
    assert [r["epoch"] for r in rows] == ["1", "2", "3"]


def test_distillation_and_finetune(tmp_path):
    run(tmp_path, model="liteisp", name="teacher")
    teacher = str(tmp_path / "teacher" / "best.pt")
    run(tmp_path, "--teacher", teacher, model="liteisp", name="student")
    run(tmp_path, "--finetune", teacher, model="liteisp", name="ft")
    assert (tmp_path / "student" / "best.pt").exists() and (tmp_path / "ft" / "best.pt").exists()


def test_teacher_rejected_for_rawnet(tmp_path):
    run(tmp_path, model="liteisp", name="teacher")
    with pytest.raises(SystemExit):
        run(tmp_path, "--teacher", str(tmp_path / "teacher" / "best.pt"), model="rawnet")


def test_smoke_on_default_device(tmp_path):
    """Uses CUDA when available (device=auto) with AMP."""
    run(tmp_path, "--device", "auto", "--max-steps", "2", name="smoke")
    assert (tmp_path / "smoke" / "best.pt").exists()


def test_default_front_end_is_the_real_classical_function():
    """The fakes above cannot catch a wrong name, so resolve the real one when W1's code is present."""
    pytest.importorskip("isp.classical.pipeline")
    assert callable(train.default_front_end())
