"""Synthetic RAW datasets. A split only ever reads its own folders (see isp.data.splits)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from isp.data import splits
from isp.data.synthetic import load_test_conditions, sample_condition, synthesize
from isp.types import RawImage

IMAGE_SUFFIXES = {".png", ".tif"}


def split_files(split: str, root: Path | None = None) -> list[Path]:
    """All images of a split. Refuses partial data, because results on it would not be comparable."""
    if split not in splits.SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(splits.SPLITS)}")
    root = Path(root) if root else splits.data_root()
    files: list[Path] = []
    for dataset, count in splits.SPLITS[split].items():
        folder = root / dataset
        found = sorted(p for p in folder.glob("*") if p.suffix in IMAGE_SUFFIXES)
        if len(found) != count:
            raise FileNotFoundError(
                f"{folder} has {len(found)} images, expected {count}. Run: python scripts/download_data.py"
            )
        files += found
    return files


def _crop(image: np.ndarray, size: int, top: float, left: float) -> np.ndarray:
    """Crop at the given fractional position, reflecting the borders if the image is smaller than size."""
    h, w = image.shape[:2]
    if h < size or w < size:
        pad = ((0, max(size - h, 0)), (0, max(size - w, 0)), (0, 0))
        image = np.pad(image, pad, mode="reflect")
        h, w = image.shape[:2]
    y, x = int(top * (h - size + 1)), int(left * (w - size + 1))
    return image[y : y + size, x : x + size]


class SyntheticRawDataset:
    """Map-style dataset of (RawImage, target sRGB float32) pairs. Use a custom collate_fn with DataLoader.

    train: a new random crop and sensor condition on every access, reproducible from `seed`.
    val:   crop and condition fixed per image by `seed`, identical on every access.
    test:  fixed by configs/test_conditions.yaml; `seed` is ignored so nobody can reroll the test set.
           `condition` picks the named condition. crop=None gives full frames (any split).
    """

    def __init__(
        self,
        split: str,
        crop: int | None = None,
        seed: int = 0,
        root: Path | None = None,
        condition: str = "standard",
    ):
        if crop is not None and crop % 2:
            raise ValueError(f"crop must be even to keep the RGGB layout, got {crop}")
        self.split, self.crop, self.seed = split, crop, seed
        self.files = split_files(split, root)
        self.test_condition = load_test_conditions()[condition] if split == "test" else None
        self._draws = 0

    def __len__(self) -> int:
        return len(self.files)

    def _rng(self, index: int) -> np.random.Generator:
        if self.split != "train":
            return np.random.default_rng([self.seed, index])
        self._draws += 1
        # DataLoader workers restart from a fresh copy each epoch, so _draws alone would repeat there.
        torch = sys.modules.get("torch")
        info = torch.utils.data.get_worker_info() if torch else None
        return np.random.default_rng([self.seed, index, self._draws, info.seed if info else 0])

    def __getitem__(self, index: int) -> tuple[RawImage, np.ndarray]:
        with Image.open(self.files[index], formats=["PNG", "TIFF"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        rng = self._rng(index)
        if self.crop is not None:
            rgb = _crop(rgb, self.crop, rng.random(), rng.random())
        if self.test_condition is not None:
            cond = replace(self.test_condition, seed=self.test_condition.seed + index)
        else:
            cond = sample_condition(rng)
        return synthesize(rgb, cond)
