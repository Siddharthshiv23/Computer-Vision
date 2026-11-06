"""Split isolation and dataset behaviour. Fixture trees use tiny generated images, never the real data."""
import json

import numpy as np
import pytest
from PIL import Image

from isp.data import splits
from isp.data.datasets import SyntheticRawDataset, split_files
from isp.data.splits import MANIFEST, SPLITS, data_root
from isp.types import RawImage


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A complete fake data tree: every file gets unique pixels."""
    rng = np.random.default_rng(0)
    counts = {"train": {"a": 3, "b": 2}, "val": {"c": 2}, "test": {"d": 2}}
    monkeypatch.setattr(splits, "SPLITS", counts)
    for datasets in counts.values():
        for dataset, n in datasets.items():
            folder = tmp_path / dataset
            folder.mkdir(parents=True)
            for i in range(n):
                pixels = rng.integers(0, 256, (70, 90, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{i}.png")
    return tmp_path


def _hashes():
    table = json.loads(MANIFEST.read_text())["sha256"]
    split_of = {dataset: split for split, datasets in SPLITS.items() for dataset in datasets}
    out = {s: {} for s in SPLITS}
    for relpath, digest in table.items():
        out[split_of[relpath.split("/")[0]]][relpath] = digest
    return out


def test_manifest_has_the_prd_count_per_split():
    assert {s: len(v) for s, v in _hashes().items()} == {"train": 294, "val": 18, "test": 24}


def test_no_file_content_is_shared_between_splits():
    hashes = _hashes()
    train, val, test = (set(hashes[s].values()) for s in ("train", "val", "test"))
    assert not train & val
    assert not train & test
    assert not val & test


def test_a_dataset_belongs_to_exactly_one_split():
    seen = [d for datasets in SPLITS.values() for d in datasets]
    assert len(seen) == len(set(seen))
    assert set(SPLITS["val"]) == {"mcmaster"} and set(SPLITS["test"]) == {"kodak24"}


@pytest.mark.skipif(not (data_root() / "kodak24").exists(), reason="datasets not downloaded")
def test_evaluation_harness_finds_kodak24_in_our_layout():
    from isp.eval import harness

    assert len(harness.load_kodak(data_root() / "kodak24")) == 24


@pytest.mark.skipif(not (data_root() / "kodak24").exists(), reason="datasets not downloaded")
def test_real_files_on_disk_match_the_manifest_and_do_not_overlap():
    import hashlib

    table = json.loads(MANIFEST.read_text())["sha256"]
    paths = {s: split_files(s) for s in SPLITS}
    all_paths = [p for files in paths.values() for p in files]
    assert len(set(all_paths)) == len(all_paths) == 336
    for path in all_paths:
        rel = path.relative_to(data_root()).as_posix()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == table[rel]


def test_split_files_only_returns_that_split(root):
    assert [p.parent.name for p in split_files("val", root)] == ["c", "c"]
    assert {p.parent.name for p in split_files("train", root)} == {"a", "b"}


def test_stray_folder_in_a_split_is_not_picked_up(root):
    stray = root / "extra"
    stray.mkdir()
    Image.new("RGB", (70, 90)).save(stray / "x.png")
    assert all(p.parent.name != "extra" for p in split_files("train", root))


def test_incomplete_split_refuses_to_load(root):
    (root / "a" / "0.png").unlink()
    with pytest.raises(FileNotFoundError, match="download_data"):
        split_files("train", root)


def test_unknown_split_is_rejected(root):
    with pytest.raises(ValueError):
        SyntheticRawDataset("everything", 32, 0, root=root)


def test_odd_crop_is_rejected(root):
    with pytest.raises(ValueError):
        SyntheticRawDataset("train", 33, 0, root=root)


def test_items_are_raw_and_target_of_crop_size(root):
    ds = SyntheticRawDataset("train", 32, 0, root=root)
    assert len(ds) == 5
    raw, target = ds[0]
    assert isinstance(raw, RawImage)
    assert raw.bayer.shape == (32, 32)
    assert target.shape == (32, 32, 3) and target.dtype == np.float32


def test_crop_larger_than_image_still_yields_full_crop(root):
    raw, target = SyntheticRawDataset("train", 128, 0, root=root)[0]
    assert raw.bayer.shape == (128, 128) and target.shape == (128, 128, 3)


def test_train_draws_a_new_crop_and_condition_each_access(root):
    ds = SyntheticRawDataset("train", 32, 0, root=root)
    (raw1, t1), (raw2, t2) = ds[0], ds[0]
    assert not np.array_equal(t1, t2)
    assert raw1.meta["truth"]["ev"] != raw2.meta["truth"]["ev"]


def test_train_stream_is_reproducible_from_the_seed(root):
    a = SyntheticRawDataset("train", 32, 7, root=root)
    b = SyntheticRawDataset("train", 32, 7, root=root)
    for _ in range(3):
        np.testing.assert_array_equal(a[1][0].bayer, b[1][0].bayer)


def test_different_seed_gives_a_different_train_stream(root):
    a = SyntheticRawDataset("train", 32, 1, root=root)[0][0]
    b = SyntheticRawDataset("train", 32, 2, root=root)[0][0]
    assert not np.array_equal(a.bayer, b.bayer)


def test_val_is_identical_on_every_access(root):
    ds = SyntheticRawDataset("val", 32, 0, root=root)
    np.testing.assert_array_equal(ds[1][0].bayer, ds[1][0].bayer)
    np.testing.assert_array_equal(ds[1][1], ds[1][1])


def test_full_frame_when_no_crop(root):
    raw, target = SyntheticRawDataset("test", None, 0, root=root)[0]
    assert raw.bayer.shape == (70, 90) and target.shape == (70, 90, 3)


def test_test_split_ignores_the_caller_seed(root):
    a = SyntheticRawDataset("test", None, 1, root=root)[0][0]
    b = SyntheticRawDataset("test", None, 2, root=root)[0][0]
    np.testing.assert_array_equal(a.bayer, b.bayer)


def test_test_split_uses_the_named_condition(root):
    raw, _ = SyntheticRawDataset("test", None, 0, root=root, condition="ev_-3")[0]
    assert raw.meta["truth"]["ev"] == -3.0


def test_test_images_get_their_own_noise_seed(root):
    ds = SyntheticRawDataset("test", None, 0, root=root)
    assert ds[0][0].meta["truth"]["seed"] != ds[1][0].meta["truth"]["seed"]


def test_grayscale_source_file_is_expanded(root):
    path = root / "c" / "0.png"
    Image.new("L", (70, 90), 128).save(path)
    _, target = SyntheticRawDataset("val", 32, 0, root=root)[0]
    assert target.shape == (32, 32, 3)


def test_tiff_source_files_load(root):
    # McMaster ships as TIFF.
    png = root / "c" / "0.png"
    Image.open(png).save(png.with_suffix(".tif"))
    png.unlink()
    _, target = SyntheticRawDataset("val", 32, 0, root=root)[1]
    assert target.shape == (32, 32, 3)


def _first_target(batch):
    return batch[0][1]


def test_loader_workers_see_new_crops_each_epoch(root):
    # DataLoader workers restart from a pickled copy every epoch; their draws must not repeat.
    torch = pytest.importorskip("torch")
    ds = SyntheticRawDataset("train", 32, 0, root=root)
    epochs = []
    for epoch in range(2):
        gen = torch.Generator().manual_seed(epoch)
        loader = torch.utils.data.DataLoader(
            ds, 1, shuffle=False, collate_fn=_first_target, num_workers=1, generator=gen
        )
        epochs.append(next(iter(loader)))
    assert not np.array_equal(epochs[0], epochs[1])
