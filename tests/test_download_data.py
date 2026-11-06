"""Download planning and failure behaviour. No network: the fetch step is replaced everywhere."""
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_data as dd  # noqa: E402

ITEM = dd.Item("val", "mcmaster", "1.png", "https://example.invalid/1.png")


def _png_bytes(tmp_path):
    p = tmp_path / "src.png"
    Image.new("RGB", (40, 40), (10, 20, 30)).save(p)
    return p.read_bytes()


def test_plan_matches_prd_counts_per_split():
    items = dd.plan()
    per_split = {s: sum(i.split == s for i in items) for s in ("train", "val", "test")}
    assert per_split == {"train": 294, "val": 18, "test": 24}


def test_kodak24_lands_where_the_evaluation_scripts_look():
    assert "kodak24/kodim01.png" in {i.relpath for i in dd.plan()}


def test_plan_has_no_duplicate_targets_or_urls():
    items = dd.plan()
    assert len({i.relpath for i in items}) == len(items)
    assert len({i.url for i in items}) == len(items)


def test_all_plan_urls_are_https():
    assert all(i.url.startswith("https://") for i in dd.plan())


def test_tiff_payload_is_accepted_in_a_fresh_process(tmp_path):
    # McMaster ships as TIFF. Fresh process: saving a TIFF here would load the plugin and hide the bug.
    path = tmp_path / "src.tif"
    Image.new("RGB", (40, 40), (10, 20, 30)).save(path)
    code = "import sys; import download_data as d; d.check_image(open(sys.argv[1], 'rb').read())"
    scripts = Path(dd.__file__).parent
    subprocess.run([sys.executable, "-c", code, str(path)], cwd=scripts, check=True, capture_output=True)


def test_non_image_payload_is_rejected_and_not_kept(tmp_path):
    with pytest.raises(dd.DownloadError):
        dd.install(ITEM, tmp_path, {}, fetch=lambda url: b"<html>not found</html>")
    assert not (tmp_path / ITEM.relpath).exists()


def test_checksum_mismatch_is_rejected(tmp_path):
    good = _png_bytes(tmp_path)
    with pytest.raises(dd.DownloadError, match="checksum"):
        dd.install(ITEM, tmp_path, {ITEM.relpath: "0" * 64}, fetch=lambda url: good)
    assert not (tmp_path / ITEM.relpath).exists()


def test_existing_valid_file_is_not_refetched(tmp_path):
    good = _png_bytes(tmp_path)
    digest = hashlib.sha256(good).hexdigest()
    dd.install(ITEM, tmp_path, {ITEM.relpath: digest}, fetch=lambda url: good)

    def boom(url):
        raise AssertionError("should not refetch")

    assert dd.install(ITEM, tmp_path, {ITEM.relpath: digest}, fetch=boom) == digest


def test_existing_file_with_wrong_checksum_is_replaced(tmp_path):
    good = _png_bytes(tmp_path)
    digest = hashlib.sha256(good).hexdigest()
    target = tmp_path / ITEM.relpath
    target.parent.mkdir(parents=True)
    target.write_bytes(b"stale")
    dd.install(ITEM, tmp_path, {ITEM.relpath: digest}, fetch=lambda url: good)
    assert target.read_bytes() == good


def test_unreachable_source_exits_nonzero_and_names_the_dataset(tmp_path, monkeypatch, capsys):
    def down(url):
        raise dd.DownloadError("HTTP 404")

    monkeypatch.setattr(dd, "http_get", down)
    assert dd.main(["--root", str(tmp_path), "--only", "kodak24"]) == 1
    out = capsys.readouterr().out
    assert "kodak24" in out and "0/24" in out


def test_env_var_sets_root(tmp_path, monkeypatch):
    from isp.data.splits import data_root

    monkeypatch.setenv("ISP_DATA_ROOT", str(tmp_path))
    assert data_root() == tmp_path
