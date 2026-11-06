"""Download the train, val and test images (see docs/DATA.md) into <root>/<dataset>/.

    python scripts/download_data.py [--root DIR] [--only DATASET ...] [--write-manifest]

Sources are pinned to fixed commits and every file is checked against the SHA-256 in
configs/data_manifest.json, then decoded as an image. Exits 1 if any dataset ends up short:
nothing is ever replaced by a different dataset. Downloads are only ever decoded as images.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp.data.splits import MANIFEST, SPLITS, data_root  # noqa: E402

SELFEXSR = "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/8f6dd8c1d20cb7e8792a7177b4f6fd677633f598/data"
SWINIR = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/6545850fbf8df298df73d81f3e8cba638787c8bd/testsets"
KODAK = "https://r0k.us/graphics/kodak/kodak"

# dataset -> function from 1-based index to (saved file name, URL)
SOURCES: dict[str, Callable[[int], tuple[str, str]]] = {
    "urban100": lambda i: (f"img_{i:03d}.png", f"{SELFEXSR}/Urban100/image_SRF_2/img_{i:03d}_SRF_2_HR.png"),
    "bsd100": lambda i: (f"img_{i:03d}.png", f"{SELFEXSR}/BSD100/image_SRF_2/img_{i:03d}_SRF_2_HR.png"),
    "set14": lambda i: (f"img_{i:03d}.png", f"{SELFEXSR}/Set14/image_SRF_2/img_{i:03d}_SRF_2_HR.png"),
    "sunhays80": lambda i: (f"img_{i:03d}.png", f"{SELFEXSR}/Sun-Hays80/image_SRF_8/img_{i:03d}_SRF_8_HR.png"),
    "mcmaster": lambda i: (f"{i}.tif", f"{SWINIR}/McMaster/{i}.tif"),
    "kodak24": lambda i: (f"kodim{i:02d}.png", f"{KODAK}/kodim{i:02d}.png"),
}
MAX_BYTES = 64 * 2**20  # no image in these sets comes near this
# urllib tries a host's addresses one by one, so a dead address costs a whole timeout on every request.
SOCKET_TIMEOUT = 10
WORKERS = 8


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Item:
    split: str
    dataset: str
    name: str
    url: str

    @property
    def relpath(self) -> str:
        return f"{self.dataset}/{self.name}"


def plan() -> list[Item]:
    items = []
    for split, datasets in SPLITS.items():
        for dataset, count in datasets.items():
            for i in range(1, count + 1):
                name, url = SOURCES[dataset](i)
                items.append(Item(split, dataset, name, url))
    return items


def http_get(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "hybrid-isp-data/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=SOCKET_TIMEOUT) as response:
                body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                raise DownloadError(f"larger than {MAX_BYTES} bytes: {url}")
            return body
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise DownloadError(f"HTTP {e.code}: {url}") from e
            error: Exception = e
        except OSError as e:
            error = e
        time.sleep(2**attempt)
    raise DownloadError(f"gave up after {attempts} attempts ({error}): {url}")


def check_image(body: bytes) -> None:
    try:
        with Image.open(io.BytesIO(body), formats=["PNG", "TIFF"]) as im:
            im.load()
    except Exception as e:  # PIL raises many types on bad data; all of them mean "not an image"
        raise DownloadError(f"not a decodable PNG or TIFF ({e})") from e


def install(item: Item, root: Path, expected: dict[str, str], fetch: Callable[[str], bytes] | None = None) -> str:
    """Make sure root/item is present and valid. Returns its SHA-256."""
    target = root / item.relpath
    want = expected.get(item.relpath)
    if target.exists():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if want in (None, digest):
            return digest
    body = (fetch or http_get)(item.url)
    digest = hashlib.sha256(body).hexdigest()
    if want is not None and digest != want:
        raise DownloadError(f"checksum mismatch for {item.relpath}: got {digest}")
    check_image(body)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(body)
    os.replace(partial, target)
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None, help="default: ISP_DATA_ROOT, else the shared data folder")
    parser.add_argument("--only", nargs="*", choices=sorted(SOURCES), help="limit to these datasets")
    parser.add_argument("--write-manifest", action="store_true", help="record checksums of a full, verified download")
    args = parser.parse_args(argv)
    if args.only and args.write_manifest:
        parser.error("--write-manifest needs the full dataset, so it cannot be combined with --only")
    root = args.root or data_root()
    expected = {} if args.write_manifest or not MANIFEST.exists() else json.loads(MANIFEST.read_text())["sha256"]

    digests: dict[str, str] = {}
    failures: dict[str, list[str]] = {}
    items = [i for i in plan() if not args.only or i.dataset in args.only]

    def attempt(item: Item) -> str | DownloadError:
        try:
            return install(item, root, expected)
        except DownloadError as e:
            return e

    with ThreadPoolExecutor(WORKERS) as pool:
        for item, result in zip(items, pool.map(attempt, items)):
            if isinstance(result, DownloadError):
                failures.setdefault(item.dataset, []).append(str(result))
            else:
                digests[item.relpath] = result

    print(f"data root: {root}")
    for split, datasets in SPLITS.items():
        for dataset, count in datasets.items():
            if args.only and dataset not in args.only:
                continue
            have = sum(k.startswith(f"{dataset}/") for k in digests)
            print(f"{split:5} {dataset:10} {have}/{count}" + ("" if have == count else "  INCOMPLETE"))
            for message in failures.get(dataset, [])[:3]:
                print(f"      {message}")
    if failures:
        return 1
    if args.write_manifest:
        MANIFEST.write_text(json.dumps({"sha256": dict(sorted(digests.items()))}, indent=0) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
