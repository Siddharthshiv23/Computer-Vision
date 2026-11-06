"""Where the datasets live and which dataset belongs to which split. Standard library only."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DATA_ROOT = Path(r"C:\Users\ddart\cv-hackathon\data")
MANIFEST = REPO_ROOT / "configs" / "data_manifest.json"

# split -> {dataset folder under the data root: image count}. A dataset appears in exactly one split.
SPLITS: dict[str, dict[str, int]] = {
    "train": {"urban100": 100, "bsd100": 100, "set14": 14, "sunhays80": 80},
    "val": {"mcmaster": 18},
    "test": {"kodak24": 24},
}


def data_root() -> Path:
    """ISP_DATA_ROOT, else the shared hackathon folder, else data/ in this checkout (fresh clones)."""
    env = os.environ.get("ISP_DATA_ROOT")
    if env:
        return Path(env)
    return SHARED_DATA_ROOT if SHARED_DATA_ROOT.parent.exists() else REPO_ROOT / "data"
