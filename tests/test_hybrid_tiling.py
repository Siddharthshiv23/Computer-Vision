import numpy as np
import pytest

from isp.hybrid.tiling import run_tiled, tile_starts


@pytest.mark.parametrize("n,tile,overlap", [(100, 40, 10), (97, 40, 10), (40, 40, 10), (30, 40, 10), (1000, 256, 32)])
def test_tile_starts_cover_the_axis(n, tile, overlap):
    starts = tile_starts(n, tile, overlap)
    covered = np.zeros(n, bool)
    for s in starts:
        covered[s:s + tile] = True
    assert covered.all() and starts[-1] + min(tile, n) == n


def test_identity_function_is_reconstructed_exactly():
    x = np.random.default_rng(0).random((1, 4, 70, 91), dtype=np.float32)
    out, n = run_tiled(x, lambda t: t[0, :3], tile=32, overlap=8)
    assert n > 1 and np.allclose(out, x[0, :3], atol=1e-6)


def test_per_tile_offsets_are_blended_not_stepped():
    """Each tile adds its own constant (like a per-tile colour matrix). Feathering must turn the
    jump into a ramp: no neighbouring pixels may differ by more than offset_gap / overlap * 2."""
    x = np.zeros((1, 4, 64, 160), np.float32)
    calls = iter(range(100))
    out, n = run_tiled(x, lambda t: np.full((3, *t.shape[2:]), 0.01 * next(calls), np.float32), tile=64, overlap=32)
    assert n > 2
    step = np.abs(np.diff(out[0], axis=1)).max()
    assert step <= 2 * 0.01 / 32 + 1e-6


def test_rejects_overlap_not_smaller_than_tile():
    with pytest.raises(ValueError):
        run_tiled(np.zeros((1, 4, 64, 64), np.float32), lambda t: t[0, :3], tile=32, overlap=32)
