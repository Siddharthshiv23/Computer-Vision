"""Overlapped tiling with feathered blending, so 12 MP frames fit in memory without seams (F-HY-2)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def tile_starts(n: int, tile: int, overlap: int) -> list[int]:
    """Start offsets covering [0, n) with tiles of `tile` pixels; the last tile is flush with the edge."""
    if n <= tile:
        return [0]
    return list(range(0, n - tile, tile - overlap)) + [n - tile]


def _ramp(length: int, start: int, n: int, tile: int, overlap: int) -> np.ndarray:
    """1-D blend weights for one tile: linear fade-in/out across the overlap, only on interior edges."""
    w = np.ones(length, np.float32)
    if overlap == 0:
        return w
    up = np.minimum(1.0, (np.arange(length) + 1) / overlap)
    if start > 0:
        w = np.minimum(w, up)
    if start + length < n:
        w = np.minimum(w, up[::-1])
    return w


def run_tiled(x: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], tile: int, overlap: int
              ) -> tuple[np.ndarray, int]:
    """Apply `fn` ((1, C, h, w) -> (3, h, w)) over overlapping tiles of x (1, C, H, W).

    Returns the blended (3, H, W) result and the number of tiles. Why feathering: each tile sees
    zero padding at its border, so its outer `overlap` pixels are the least reliable; fading them out
    in favour of the neighbour's interior hides the seam. Weights are normalised per pixel, so
    every pixel is a convex combination of the tiles that cover it.
    """
    if not 0 <= overlap < tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")
    h, w = x.shape[2:]
    ys, xs = tile_starts(h, tile, overlap), tile_starts(w, tile, overlap)
    if len(ys) * len(xs) == 1:
        return fn(x), 1
    acc = weight = None
    for y in ys:
        for x0 in xs:
            out = fn(x[:, :, y:y + tile, x0:x0 + tile])
            th, tw = out.shape[1:]
            win = _ramp(th, y, h, tile, overlap)[:, None] * _ramp(tw, x0, w, tile, overlap)[None, :]
            if acc is None:
                acc, weight = np.zeros((out.shape[0], h, w), np.float32), np.zeros((h, w), np.float32)
            acc[:, y:y + th, x0:x0 + tw] += out * win
            weight[y:y + th, x0:x0 + tw] += win
    return acc / weight, len(ys) * len(xs)
