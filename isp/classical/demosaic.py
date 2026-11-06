"""Demosaicing of an RGGB mosaic (F-CV-1): bilinear, Malvar-He-Cutler, OpenCV-EA."""
from __future__ import annotations

import cv2
import numpy as np

_B = cv2.BORDER_REFLECT_101  # keeps the Bayer phase: plain REFLECT repeats the edge and reads the wrong colour


def _masks(h: int, w: int) -> np.ndarray:
    """Boolean sample sites (R, G, B) of an RGGB mosaic, shape (3, H, W)."""
    m = np.zeros((3, h, w), bool)
    m[0, 0::2, 0::2] = True
    m[1, 0::2, 1::2] = m[1, 1::2, 0::2] = True
    m[2, 1::2, 1::2] = True
    return m


def demosaic_bilinear(bayer: np.ndarray) -> np.ndarray:
    """Bilinear interpolation: the simplest baseline, blurs and zippers on edges.

    Why: it is the reference every other method is compared against.
    """
    m = _masks(*bayer.shape)
    k_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    out = []
    for c, k in enumerate((k_rb, k_g, k_rb)):
        plane = np.where(m[c], bayer, 0).astype(np.float32)
        out.append(cv2.filter2D(plane, -1, k, borderType=_B))
    return np.clip(np.stack(out, -1), 0, 1).astype(np.float32)


_G_AT_RB = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
                     [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], np.float32) / 8
_RB_AT_G_ROW = np.array([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1],
                         [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]], np.float32) / 8
_RB_AT_G_COL = _RB_AT_G_ROW.T.copy()
_RB_AT_BR = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5],
                      [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], np.float32) / 8


def demosaic_malvar(bayer: np.ndarray) -> np.ndarray:
    """Malvar-He-Cutler (2004): bilinear plus a gradient correction from the other channel.

    Why: a fixed 5x5 linear filter that is almost as cheap as bilinear but removes
    most colour fringing; best quality-per-cost for a mobile front end.
    """
    h, w = bayer.shape
    m = _masks(h, w)
    f = lambda k: cv2.filter2D(bayer, -1, k, borderType=_B)  # noqa: E731
    g_rb, row, col, br = f(_G_AT_RB), f(_RB_AT_G_ROW), f(_RB_AT_G_COL), f(_RB_AT_BR)
    r, g, b = bayer.copy(), bayer.copy(), bayer.copy()
    even_r = np.zeros((h, w), bool)
    even_r[0::2, :] = True
    g_r_row = m[1] & even_r      # green with red in its row (blue in its column)
    g_b_row = m[1] & ~even_r     # green with blue in its row (red in its column)
    g[m[0] | m[2]] = g_rb[m[0] | m[2]]
    r[g_r_row], r[g_b_row], r[m[2]] = row[g_r_row], col[g_b_row], br[m[2]]
    b[g_r_row], b[g_b_row], b[m[0]] = col[g_r_row], row[g_b_row], br[m[0]]
    return np.clip(np.stack([r, g, b], -1), 0, 1).astype(np.float32)


def demosaic_opencv_ea(bayer: np.ndarray) -> np.ndarray:
    """OpenCV edge-aware demosaic: interpolates along edges instead of across them.

    Why: a well-tested library reference to sanity-check our own implementations.
    """
    u16 = np.round(np.clip(bayer, 0, 1) * 65535).astype(np.uint16)
    # OpenCV names Bayer codes by the second row/column, so its "BG" code means RGGB data.
    rgb = cv2.cvtColor(u16, cv2.COLOR_BAYER_BG2RGB_EA)
    return (rgb.astype(np.float32) / 65535).clip(0, 1)


_METHODS = {"bilinear": demosaic_bilinear, "malvar": demosaic_malvar, "opencv_ea": demosaic_opencv_ea}


def demosaic(bayer: np.ndarray, method: str = "malvar") -> np.ndarray:
    """Dispatch by name; raises ValueError for unknown methods."""
    if method not in _METHODS:
        raise ValueError(f"Unknown demosaic method '{method}'; choose from {sorted(_METHODS)}.")
    return _METHODS[method](bayer)
