"""Bayer RAW pre-processing: levels, CFA normalisation to RGGB, pack/unpack (F-CV-1)."""
from __future__ import annotations

import numpy as np

from isp.types import InputError

# Crop (rows, cols) that turns each CFA pattern into RGGB.
_CROP = {"RGGB": (0, 0), "BGGR": (1, 1), "GRBG": (0, 1), "GBRG": (1, 0)}


def normalize_cfa(mosaic: np.ndarray, pattern: str, black: float, white: float) -> np.ndarray:
    """Black/white-level normalise and re-phase any 2x2 CFA to RGGB.

    Why: every later stage assumes one layout. Dropping one row/column shifts the
    pattern to RGGB without interpolating, so no information is invented.
    """
    if mosaic.ndim != 2:
        raise InputError("Expected a single-channel Bayer mosaic.")
    if pattern.upper() not in _CROP:
        raise InputError(f"Unsupported colour filter pattern '{pattern}'; use RGGB, BGGR, GRBG or GBRG.")
    if white <= black:
        raise InputError("White level must be larger than black level.")
    dr, dc = _CROP[pattern.upper()]
    x = (mosaic[dr:, dc:].astype(np.float32) - black) / (white - black)
    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    return np.clip(x[:h, :w], 0.0, 1.0).astype(np.float32)


def pack_rggb(bayer: np.ndarray) -> np.ndarray:
    """(H, W) RGGB mosaic -> (H/2, W/2, 4) planes ordered R, G1, G2, B.

    Why: each plane is then a smooth same-colour image, ideal for noise
    estimation and filtering that must not mix colour channels.
    """
    return np.stack([bayer[0::2, 0::2], bayer[0::2, 1::2], bayer[1::2, 0::2], bayer[1::2, 1::2]], -1)


def unpack_rggb(planes: np.ndarray) -> np.ndarray:
    """Inverse of pack_rggb."""
    h, w, _ = planes.shape
    out = np.empty((h * 2, w * 2), planes.dtype)
    out[0::2, 0::2], out[0::2, 1::2] = planes[..., 0], planes[..., 1]
    out[1::2, 0::2], out[1::2, 1::2] = planes[..., 2], planes[..., 3]
    return out


def mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    """Sample an RGB image through an RGGB colour filter array (test/synthesis helper)."""
    out = np.empty(rgb.shape[:2], rgb.dtype)
    out[0::2, 0::2] = rgb[0::2, 0::2, 0]
    out[0::2, 1::2] = rgb[0::2, 1::2, 1]
    out[1::2, 0::2] = rgb[1::2, 0::2, 1]
    out[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return out
