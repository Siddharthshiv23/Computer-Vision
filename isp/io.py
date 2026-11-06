"""load_input: any supported upload -> RawImage (float32, RGGB, even size, black level removed).

    DNG / camera RAW    rawpy, Bayer only, with as-shot white balance and colour matrix
    .npy                2-D Bayer, or (H, W, 3) RGB
    16-bit PNG or TIFF  single-channel Bayer
    JPEG / PNG / others treated as a finished photo and unprocessed

Files are identified by content first; the extension is only a hint. Every failure is an InputError.
For .npy and 16-bit files the CFA comes from a filename token (`scene_bggr.npy`), otherwise RGGB is
assumed and meta["cfa_assumed"] is True.
"""
from __future__ import annotations

import io
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
from PIL import Image, ImageOps

from isp.data.synthetic import MIN_SIDE, RGB2XYZ, unprocess_jpeg
from isp.types import InputError, RawImage

MAX_MEGAPIXELS = 24  # above this, downscale; keeps 12 MP phone shots at full resolution
LIMIT_MEGAPIXELS = 120  # above this, refuse before decoding
MAX_FILE_BYTES = 512 * 2**20

Image.init()  # Image.open(formats=...) raises KeyError for plugins that are not loaded yet
PIL_FORMATS = ["JPEG", "PNG", "TIFF", "BMP", "WEBP"]  # MPO phone shots open through JPEG
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
# Offset that moves each 2x2 layout onto RGGB.
CFA_SHIFT = {"RGGB": (0, 0), "GRBG": (0, 1), "GBRG": (1, 0), "BGGR": (1, 1)}
CFA_TOKEN = re.compile(r"(?<![a-z])(rggb|grbg|gbrg|bggr)(?![a-z])")
ROTATIONS = {0: 0, 3: 2, 5: 1, 6: 3}  # LibRaw flip code -> quarter turns anticlockwise
TOO_LARGE = (
    f"This image is too large to process (the limit is {LIMIT_MEGAPIXELS} megapixels). Please upload a smaller one."
)
TOO_SMALL = f"This image is too small. Please use one at least {MIN_SIDE} pixels on each side."


def load_input(source: str | bytes, filename: str | None = None) -> RawImage:
    data, name = _read(source, filename)
    decoders = [_from_pil, _from_rawpy] if _looks_like_photo(data, name) else [_from_rawpy, _from_pil]
    if data.startswith(b"\x93NUMPY"):
        decoders = [_from_npy]
    for decoder in decoders:
        try:
            return decoder(data, name)
        except InputError:
            raise
        except Image.DecompressionBombError as e:
            raise InputError(TOO_LARGE) from e
        except rawpy.LibRawFileUnsupportedError:
            continue  # not a RAW file at all
        except rawpy.LibRawDataError as e:
            # A RAW that LibRaw recognised. Stop here: Pillow would happily return its embedded preview.
            raise InputError("This RAW file looks damaged, so we cannot read it. Please try another file.") from e
        except Exception:  # decoders raise dozens of types on hostile input; the next one gets a turn
            continue
    label = f'"{name}"' if name else "this file"
    raise InputError(
        f"We could not read {label}. It may be damaged or in an unsupported format. "
        "Please try a JPEG, PNG, DNG or Bayer .npy file."
    )


def _read(source: Any, filename: str | None) -> tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray)):
        data, name = bytes(source), filename or ""
    elif isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.is_file():
            raise InputError(f'We could not find a file at "{path}".')
        if path.stat().st_size > MAX_FILE_BYTES:
            raise InputError(TOO_LARGE)
        data, name = path.read_bytes(), filename or path.name
    else:
        raise InputError("Please provide an image file.")
    if not data:
        raise InputError("This file is empty. Please upload it again.")
    if len(data) > MAX_FILE_BYTES:
        raise InputError(TOO_LARGE)
    return data, name


def _looks_like_photo(data: bytes, name: str) -> bool:
    if data.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")):
        return True  # the content wins even if the name says .dng
    return Path(name).suffix.lower() in PHOTO_SUFFIXES


def _downscale_factor(h: int, w: int) -> int:
    """Refuses frames over the limit. Returns the integer factor that brings a frame under MAX_MEGAPIXELS."""
    megapixels = h * w / 1e6
    if megapixels > LIMIT_MEGAPIXELS:
        raise InputError(TOO_LARGE)
    return max(1, math.ceil(math.sqrt(megapixels / MAX_MEGAPIXELS)))


def _named_cfa(name: str) -> str | None:
    match = CFA_TOKEN.search(Path(name).stem.lower())
    return match.group(1).upper() if match else None


def _finish_bayer(mosaic: np.ndarray, cfa: str, **fields: Any) -> RawImage:
    """Mosaic in [0, 1] with the given layout -> RGGB, even size, under the size limit."""
    original = mosaic.shape
    k = _downscale_factor(*original)
    dy, dx = CFA_SHIFT[cfa]
    mosaic = mosaic[dy:, dx:]
    if min(mosaic.shape) < MIN_SIDE:
        raise InputError(TOO_SMALL)
    # Trim to whole 2k blocks (k = 1 is the plain even crop); then average same-colour sites per block.
    h, w = mosaic.shape[0] // (2 * k), mosaic.shape[1] // (2 * k)
    mosaic = mosaic[: h * 2 * k, : w * 2 * k]
    meta = fields.pop("meta", {})
    if k > 1:
        mosaic = mosaic.reshape(h, k, 2, w, k, 2).mean(axis=(1, 4)).reshape(2 * h, 2 * w)
        meta["downscaled_from"] = original
    return RawImage(np.ascontiguousarray(mosaic, np.float32), meta=meta, **fields)


def _unit_range(array: np.ndarray) -> np.ndarray:
    """Numeric array -> float32 in [0, 1]. Integers scale by the smallest usual sensor depth that fits."""
    if array.dtype.kind == "f":
        if not np.isfinite(array).all():
            raise InputError("This file contains invalid pixel values (NaN or infinity).")
        if array.size and (array.min() < 0 or array.max() > 1):
            raise InputError("Floating-point images must have values between 0 and 1.")
        return array.astype(np.float32, copy=False)
    if array.dtype.kind not in "ui":
        raise InputError(f"Pixel values of type {array.dtype} are not supported. Please use integers or floats.")
    if array.size and array.min() < 0:
        raise InputError("This file contains negative pixel values.")
    peak = int(array.max()) if array.size else 0
    depths = (8, 10, 12, 14, 16) if array.dtype.itemsize == 1 else (10, 12, 14, 16)
    depth = next((d for d in depths if peak < 2**d), None)
    if depth is None:
        raise InputError("Pixel values are larger than 16 bits, which is not supported.")
    return array.astype(np.float32) / np.float32(2**depth - 1)


def _bayer_from_array(array: np.ndarray, name: str) -> RawImage:
    if min(array.shape) < MIN_SIDE:
        raise InputError(TOO_SMALL)
    _downscale_factor(*array.shape)  # refuse before converting a huge array to float
    cfa = _named_cfa(name)
    meta = {"source": "bayer", "cfa": cfa or "RGGB", "cfa_assumed": cfa is None}
    return _finish_bayer(_unit_range(array), cfa or "RGGB", meta=meta)


def _from_npy(data: bytes, name: str) -> RawImage:
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)  # pickled arrays can run code: never allow
    except Exception as e:
        raise InputError(f'"{name or "This file"}" is not a plain numeric .npy array.') from e
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 2:
        return _bayer_from_array(array, name)
    if array.ndim == 3 and array.shape[2] == 3:
        k = _downscale_factor(*array.shape[:2])
        return unprocess_jpeg(_unit_range(array[::k, ::k]))
    raise InputError(f"Expected a 2-D Bayer array or an (H, W, 3) image, but this array has shape {array.shape}.")


def _from_pil(data: bytes, name: str) -> RawImage:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # bomb and truncation warnings: our own checks below and above decide
        image = Image.open(io.BytesIO(data), formats=PIL_FORMATS)
    width, height = image.size
    k = _downscale_factor(height, width)
    if min(width, height) < MIN_SIDE:
        raise InputError(TOO_SMALL)
    if image.mode.startswith("I"):  # 16-bit, single channel
        return _bayer_from_array(np.asarray(image), name)

    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        image = Image.new("RGBA", rgba.size, "white")
        image.alpha_composite(rgba)
    image = image.convert("RGB")
    if k > 1:
        image = image.reduce(k)
    raw = unprocess_jpeg(np.asarray(image))
    if k > 1:
        raw.meta["downscaled_from"] = (height, width)
    return raw


def _from_rawpy(data: bytes, name: str) -> RawImage:
    with rawpy.imread(io.BytesIO(data)) as r:
        pattern = r.raw_pattern
        if pattern is None or pattern.shape != (2, 2):
            if pattern is not None and pattern.shape == (6, 6):
                raise InputError(
                    "This is a Fujifilm X-Trans RAW, which is not supported. "
                    "Please upload a Bayer RAW or a JPEG from the same camera."
                )
            raise InputError("This RAW file does not use a Bayer sensor, so it is not supported. Please upload a JPEG.")
        turns = ROTATIONS.get(r.sizes.flip, 0)
        colours = np.rot90(r.raw_colors_visible, turns)
        pixels = np.rot90(r.raw_image_visible, turns).astype(np.float32)
        desc = r.color_desc.decode("ascii")
        cfa = "".join(desc[i] for i in colours[:2, :2].ravel())
        if cfa not in CFA_SHIFT:
            raise InputError(f"This RAW file uses a {cfa} colour filter, which is not supported. Please upload a JPEG.")
        black = np.asarray(r.black_level_per_channel, np.float32)[colours]
        white = np.float32(r.white_level or 65535)
        wb = np.asarray(r.camera_whitebalance, np.float64)
        ccm = _camera_ccm(np.asarray(r.rgb_xyz_matrix, np.float64)[:3], np.asarray(r.color_matrix, np.float64)[:3, :3])

    mosaic = np.clip((pixels - black) / np.maximum(white - black, 1.0), 0.0, 1.0)
    wb_gains = (wb[:3] / wb[1]).astype(np.float32) if (wb[:3] > 0).all() else None
    meta = {"source": "raw", "cfa": cfa, "cfa_assumed": False}
    return _finish_bayer(mosaic, cfa, wb_gains=wb_gains, ccm=ccm, meta=meta)


def _camera_ccm(xyz2cam: np.ndarray, color_matrix: np.ndarray) -> np.ndarray | None:
    """Camera RGB -> linear sRGB, or None if the file carries no colour data.

    LibRaw fills rgb_xyz_matrix for known cameras and color_matrix for DNGs, leaving the other at zero.
    """
    if np.isfinite(xyz2cam).all() and xyz2cam.any():
        rgb2cam = xyz2cam @ RGB2XYZ
        sums = rgb2cam.sum(axis=1, keepdims=True)
        if (np.abs(sums) > 1e-6).all() and abs(np.linalg.det(rgb2cam / sums)) > 1e-6:
            return np.linalg.inv(rgb2cam / sums).astype(np.float32)
    if np.isfinite(color_matrix).all() and color_matrix.any() and not np.allclose(color_matrix, np.eye(3)):
        return color_matrix.astype(np.float32)
    return None
