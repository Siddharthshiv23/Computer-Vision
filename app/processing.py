"""Everything between an uploaded file and displayable results. No UI code."""
from __future__ import annotations

import dataclasses
import io
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.adapter import Slot
from isp.types import InputError, RawImage

log = logging.getLogger(__name__)

MAX_PREVIEW_PIXELS = 2_000_000
MAX_REFERENCE_PIXELS = 64_000_000
MIN_SIDE = 32
ASPECT_TOLERANCE = 0.02
TOO_SMALL = f"This image is too small to process. Each side must be at least {MIN_SIDE} pixels after preview scaling."
UNREADABLE = "This file could not be read as an image. Try a JPEG, PNG or DNG."
BAD_REFERENCE = "The reference image could not be read. Try a JPEG or PNG of the same scene."
PIPELINE_FAILED = "This pipeline could not process the image. The other results are unaffected."
BAD_OUTPUT = "This pipeline produced invalid pixels, so its result is not shown."


@dataclass
class Outcome:
    name: str
    label: str
    status: str  # unavailable, waiting, running, done, failed
    message: str = ""
    rgb: np.ndarray | None = None  # uint8, ready to display
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    info: dict = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class Report:
    filename: str = ""
    note: str = ""
    error: str = ""
    status: str = ""
    has_reference: bool = False
    metrics_note: str = ""
    outcomes: list[Outcome] = field(default_factory=list)


def read_and_delete(path: str) -> bytes:
    """Take an upload into memory and remove Gradio's temporary copy of it."""
    try:
        data = Path(path).read_bytes()
        os.unlink(path)
    except OSError:
        raise InputError("That upload is no longer available. Choose the file again.") from None
    return data


def _downscale_mosaic(bayer: np.ndarray, plane_size: tuple[int, int]) -> np.ndarray:
    """Area-resample each Bayer plane separately so the RGGB phase survives."""
    ph, pw = plane_size
    out = np.empty((ph * 2, pw * 2), np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            plane = np.ascontiguousarray(bayer[dy::2, dx::2])
            out[dy::2, dx::2] = cv2.resize(plane, (pw, ph), interpolation=cv2.INTER_AREA)
    return out


def prepare_raw(raw: RawImage) -> tuple[RawImage, str]:
    """Check a mosaic and bring it to preview size. Returns the mosaic and a note for the user."""
    bayer = raw.bayer
    h, w = bayer.shape
    if not np.isfinite(bayer).all():
        raise InputError("This file contains invalid pixel values, so it cannot be processed.")
    if h % 2 or w % 2:
        h, w = h - h % 2, w - w % 2
        bayer = bayer[:h, :w]
        raw = dataclasses.replace(raw, bayer=bayer)
    note = ""
    if h * w > MAX_PREVIEW_PIXELS:
        scale = math.sqrt(MAX_PREVIEW_PIXELS / (h * w))
        ph, pw = int(h / 2 * scale), int(w / 2 * scale)
        if min(ph, pw) * 2 < MIN_SIDE:
            raise InputError(TOO_SMALL)
        raw = dataclasses.replace(raw, bayer=_downscale_mosaic(bayer, (ph, pw)))
        note = f"Previewing at {pw * 2} x {ph * 2}, reduced from {w} x {h} to stay within 2 MP."
    elif min(h, w) < MIN_SIDE:
        raise InputError(TOO_SMALL)
    return raw, note


def read_reference(data: bytes) -> np.ndarray:
    """Decode a reference image to float32 RGB in [0, 1]."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            if image.width * image.height > MAX_REFERENCE_PIXELS:
                raise InputError("The reference image is too large. Use one under 64 megapixels.")
            image = ImageOps.exif_transpose(image)
            if image.mode.startswith("I"):  # 16-bit greyscale would clip in convert("RGB")
                depth = np.asarray(image, np.float32) / 65535.0
                return np.repeat(np.clip(depth, 0, 1)[..., None], 3, axis=2)
            return np.asarray(image.convert("RGB"), np.float32) / 255.0
    except InputError:
        raise
    except Exception:
        raise InputError(BAD_REFERENCE) from None


def fit_reference(ref: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Match a reference to the preview: crop odd edges, then follow the downscale."""
    h, w = shape
    ref = ref[: ref.shape[0] - ref.shape[0] % 2, : ref.shape[1] - ref.shape[1] % 2]
    if ref.shape[:2] == (h, w):
        return ref
    if abs(ref.shape[1] / ref.shape[0] / (w / h) - 1) > ASPECT_TOLERANCE:
        raise InputError(
            f"The reference image has a different shape from the input ({ref.shape[1]} x {ref.shape[0]}). "
            "Use a reference of the same scene."
        )
    return cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)


def to_display(image: np.ndarray) -> np.ndarray | None:
    """Return (H, W, 3) uint8, or None when the array cannot be shown truthfully."""
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    if image.dtype == np.uint8:
        return image
    if not np.isfinite(image).all():
        return None
    return (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)


def _run(slot: Slot, outcome: Outcome, raw: RawImage) -> np.ndarray | None:
    """Fill in an outcome. Returns the float result for scoring."""
    start = time.perf_counter()
    try:
        result = slot.pipeline.run(raw)
        outcome.total_ms = (time.perf_counter() - start) * 1000
        outcome.rgb = to_display(result.rgb)
        if outcome.rgb is None:
            outcome.status, outcome.message = "failed", BAD_OUTPUT
            return None
        shown = {name: to_display(image) for name, image in result.stages.items()}
        outcome.stages = {name: image for name, image in shown.items() if image is not None}
        outcome.timings_ms = dict(result.timings_ms)
        outcome.info = dict(result.info)
    except InputError as exc:
        outcome.status, outcome.message = "failed", str(exc)
        return None
    except Exception:
        log.exception("%s pipeline failed", slot.name)
        outcome.rgb = None
        outcome.status, outcome.message = "failed", PIPELINE_FAILED
        return None
    outcome.status = "done"
    return result.rgb


def _score(rgb: np.ndarray, ref: np.ndarray, metrics: dict[str, Callable]) -> dict[str, float]:
    if rgb.shape != ref.shape:
        return {}
    scores = {}
    for name, fn in metrics.items():
        try:
            value = float(fn(rgb, ref))
        except Exception:
            log.exception("metric %s failed", name)
            continue
        if math.isfinite(value):
            scores[name] = value
    return scores


def develop(
    upload_path: str | None,
    reference_path: str | None,
    *,
    slots: list[Slot],
    load: Callable | None,
    load_reason: str,
    metrics: dict[str, Callable] | None,
    metrics_reason: str,
) -> Iterator[Report]:
    """Run every available pipeline, yielding the report after each step.

    Deletes both uploads as soon as they are read, so callers must pass disposable copies.
    """
    report = Report(
        outcomes=[Outcome(s.name, s.label, "waiting" if s.pipeline else "unavailable", s.reason) for s in slots]
    )
    try:
        data, ref_data = (read_and_delete(p) if p else None for p in (upload_path, reference_path))
        if data is None:
            raise InputError("Choose an image first.")
        report.filename = Path(upload_path).name
        if load is None:
            raise InputError(load_reason)
        report.status = "Reading the image"
        yield report
        raw, report.note = prepare_raw(load(data, filename=report.filename))
        ref = None
        if ref_data is not None:
            report.has_reference = True
            ref = fit_reference(read_reference(ref_data), raw.bayer.shape)
            report.metrics_note = metrics_reason if metrics is None else ""
    except InputError as exc:
        report.error, report.status = str(exc), ""
        yield report
        return
    except Exception:
        log.exception("reading upload failed")
        report.error, report.status = UNREADABLE, ""
        yield report
        return

    for slot, outcome in zip(slots, report.outcomes):
        if slot.pipeline is None:
            continue
        outcome.status = "running"
        report.status = f"Developing with {slot.label}"
        yield report
        rgb = _run(slot, outcome, raw)
        if rgb is not None and ref is not None and metrics is not None:
            outcome.metrics = _score(rgb, ref, metrics)
    report.status = ""
    yield report
