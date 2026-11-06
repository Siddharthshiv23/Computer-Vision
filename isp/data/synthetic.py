"""Synthetic sensor model: sRGB -> RAW by unprocessing (Brooks et al., CVPR 2019).

Forward model, in order: invert tone curve, invert gamma, invert CCM, apply the illuminant
(with Brooks' highlight protection), apply exposure, mosaic to RGGB, add Poisson-Gaussian noise,
clip, quantise to 12 bits. Camera matrices and the noise line are from the Apache-2.0 reference code.

Deliberate differences from Brooks:
* Exposure is an explicit 0 to -3 EV, applied after the illuminant so highlights dim too.
* Illuminants and noise reach further (PRD F-DL-4), so training covers every level in
  configs/test_conditions.yaml. Brooks draws red gain 1.9-2.4, blue 1.5-1.9 and shot noise up to 0.012.
* Shot noise is true Poisson rather than a Gaussian approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from isp.data.splits import REPO_ROOT
from isp.types import InputError, RawImage

MIN_SIDE = 32
WHITE_CODE = 4095  # 12-bit sensor
TEST_CONDITIONS = REPO_ROOT / "configs" / "test_conditions.yaml"

XYZ2CAMS = np.array(
    [
        [[1.0234, -0.2969, -0.2266], [-0.5625, 1.6328, -0.0469], [-0.0703, 0.2188, 0.6406]],
        [[0.4913, -0.0541, -0.0202], [-0.613, 1.3513, 0.2906], [-0.1564, 0.2151, 0.7183]],
        [[0.838, -0.263, -0.0639], [-0.2887, 1.0725, 0.2496], [-0.0627, 0.1427, 0.5438]],
        [[0.6596, -0.2079, -0.0562], [-0.4782, 1.3016, 0.1933], [-0.097, 0.1581, 0.5181]],
    ]
)
RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
# Middle of Brooks' white balance ranges (red 2.15, blue 1.70): what unprocess_jpeg assumes.
AVERAGE_ILLUMINANT = np.array([1 / 2.15, 1.0, 1 / 1.70])


@dataclass
class SensorCondition:
    """One simulated capture.

    iso_a, iso_b: noise variance = iso_a * signal + iso_b, signal in [0, 1].
    ev: exposure in stops. illuminant: raw response to white as (r, 1, b); WB gains are its reciprocal.
    ccm: 3x3 camera RGB -> linear sRGB, same meaning as RawImage.ccm.
    seed: noise seed. None draws one from numpy's global generator, so np.random.seed() also fixes the noise.
    """

    iso_a: float
    iso_b: float
    ev: float
    illuminant: np.ndarray
    ccm: np.ndarray
    seed: int | None = None


def camera_ccm(weights: np.ndarray) -> np.ndarray:
    """Camera RGB -> linear sRGB for a convex mix of the real cameras. Maps white to white."""
    weights = np.asarray(weights, np.float64)
    xyz2cam = np.tensordot(weights / weights.sum(), XYZ2CAMS, axes=1)
    rgb2cam = xyz2cam @ RGB2XYZ
    return np.linalg.inv(rgb2cam / rgb2cam.sum(axis=1, keepdims=True))


def sample_condition(rng: np.random.Generator) -> SensorCondition:
    log_a = rng.uniform(np.log(1e-4), np.log(0.02))
    log_b = 2.18 * log_a + 1.20 + rng.normal(0.0, 0.26)
    # Raw response to white, relative to green: from strongly green-dominant sensors to warm or cool casts.
    red = np.exp(rng.uniform(np.log(1 / 2.8), np.log(1.4)))
    blue = np.exp(rng.uniform(np.log(1 / 2.4), np.log(1.4)))
    return SensorCondition(
        iso_a=float(np.exp(log_a)),
        iso_b=float(np.exp(log_b)),
        ev=float(rng.uniform(-3.0, 0.0)),
        illuminant=np.array([red, 1.0, blue]),
        ccm=camera_ccm(rng.uniform(1e-8, 1e8, len(XYZ2CAMS))),
        seed=int(rng.integers(2**31)),
    )


def load_test_conditions(path: Path = TEST_CONDITIONS) -> dict[str, SensorCondition]:
    """The fixed Kodak24 conditions: "standard", plus one per axis level ("ev_-3", "iso_6400", "warm"...).

    A level changes its own axis and keeps the standard level on the others. Each condition carries
    the file's base noise seed.
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    def build(levels: dict[str, str]) -> SensorCondition:
        c = dict(config["base"])
        for axis, label in levels.items():
            c.update(config["axes"][axis][label])
        return SensorCondition(
            iso_a=c["iso_a"],
            iso_b=c["iso_b"],
            ev=c["ev"],
            illuminant=np.array(c["illuminant"], np.float64),
            ccm=np.array(c["ccm"], np.float64),
            seed=config["seed"],
        )

    out = {"standard": build(config["standard"])}
    for axis, levels in config["axes"].items():
        for label in levels:
            out[label] = build({**config["standard"], axis: label})
    return out


def _prepare_rgb(rgb: Any) -> np.ndarray:
    """Validate an RGB array and return it as float32 in [0, 1] with even height and width."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise InputError("Expected a colour image with three channels (red, green, blue).")
    if min(rgb.shape[:2]) < MIN_SIDE:
        raise InputError(f"This image is too small. Please use one at least {MIN_SIDE} pixels on each side.")
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    rgb = rgb.astype(np.float32, copy=False)
    if not np.isfinite(rgb).all():
        raise InputError("This image contains invalid pixel values (NaN or infinity).")
    h, w = rgb.shape[0] & ~1, rgb.shape[1] & ~1
    return np.clip(rgb[:h, :w], 0.0, 1.0)


def _to_camera(rgb: np.ndarray, ccm: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
    """sRGB -> linear camera RGB under the illuminant, in [0, 1]."""
    linear = (0.5 - np.sin(np.arcsin(1.0 - 2.0 * rgb) / 3.0)) ** 2.2
    camera = linear @ np.linalg.inv(np.asarray(ccm, np.float64)).T.astype(np.float32)
    # Gains fade to 1 near white, so clipped highlights stay white instead of taking the illuminant's tint.
    gains = np.asarray(illuminant, np.float32)
    grey = camera.mean(axis=2, keepdims=True)
    mask = (np.maximum(grey - 0.9, 0.0) / 0.1) ** 2
    return np.clip(camera * np.maximum(mask + (1.0 - mask) * gains, gains), 0.0, 1.0)


def _mosaic(image: np.ndarray) -> np.ndarray:
    bayer = np.empty(image.shape[:2], np.float32)
    bayer[0::2, 0::2] = image[0::2, 0::2, 0]
    bayer[0::2, 1::2] = image[0::2, 1::2, 1]
    bayer[1::2, 0::2] = image[1::2, 0::2, 1]
    bayer[1::2, 1::2] = image[1::2, 1::2, 2]
    return bayer


def synthesize(rgb: np.ndarray, cond: SensorCondition) -> tuple[RawImage, np.ndarray]:
    """Returns the degraded RAW and its target: the clean sRGB input, at the same (even) size.

    The target ignores cond.ev, so at -3 EV it is still the normally exposed image: recovering the
    exposure is part of the task.
    RawImage.wb_gains stays None because handling white balance errors is part of the task. The true
    parameters are in meta["truth"] for evaluation only; pipelines must not read them.
    """
    target = _prepare_rgb(rgb)
    signal = _mosaic(_to_camera(target, cond.ccm, cond.illuminant)) * np.float32(2.0**cond.ev)

    seed = int(np.random.randint(2**31)) if cond.seed is None else cond.seed
    rng = np.random.default_rng(seed)
    noisy = signal.astype(np.float64)
    if cond.iso_a > 0:
        noisy = cond.iso_a * rng.poisson(noisy / cond.iso_a)
    if cond.iso_b > 0:
        noisy = noisy + rng.normal(0.0, np.sqrt(cond.iso_b), noisy.shape)
    bayer = (np.round(np.clip(noisy, 0.0, 1.0) * WHITE_CODE) / WHITE_CODE).astype(np.float32)

    truth = {
        "noise_a": cond.iso_a,
        "noise_b": cond.iso_b,
        "ev": cond.ev,
        "illuminant": np.asarray(cond.illuminant),
        "seed": seed,
    }
    raw = RawImage(bayer, ccm=np.asarray(cond.ccm, np.float32), meta={"source": "synthetic", "truth": truth})
    return raw, target


def unprocess_jpeg(rgb: np.ndarray) -> RawImage:
    """Turn an already-rendered photo into RAW for the pipelines: no noise, no exposure change.

    Assumes the average camera and illuminant, and reports them as the as-shot metadata, so a pipeline
    that honours wb_gains and ccm gets the photo's own colours back.
    """
    ccm = camera_ccm(np.ones(len(XYZ2CAMS)))
    bayer = _mosaic(_to_camera(_prepare_rgb(rgb), ccm, AVERAGE_ILLUMINANT))
    return RawImage(
        bayer,
        wb_gains=(1.0 / AVERAGE_ILLUMINANT).astype(np.float32),
        ccm=ccm.astype(np.float32),
        meta={"source": "unprocessed"},
    )
