"""Classical ISP: the conventional baseline and the front end of the hybrid (F-CV-1..6)."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from isp.classical import white_balance as wb
from isp.classical.color import apply_ccm, resolve_ccm, srgb_encode
from isp.classical.demosaic import demosaic
from isp.classical.denoise import bilateral, denoise_luma_chroma, denoise_raw, estimate_noise, nlm
from isp.classical.sharpen import noise_aware_unsharp
from isp.classical.tone import auto_exposure, clahe, gamma22, s_curve, srgb_gamma
from isp.types import InputError, PipelineResult, RawImage

_AWB = {"gray_world": wb.gray_world, "shades_of_gray": wb.shades_of_gray,
        "gray_edge": wb.gray_edge, "white_patch": wb.white_patch, "fusion": wb.fuse_awb}


@dataclass
class ClassicalConfig:
    """Switches used by the ablations. Defaults are the full classical baseline."""

    demosaic: str = "malvar"          # bilinear | malvar | opencv_ea
    awb: str = "fusion"               # gray_world | shades_of_gray | gray_edge | white_patch | fusion | metadata
    denoise: bool = True              # RAW-domain VST + guided filter
    denoise_strength: float = 1.0
    post_denoise: str = "none"        # none | luma_chroma | bilateral | nlm (on the sRGB output)
    gamma: str = "srgb"               # srgb | gamma22
    s_curve: float = 0.25
    clahe_clip: float = 0.0           # 0 disables CLAHE
    sharpen: float = 0.6


def _gray3(x: np.ndarray) -> np.ndarray:
    """Displayable (H, W, 3) view of a single-channel linear mosaic."""
    return np.repeat(srgb_encode(x)[..., None], 3, axis=2)


def _estimate_gains(rgb: np.ndarray, raw: RawImage, mode: str) -> np.ndarray:
    if mode == "metadata" and raw.wb_gains is not None:
        g = np.asarray(raw.wb_gains, np.float32)
        return g / max(float(g[1]), 1e-6)
    if mode == "metadata":
        mode = "fusion"
    if mode not in _AWB:
        raise InputError(f"Unknown white balance mode '{mode}'.")
    return _AWB[mode](rgb[::2, ::2])  # illuminant statistics do not need every pixel


def _validate(raw: RawImage) -> np.ndarray:
    b = raw.bayer
    if b.ndim != 2 or b.shape[0] < 2 or b.shape[1] < 2 or b.shape[0] % 2 or b.shape[1] % 2:
        raise InputError("RAW data must be a single-channel Bayer mosaic with even width and height.")
    return np.clip(np.nan_to_num(b.astype(np.float32)), 0, 1)


def _front_end(raw: RawImage, denoise: bool, cfg: ClassicalConfig, timings: dict[str, float]):
    stages: dict[str, np.ndarray] = {}
    t = [perf_counter()]

    def lap(name: str) -> None:
        now = perf_counter()
        timings[name] = (now - t[0]) * 1000
        t[0] = now

    bayer = _validate(raw)
    stages["raw"] = _gray3(bayer)
    a, b = estimate_noise(bayer)
    lap("noise_estimation")
    if denoise:
        bayer = denoise_raw(bayer, a, b, cfg.denoise_strength)
        stages["denoised_raw"] = _gray3(bayer)
        lap("denoised_raw")
    lin = demosaic(bayer, cfg.demosaic)
    stages["demosaic"] = srgb_encode(lin)
    lap("demosaic")
    lin = wb.apply_gains(lin, _estimate_gains(lin, raw, cfg.awb))
    stages["white_balance"] = srgb_encode(lin)
    lap("white_balance")
    lin = apply_ccm(lin, resolve_ccm(raw.ccm))
    stages["ccm"] = srgb_encode(lin)
    lap("ccm")
    lin, gain = auto_exposure(lin)
    stages["exposure"] = srgb_encode(lin)
    lap("exposure")
    out = (gamma22 if cfg.gamma == "gamma22" else srgb_gamma)(lin)
    out = s_curve(out, cfg.s_curve)
    if cfg.clahe_clip > 0:
        out = clahe(out, cfg.clahe_clip)
    out = np.clip(np.nan_to_num(out), 0, 1).astype(np.float32)
    stages["tone"] = out
    lap("tone")
    return out, stages, {"noise_a": a, "noise_b": b, "exposure": gain}


def front_end(raw: RawImage, denoise: bool, config: ClassicalConfig | None = None):
    """Classical front end used by the hybrid: normalised RAW -> sRGB for the network.

    Returns (rgb, stages, {"noise_a", "noise_b", "exposure"}). `denoise` turns the RAW-domain
    denoiser on (classical mode) or off (hybrid mode, where the network denoises).
    """
    return _front_end(raw, denoise, config or ClassicalConfig(), {})


def output_sigma(noise_a: float, noise_b: float, exposure: float) -> float:
    """Approximate noise sd in the sRGB output at mid-grey, for coring and post-denoising.

    Raw sd at the pre-gain level, scaled by the exposure gain and by the local slope (~1.16)
    of the gamma curve at 0.18.
    """
    lin_sd = exposure * np.sqrt(max(noise_a * 0.18 / exposure + noise_b, 0.0))
    return float(lin_sd * 1.16)


class ClassicalISP:
    """Conventional ISP baseline: every stage is a fixed, hand-designed algorithm."""

    name = "classical"

    def __init__(self, config: ClassicalConfig | None = None) -> None:
        self.config = config or ClassicalConfig()

    def _post_denoise(self, rgb: np.ndarray, sigma: float) -> np.ndarray:
        mode, s = self.config.post_denoise, self.config.denoise_strength
        if mode == "luma_chroma":
            return denoise_luma_chroma(rgb, sigma, s)
        if mode == "bilateral":
            return bilateral(rgb, sigma * s)
        if mode == "nlm":
            return nlm(rgb, sigma * s)
        return rgb

    def run(self, raw: RawImage) -> PipelineResult:
        start = perf_counter()
        timings: dict[str, float] = {}
        cfg = self.config
        rgb, stages, meta = _front_end(raw, cfg.denoise, cfg, timings)
        sigma = output_sigma(**meta)
        t = perf_counter()
        if cfg.post_denoise != "none":
            rgb = self._post_denoise(rgb, sigma)
            stages["post_denoise"] = rgb
            timings["post_denoise"] = (perf_counter() - t) * 1000
        t = perf_counter()
        rgb = noise_aware_unsharp(rgb, sigma, cfg.sharpen)
        rgb = np.clip(np.nan_to_num(rgb), 0, 1).astype(np.float32)
        stages["sharpen"] = rgb
        timings["sharpen"] = (perf_counter() - t) * 1000
        timings["total"] = (perf_counter() - start) * 1000
        return PipelineResult(rgb=rgb, stages=stages, timings_ms=timings, info=dict(meta))
