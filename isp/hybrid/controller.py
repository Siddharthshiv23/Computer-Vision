"""AdaptiveController: turns the estimated noise and exposure into per-image pipeline settings (F-AD-1).

All thresholds are plain constructor arguments. They are UNTUNED defaults chosen from the scale of
`output_sigma` (about 0.007 for a clean ISO-100 frame, 0.1 for heavy noise, 0.5 for extreme); tune them
on the validation split (McMaster), never on Kodak24.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from isp.classical.pipeline import output_sigma
from isp.types import InputError


@dataclass
class AdaptiveController:
    clean_sigma: float = 0.01       # below this the classical output is already clean: skip the network
    extreme_sigma: float = 0.3      # above this the larger teacher is worth its cost
    teacher_max_mp: float = 2.0     # the teacher is several times slower, so only for small frames
    base_sharpen: float = 0.6       # same amount as the classical baseline
    sharpen_fade_sigma: float = 0.15  # sharpening fades out linearly up to this noise level
    tile_above_mp: float = 2.0
    tile: int = 1024
    overlap: int = 64

    def decide(self, noise_a: float, noise_b: float, exposure: float, megapixels: float) -> dict:
        """noise_a, noise_b: Poisson-Gaussian parameters in the RAW domain (before the exposure gain).

        Returns denoise_strength (0..1, how far the output moves from the front end towards the
        network result), sharpen_amount, model ("student" | "teacher" | "classical"), tiling and the
        noise_sigma that drove the decision. An unusable estimate (NaN, Inf, negative) selects
        the classical path, which needs no noise map.
        """
        if not megapixels > 0:
            raise InputError("Image size must be positive.")
        usable = all(map(math.isfinite, (noise_a, noise_b, exposure))) and min(noise_a, noise_b) >= 0 and exposure > 0
        sigma = output_sigma(noise_a, noise_b, exposure) if usable else 0.0
        return {
            "noise_sigma": sigma,
            "denoise_strength": min(1.0, max(0.5, sigma / 0.05)),
            "sharpen_amount": self.base_sharpen * min(1.0, max(0.2, 1.0 - sigma / self.sharpen_fade_sigma)),
            "model": self._model(usable, sigma, megapixels),
            "tiling": {"enabled": megapixels > self.tile_above_mp, "tile": self.tile, "overlap": self.overlap},
        }

    def _model(self, usable: bool, sigma: float, megapixels: float) -> str:
        if not usable or sigma < self.clean_sigma:
            return "classical"
        if sigma > self.extreme_sigma and megapixels <= self.teacher_max_mp:
            return "teacher"
        return "student"
