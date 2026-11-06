"""Tiny stand-ins for the real pipelines. Numbers they produce exist only inside tests."""
import numpy as np

from isp.eval.harness import Sample
from isp.eval.metrics import delta_e2000, psnr, ssim
from isp.types import PipelineResult, RawImage


def fake_lpips(pred, target):
    return float(np.abs(pred - target).mean())


# Real PSNR/SSIM/DeltaE, with a cheap LPIPS substitute so harness tests need no network weights.
FAST_METRICS = {"psnr": psnr, "ssim": ssim, "delta_e2000": delta_e2000, "lpips": fake_lpips}


class FakePipeline:
    """Greyscale passthrough of the mosaic plus a constant bias; records what it was given."""

    def __init__(self, name, bias=0.0, info=None, mutate_input=False):
        self.name, self.bias, self.info, self.mutate_input = name, bias, info or {}, mutate_input
        self.seen = []

    def run(self, raw):
        self.seen.append(raw.bayer.copy())
        rgb = np.repeat(raw.bayer[..., None], 3, axis=2) + np.float32(self.bias)
        if self.mutate_input:
            raw.bayer[:] = 0
        return PipelineResult(rgb=np.clip(rgb, 0, 1).astype(np.float32), info=dict(self.info))


def trio():
    return [FakePipeline("classical", 0.10), FakePipeline("learned", 0.05), FakePipeline("hybrid", 0.01)]


def make_sample(image_id="kodim01", seed=0, size=64):
    bayer = np.random.default_rng(seed).uniform(0.2, 0.8, (size, size)).astype(np.float32)
    return Sample(image_id, RawImage(bayer=bayer), np.repeat(bayer[..., None], 3, axis=2))


def fake_synth(rgb, fields, seed):
    """Degrades by the ev and noise fields so sweeps vary; deterministic in seed."""
    noise = np.random.default_rng(seed).standard_normal(rgb.shape[:2]).astype(np.float32)
    gain = np.float32(2.0 ** fields.get("ev", 0.0))
    bayer = np.clip(rgb[..., 1] * gain + fields.get("iso_a", 0.0) * noise, 0, 1).astype(np.float32)
    return RawImage(bayer=bayer), rgb


CONFIG = {
    "seed": 7,
    "base": {},
    "standard": {"iso": "low", "ev": "ev0", "cast": "neutral"},
    "axes": {
        "iso": {"low": {"iso_a": 0.0}, "high": {"iso_a": 0.1}},
        "ev": {"ev0": {"ev": 0.0}, "ev-1": {"ev": -1.0}, "ev-3": {"ev": -3.0}},
        "cast": {"neutral": {"illuminant": [1.0, 1.0, 1.0]}, "warm": {"illuminant": [1.4, 1.0, 0.7]}},
    },
}
