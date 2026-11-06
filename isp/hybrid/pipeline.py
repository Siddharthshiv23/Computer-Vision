"""HybridISP: classical front end -> LiteISPNet -> noise-aware sharpen -> clip (F-HY-1..3, F-AD-1)."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from isp.classical.pipeline import ClassicalConfig, ClassicalISP, front_end, output_sigma
from isp.classical.sharpen import noise_aware_unsharp
from isp.hybrid.backend import Backend
from isp.hybrid.controller import AdaptiveController
from isp.hybrid.tiling import run_tiled
from isp.learned.inputs import liteisp_input, validated_bayer
from isp.types import InputError, PipelineResult, RawImage

TOGGLES = {"colour_head", "sharpen", "ccm", "awb", "noise_map", "raw_denoise"}


def _ms(t0: float) -> float:
    return (perf_counter() - t0) * 1000


class HybridISP:
    """The P2 pipeline. `backend` is a student .pt or .onnx; `teacher` an optional larger one.

    ablation: optional dict for scripts/ablation.py. Keys: colour_head=False (identity colour matrix,
    .pt only), sharpen=False, ccm=False, awb=<mode name>, noise_map=False (zero sigma channel),
    raw_denoise=True (classical RAW denoiser before the network, off by default because the
    network denoises).
    """

    name = "hybrid"

    def __init__(self, backend: str, ablation: dict | None = None, controller: AdaptiveController | None = None,
                 teacher: str | None = None, threads: int | None = None) -> None:
        self.toggles = dict(ablation or {})
        unknown = set(self.toggles) - TOGGLES
        if unknown:
            raise InputError(f"Unknown ablation toggle(s): {', '.join(sorted(unknown))}.")
        self.controller = controller or AdaptiveController()
        head = self.toggles.get("colour_head", True)
        self.student = Backend(backend, threads, head)
        self.teacher = Backend(teacher, threads, head) if teacher else None
        base = ClassicalConfig(awb=self.toggles.get("awb", "fusion"), ccm=self.toggles.get("ccm", True))
        # fast_stats: noise and AWB statistics from ~64k samples; measured error is below sampling noise
        self._fe_cfg = replace(base, fast_stats=True)
        self._classical_cfg = base

    def run(self, raw: RawImage) -> PipelineResult:
        t0 = perf_counter()
        raw = RawImage(validated_bayer(raw), raw.wb_gains, raw.ccm, raw.meta)
        fe_t: dict[str, float] = {}
        rgb, stages, meta = front_end(raw, bool(self.toggles.get("raw_denoise")), self._fe_cfg, fe_t)
        timings = {f"fe_{k}": v for k, v in fe_t.items()}
        timings["front_end"] = _ms(t0)
        decision = self.controller.decide(meta["noise_a"], meta["noise_b"], meta["exposure"],
                                          rgb.shape[0] * rgb.shape[1] / 1e6)
        info = {**meta, "decision": decision, "fallback": False, "tiles": 0}
        net = self._backend_for(decision["model"])
        if net is None:
            return self._classical(raw, info, decision["denoise_strength"], t0)
        out, reason = self._network(net, rgb, meta, decision["tiling"], timings, info)
        if reason:
            info["fallback_reason"] = reason
            return self._classical(raw, {**info, "fallback": True, "fallback_from": info["model"]}, 1.0, t0)
        stages["network"] = np.clip(out, 0, 1)
        return self._finish(out, stages, timings, info, decision, t0)

    def _backend_for(self, choice: str) -> Backend | None:
        """None means the classical path. A teacher is used only if one was supplied."""
        if choice == "classical":
            return None
        return self.teacher if choice == "teacher" and self.teacher else self.student

    def _network(self, net, rgb, meta, tiling, timings, info):
        """Run the network (tiled if asked). Returns (rgb or None, failure reason or None)."""
        info["model"] = "teacher" if net is self.teacher else "student"
        t = perf_counter()
        x = liteisp_input(rgb, meta["noise_a"], meta["noise_b"], meta["exposure"])[None].numpy()
        if self.toggles.get("noise_map") is False:
            x[:, 3] = 0.0
        timings["input_prep"] = _ms(t)
        t = perf_counter()
        try:
            y, info["tiles"] = (run_tiled(x, net, tiling["tile"], tiling["overlap"]) if tiling["enabled"]
                                else (net(x), 1))
        except Exception as exc:  # a failing runtime must degrade to classical, not crash the app (F-HY-3)
            return None, f"network error: {type(exc).__name__}: {exc}"
        timings["network"] = _ms(t)
        y = np.asarray(y, np.float32).transpose(1, 2, 0)
        if y.shape != rgb.shape:
            return None, f"unexpected output shape {y.shape}"
        return (y, None) if np.isfinite(y).all() else (None, "non-finite network output")

    def _finish(self, out, stages, timings, info, decision, t0) -> PipelineResult:
        t = perf_counter()
        if self.toggles.get("sharpen") is not False:
            out = noise_aware_unsharp(np.clip(out, 0, 1), decision["noise_sigma"], decision["sharpen_amount"])
        timings["sharpen"] = _ms(t)
        out = np.clip(out, 0, 1).astype(np.float32)
        stages["output"] = out
        timings["total"] = _ms(t0)
        return PipelineResult(rgb=out, stages=stages, timings_ms=timings, info=info)

    def _classical(self, raw, info, strength, t0) -> PipelineResult:
        """Classical output: for clean frames (controller) or when the network output is unusable."""
        cfg = replace(self._classical_cfg, denoise_strength=strength)
        res = ClassicalISP(cfg).run(raw)
        res.timings_ms["total"] = _ms(t0)
        res.timings_ms.setdefault("network", 0.0)
        res.info.update({**info, "model": "classical"})
        return res
