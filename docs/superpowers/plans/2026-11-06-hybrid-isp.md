# Hybrid Smartphone ISP Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship the P2 hybrid ISP in `docs/PRD.md`: classical, learned and hybrid pipelines, an INT8 student, full evaluation and a public demo.

**Architecture:** Classical front end produces sRGB, LiteISPNet restores it, classical post sharpens. All pipelines implement `isp.types.Pipeline`.

**Tech Stack:** Python 3.11, PyTorch (CUDA, RTX 4050 6 GB), ONNX Runtime, OpenCV, rawpy, Gradio, Docker.

**Spec:** `docs/PRD.md`

## Global Constraints

- Deadline under 24 hours: synthetic training only unless time remains. Real-RAW fine-tuning is optional.
- Student: < 200 k params, <= 0.5 MB INT8. Latency < 1.5 s at 1 MP on 2 vCPU. App < 10 s.
- Kodak24 is test only, never used for training or selection. McMaster is validation only.
- Metric order: PSNR, SSIM, DeltaE2000, LPIPS (AlexNet), params and size, latency and memory.
- Report measured numbers only. Never hard-code, stub or estimate a result. Missed targets are reported as missed.
- All bad input raises `isp.types.InputError` with a friendly message.
- No secrets in the repo; the repo is public. `data/`, `runs/`, `*.pt` stay untracked. Final ONNX models go in `weights/`.
- Team ID stays `TXX` and owners stay TBD until the user supplies them. Never put `?` in a filename: Windows rejects it.
- Use the shared venv: `C:\Users\ddart\cv-hackathon\.venv\Scripts\python.exe`. Only W3 and W4 use the GPU.

## Review Focus

1. Odd-sized or tiny images: crop to even dimensions; reject below 32 px with `InputError`.
2. Grayscale, CMYK, RGBA, corrupt or X-Trans uploads: `InputError` or a clean conversion, never a traceback.
3. 12 MP+ uploads: auto-downscale for preview and stay under 10 s.
4. Saturated or all-black frames: AWB and noise estimation must not divide by zero or emit NaN.
5. NaN or Inf network output: hybrid falls back to classical and records `info["fallback"]=True`.

## Interfaces (exact)

| Owner | Module | Signature |
|---|---|---|
| lead | `isp/types.py` | `InputError`, `RawImage`, `PipelineResult`, `Pipeline` |
| W2 | `isp/io.py` | `load_input(source: str \| bytes, filename: str \| None = None) -> RawImage` |
| W2 | `isp/data/synthetic.py` | `SensorCondition(iso_a, iso_b, ev, illuminant, ccm)`; `sample_condition(rng) -> SensorCondition`; `synthesize(rgb, cond) -> tuple[RawImage, np.ndarray]`; `unprocess_jpeg(rgb) -> RawImage` |
| W2 | `isp/data/datasets.py` | `SyntheticRawDataset(split, crop, seed)`, yielding `(raw, target)` |
| W1 | `isp/classical/pipeline.py` | `ClassicalISP(config=None)` with `name="classical"`; `front_end(raw, denoise: bool) -> tuple[np.ndarray, dict, dict]` returning rgb, stages, `{"noise_a","noise_b","exposure"}` |
| W1 | `isp/classical/sharpen.py` | `noise_aware_unsharp(rgb, sigma, amount) -> np.ndarray` |
| W3 | `isp/models/liteisp.py` | `LiteISPNet(width, in_ch=4)`: `(B,4,H,W)` sRGB + sigma map to `(B,3,H,W)` |
| W3 | `isp/models/rawnet.py` | `RawNet(width)`: `(B,5,H/2,W/2)` packed Bayer + sigma to `(B,3,H,W)` |
| W3 | `isp/learned/pipeline.py` | `LearnedISP(ckpt)` with `name="learned"`; ckpt is a `.pt` or `.onnx` path, and deployment uses the `.onnx` in `weights/` |
| W3 | `scripts/train.py` | `--model {liteisp,rawnet} --teacher PATH --finetune PATH --epochs N` writes `runs/<name>/best.pt` |
| W4 | `isp/hybrid/pipeline.py` | `HybridISP(backend)` with `name="hybrid"`; backend is a `.pt` or `.onnx` path |
| W4 | `isp/hybrid/controller.py` | `AdaptiveController.decide(noise_a, noise_b, exposure, megapixels) -> dict` |
| W4 | `isp/optim/` | `quantize_int8(onnx_in, onnx_out, calib)`, `prune_sweep(ckpt, ratios)`, `export_onnx(ckpt, out)` |
| W4 | `scripts/benchmark.py` | writes `results/benchmark.json` |
| W5 | `isp/eval/metrics.py` | `psnr`, `ssim`, `delta_e2000`, `lpips_alex`, each `(pred, target) -> float` |
| W5 | `scripts/evaluate.py`, `robustness.py`, `ablation.py` | write `results/*.json`, `results/*.png`, `eval_P2_<TeamID>.md` via `--team-id` (default `TXX`) |
| W6 | `app/app.py` | Gradio app; `Dockerfile`, `Makefile`, `docs/*` |

## Waves

**Wave 1 (parallel):** W1 classical, W2 data and IO, W3 models and trainer, W5 metrics, W6 app shell and docs skeleton.
**Wave 2:** W3 trains teacher, student and RawNet. W4 builds hybrid and controller once W1 and W3 merge.
**Wave 3:** W4 KD, INT8, pruning, benchmark. W5 full evaluation, robustness, ablation, failure cases.
**Wave 4:** W6 deployment (needs the user's HF account) and final docs. Independent review of the whole repo.

Every task is test-first: write the failing test in `tests/test_<module>.py`, run it, implement, pass, commit. Each workstream lands on its own branch `w<N>-<topic>` and opens a PR into `main`.
