# Viva preparation

Every member must be able to explain their own part, the model, the classical choices, the metrics and at least one failure case. Answer with what we measured. If a number is not in `results/`, say that it was not measured.

## Ownership

| Workstream | Owner | Must be able to walk through |
|---|---|---|
| W1 Classical ISP | TBD | `isp/classical/pipeline.py` from mosaic to sRGB |
| W2 Data and IO | TBD | `isp/io.py`, `isp/data/synthetic.py` |
| W3 Models and training | TBD | `isp/models/liteisp.py`, `scripts/train.py` |
| W4 Hybrid and optimisation | TBD | `isp/hybrid/pipeline.py`, `isp/hybrid/controller.py`, `isp/optim/` |
| W5 Evaluation | TBD | `isp/eval/metrics.py`, `scripts/evaluate.py` |
| W6 App and docs | TBD | `app/adapter.py`, `app/processing.py`, `Dockerfile` |

## Questions everyone should answer

**Why hybrid rather than end to end?** Linear stages such as white balance and colour correction are cheap and known, so spending network capacity on them is wasteful under a 200 k parameter budget. The network handles what classical filters do poorly. Evidence: learned against hybrid in the results table, TBD-from-results.

**How do you know it beats the baseline?** Same inputs, same conditions, fixed seeds, on Kodak24, which was never used for training or selection. Margin: TBD-from-results.

**What does each metric measure?** PSNR is pixel error on a log scale. SSIM compares local structure. DeltaE2000 is perceptual colour difference, where lower is better. LPIPS is distance in deep-feature space, where lower is better.

**Why might PSNR and LPIPS disagree?** A smooth result can score well on PSNR and look worse, because PSNR rewards averaging. Our example: TBD-from-results.

**Show a failure case.** Which image, what went wrong, why, and what would fix it: TBD-from-results.

**What was the hardest bug?** Each member prepares their own.

## W1 Classical

- Why demosaic with Malvar-He-Cutler rather than bilinear? It uses a gradient correction from the other colour channels, so edges show less zipper and false colour. Comparison: TBD-from-results.
- When does gray-world fail? On scenes dominated by one colour. How does the adaptive fusion handle that? TBD.
- Why denoise in a variance-stabilised domain? Shot noise grows with signal, so stabilising makes a single filter strength valid across brightness levels.
- What happens on an all-black or saturated frame? Point to the guard and its test.

## W2 Data

- What is unprocessing? Inverting tone, gamma, colour and white balance to synthesise plausible RAW from sRGB, then adding sensor noise.
- What does it not model? Fixed-pattern noise, lens shading, crosstalk.
- How do you know the splits are disjoint? Different source datasets, with Kodak24 reserved for test and McMaster for validation.
- How is JPEG input handled? Through `unprocess_jpeg`, which is approximate.

## W3 Models

- Why depthwise-separable convolutions? They need far fewer multiply-adds and parameters than standard convolutions at the same width.
- Why a residual output? The input is already close to the target, so the network learns a small correction.
- Why feed a noise-sigma map? One network can then adapt across noise levels. Evidence: ablation, TBD-from-results.
- Parameter count and how you verified it: TBD-from-results.

## W4 Hybrid and optimisation

- What does knowledge distillation add over training the student directly? Measured difference: TBD-from-results.
- Why static INT8 rather than dynamic? Convolution activations are quantised using calibration data, with no per-inference range computation. Accuracy cost: TBD-from-results.
- What was the calibration set? It must not be Kodak24. Name it: TBD.
- What happens if the network emits NaN? The result falls back to classical, and `info["fallback"]` is set and shown in the app.
- What decisions does the controller take, and from which estimates? TBD.

## W5 Evaluation

- Why this metric order? The brief requires it: PSNR, SSIM, DeltaE2000, LPIPS, parameters and size, latency and memory.
- How was latency measured? Median over repeated runs, on which hardware: TBD-from-results.
- Do the robustness curves degrade gracefully? Show them: TBD-from-results.

## W6 App and docs

- Why does the app import pipelines through one adapter? The interface can run before every pipeline lands, and anything missing is reported instead of faked.
- Are uploads stored? Gradio spools an upload to a temporary file, the app reads it into memory and deletes it at once, and results go to the browser inline. Files uploaded but never developed go in a cache sweep.
- How do you downscale a mosaic without destroying it? Split the four Bayer planes, area-resample each, and interleave them again, so the RGGB phase survives.
- Why does the Docker image not contain CUDA? Torch comes from the CPU wheel index, and `requirements.txt` has no torch dependency.
- How does the architecture diagram stay true? `scripts/gen_architecture.py` builds it from the modules and imports in the repo.
