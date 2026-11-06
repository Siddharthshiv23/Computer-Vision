# PRD — Hybrid Smartphone ISP (CV Hackathon 2026, Problem P2)

| | |
|---|---|
| **Problem ID** | P2 — Intelligent and Computationally Efficient Smartphone ISP |
| **Team ID** | `T??` *(to be filled)* |
| **Repo** | `Computer-Vision` |
| **Status** | v1.0 — drives all implementation; every feature below has an ID that code, issues and docs refer to |
| **Last updated** | 2026-11-06 |

---

## 1. Problem statement (from the brief)

> *How can you develop a learned smartphone ISP pipeline that transforms degraded RAW sensor data into high-quality RGB images while optimising colour accuracy, noise reduction, sharpness and computational efficiency under mobile-device resource constraints?*

The solution must be a **hybrid** ISP. A purely classical pipeline does not qualify, and neither does an off-the-shelf network. It must handle sensor noise, colour distortion, varying illumination, white-balance errors and artefacts. It must also be **measurably better than a conventional ISP baseline** and **optimised for mobile** (size, memory, latency).

## 2. Goals and non-goals

### Goals
1. **G1 — Quality:** beat the classical ISP baseline on PSNR, SSIM, ΔE2000 and LPIPS on a held-out test set, by a clear margin (target ≥ +3 dB PSNR, ≥ 30 % lower ΔE).
2. **G2 — Efficiency:** ship an optimised model with < 200 k params and ≤ 0.5 MB on disk (INT8). Target latency is < 1.5 s for a 1 MP image on a 2-vCPU CPU (similar to a free hosted tier, and a stand-in for a phone CPU).
3. **G3 — Robustness:** show graceful degradation and recovery across ISO / noise, exposure (0 to −3 EV) and colour-cast sweeps, with curves for all three pipelines.
4. **G4 — Live demo:** a public URL that works on phone and laptop on first click and answers in < 10 s, with loading and error states. It must accept a judge's unseen image (RAW or JPEG).
5. **G5 — Explainability:** every stage is visible (intermediate outputs), and every team member can explain their part.

### Non-goals
* Training large SOTA models (PyNET-full, 47 M params). They contradict G2, and there's no GPU budget for them.
* Burst/multi-frame HDR, video ISP, X-Trans/Quad-Bayer sensors.
* Bit-exact reproduction of any vendor's ISP "look".

## 3. Users and scenarios

| User | Scenario | Need |
|---|---|---|
| Judge (laptop/phone) | Uploads an unseen JPEG or RAW during the 3-min demo | Fast, clear result; side-by-side Classical / Learned / Hybrid; metrics |
| Judge (repo review) | Clones the repo | `make demo` / `docker run` works; README reproduces the numbers |
| Team member (viva) | Explains their module | Clear module ownership; design doc explains choices |

## 4. Rubric → strategy (what we optimise for)

The table in the brief lists Evaluation Metrics 25, Implementation 20, UI/UX 15, Documentation 15 and Demo/Viva 25, for a total of 100. The sentence above the table says *"Demo and viva 25, each of the other five 15"*. That doesn't add up, so we treat the **table** as authoritative and are confirming with the organisers.

| Criterion | Max | Top-band requirement | How we hit it | Feature IDs |
|---|---|---|---|---|
| Evaluation metrics | 25 | Average metrics in top 10 % of teams | Strong hybrid model and honest, complete metrics in the required order | F-DL-*, F-HY-*, F-EV-* |
| Implementation | 20 | Works on unseen inputs, proper train/val split, clean modular code, bad input handled | Disjoint train/val/test splits; `isp/` package; `InputError` everywhere; tests | F-IO-*, F-DATA-*, F-QA-* |
| UI/UX & deployment | 15 | Public URL, phone + laptop, ≤ 10 s, loading/error states, one-command run / Dockerfile, no secrets | Gradio app on Hugging Face Spaces (Docker); auto-downscale; `make app`; `.gitignore` for secrets | F-APP-* |
| Documentation | 15 | Architecture diagram matches code, design doc complete, README reproduces, limitations + ethics | `docs/architecture.md` (Mermaid generated from module names); `docs/design_document.md`; README commands | F-DOC-* |
| Demo & viva | 25 | 3-min demo on live URL with judge's input; every member explains part, model and classical choices, metrics, a failure case | Scripted demo (`docs/demo_script.md`); ownership table; failure gallery | F-DOC-4, F-DOC-5 |

**Deductions to avoid:** no live URL caps UI/UX at 0. No GitHub link caps Implementation at 0. No architecture diagram or design doc caps Documentation at 5. Late submission costs −5 per 15 minutes.

## 5. Requirements traceability

Each requirement from the brief maps to a feature, a module and the evidence that proves it.

### A. Classical computer vision

| Req (brief) | ID | Implementation | Evidence |
|---|---|---|---|
| Bayer RAW pre-processing & demosaicing | F-CV-1 | `isp/classical/bayer.py` (black/white level, CFA normalisation to RGGB, pack/unpack), `demosaic.py` (bilinear, **Malvar-He-Cutler**, OpenCV-EA) | Demosaic comparison table; unit tests |
| White-balance correction | F-CV-2 | `white_balance.py`: gray-world, shades-of-gray, gray-edge, white-patch, **adaptive median fusion** | AWB ablation; colour-cast sweep |
| Colour correction matrix | F-CV-3 | `color.py`: CCM from DNG metadata or average-camera default | ΔE with/without CCM |
| Denoising & edge-preserving filtering | F-CV-4 | `denoise.py`: Poisson-Gaussian noise estimation, VST + guided filter on RAW, luma/chroma guided, bilateral and NLM | Noise sweep; ablation |
| Gamma correction & tone mapping | F-CV-5 | `tone.py`: auto-exposure, sRGB/2.2 gamma, S-curve, Reinhard, CLAHE | Stage visualisation |
| Sharpening & colour-space transforms | F-CV-6 | `sharpen.py` (noise-aware unsharp mask on Y), YCbCr / Lab / XYZ in `color.py` | Ablation (sharpen on/off) |

### B. Deep learning

| Req | ID | Implementation | Evidence |
|---|---|---|---|
| Learned ISP architecture (PyNET / U-Net / RAW→RGB) | F-DL-1 | `isp/models/`: **LiteISPNet**, a depthwise-separable U-Net with a global colour-transform head. The learned-only baseline (RAW→RGB, PixelShuffle) is a PyNET-style end-to-end network | Model card; params |
| Adaptation / fine-tuning on a dataset | F-DL-2 | Pre-train on synthetic RAW (unprocessing of 294 HR images), then fine-tune on real RAW (SIDD / Zurich RAW-to-RGB) via `scripts/train.py --finetune` | Training logs; before/after table |
| Learn restoration + colour transform | F-DL-3 | Residual restoration branch + learned 3×3 colour matrix and bias | Ablation (no colour head) |
| Robust to noise & illumination | F-DL-4 | Noise-map conditioning; training over 0 to −3 EV and wide noise / illuminant ranges | Robustness curves |

### C. Hybrid ISP architecture

| Req | ID | Implementation | Evidence |
|---|---|---|---|
| Integrate classical stages with learned components | F-HY-1 | `isp/hybrid/pipeline.py`: classical front end (normalise → gain → noise estimation → demosaic → AWB → CCM → AE → gamma/tone), then LiteISPNet residual restoration and colour refinement, then noise-aware sharpening | Architecture diagram |
| Optimised processing sequence | F-HY-2 | Cheap linear stages run in classical code. The network runs once at full resolution; large images are processed in tiles | Latency breakdown per stage |
| Avoid colour distortion & detail loss | F-HY-3 | ΔE-aware loss (L1 + Lab colour loss + gradient loss); fallback to the classical output if the network output is invalid | ΔE; failure analysis |
| Compare traditional / learned / hybrid | F-HY-4 | `scripts/evaluate.py` runs all three on the same inputs | `results/` table |

### Advanced challenge requirements

| # | Req | ID | Implementation |
|---|---|---|---|
| 1 | Adaptive pipeline for varying noise & illumination | F-AD-1 | `AdaptiveController`: estimated (a, b) noise and exposure set denoise strength, sharpening, the network's noise map and model choice (student / teacher / classical fallback) |
| 2 | RAW-to-RGB reconstruction from Bayer | F-AD-2 | Native DNG/RAW support (rawpy), `.npy` and 16-bit PNG Bayer |
| 3 | ≥ 1 optimisation (quantisation / pruning / KD / reduced precision) | F-OPT-1..3 | **Knowledge distillation** (teacher → student), **INT8 static PTQ** (ONNX Runtime), **L1 pruning** sweep, FP16 |
| 4 | Measure size, memory, latency | F-OPT-4 | `scripts/benchmark.py`: params, MB on disk, peak RSS, median latency at 512² / 1 MP / 12 MP-tiled |
| 5 | Degradation / recovery under challenging conditions | F-EV-3 | `scripts/robustness.py`: ISO × EV × colour-cast sweeps, curves for each pipeline |
| 6 | Compare with conventional ISP baseline | F-EV-1 | Classical ISP is the baseline; deltas reported |

### Evaluation metrics (reported in this order in `eval_P2_<TeamID>.md`)
1. PSNR  2. SSIM  3. ΔE2000 (colour accuracy)  4. LPIPS (AlexNet)  5. Parameters and model size  6. Inference latency and memory

## 6. System design (summary; details in `docs/design_document.md`)

```
RAW (DNG/npy/png16)  ─┐
JPEG ─► unprocess ────┤
                      ▼
 [Classical front end]  normalise → digital gain → noise (a,b) estimation → (RAW VST denoise*) →
                        Malvar demosaic → adaptive AWB → CCM → auto-exposure → gamma + S-curve
                      ▼                                           ▲
 [AdaptiveController] ── picks: denoise strength, sharpen amount, model (student/teacher), tiling
                      ▼
 [LiteISPNet]          input: classical sRGB + noise-σ map → residual restoration + global colour matrix
                      ▼
 [Classical post]      noise-aware unsharp mask → clip → sRGB out   (+ fallback to classical on failure)
```
`*` used in classical-only mode. In hybrid mode the network does the denoising, which the ablation justifies.

## 7. Data

| Split | Source | Count | Use |
|---|---|---|---|
| train | Urban100 + BSD100 + Set14 + Sun-Hays80 (HR) → synthetic RAW | 294 images, random crops and random sensor conditions per iteration | Training |
| val | McMaster | 18 | Checkpoint / hyper-parameter selection only |
| test | Kodak24 | 24 | **Never used** for training or selection. Fixed seeds and fixed conditions (`configs/test_conditions.yaml`) |
| real (optional) | SIDD-Medium RAW, Zurich RAW-to-RGB, Modular Neural ISP data (datasheet) | — | Fine-tuning / generalisation on real sensors (needs GPU and a manual download) |

The synthetic sensor model is unprocessing (Brooks et al., 2019). It uses random real-camera CCMs, random illuminants, 0 to −3 EV exposure, calibrated Poisson-Gaussian noise and 12-bit quantisation.

## 8. Success metrics and targets

| Metric | Classical baseline | Target (hybrid student INT8) |
|---|---|---|
| PSNR (Kodak24, standard conditions) | measured | ≥ baseline + 3 dB |
| SSIM | measured | ≥ baseline + 0.05 |
| ΔE2000 | measured | ≤ 0.7 × baseline |
| LPIPS | measured | ≤ 0.7 × baseline |
| Params / size | 0 | < 200 k / ≤ 0.5 MB |
| Latency 1 MP (CPU) | measured | < 1.5 s end-to-end |
| App response | — | < 10 s for images up to 12 MP (auto-downscale preview) |

## 9. Milestones (hackathon timeline)

| # | Milestone | Exit criterion |
|---|---|---|
| M0 | Repo, PRD, data download | `scripts/download_data.py` works; PRD merged |
| M1 | Classical ISP baseline | Runs end-to-end on DNG and synthetic RAW; stage visualisation; tests pass |
| M2 | Learned-only and hybrid models trained (CPU-sized) | Val PSNR hybrid > classical |
| M3 | Optimisation | KD student, INT8 ONNX, pruning sweep, benchmark table |
| M4 | Evaluation | Test table, robustness curves, ablation, ≥ 3 failure cases, `eval_P2_T??.md` |
| M5 | App and deployment | Public URL on HF Spaces; Dockerfile; phone check |
| M6 | Docs and demo | Design doc, architecture diagram, README reproduction, demo script, viva prep |

## 10. Team workstreams (assign one owner per row; each owner presents it in the viva)

| Workstream | Directory | Owner |
|---|---|---|
| W1 Classical ISP (demosaic, AWB, CCM, denoise, tone, sharpen) | `isp/classical/` | _TBD_ |
| W2 Data & synthetic sensor model, real-RAW loaders | `isp/data/`, `scripts/download_data.py` | _TBD_ |
| W3 Learned models, training, KD | `isp/models/`, `isp/learned/`, `scripts/train.py` | _TBD_ |
| W4 Hybrid pipeline, adaptive controller, optimisation (INT8/pruning) | `isp/hybrid/`, `isp/optim/` | _TBD_ |
| W5 Evaluation, robustness, ablation, failure analysis | `isp/eval/`, `scripts/evaluate.py` … | _TBD_ |
| W6 App, Docker, deployment, docs | `app/`, `Dockerfile`, `docs/` | _TBD_ |

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No GPU, so training is small | Lightweight models by design (which G2 requires anyway); synthetic data; Colab/Kaggle notebook for real-data fine-tuning |
| Synthetic-to-real gap | Real-RAW fine-tuning path, DNG demo inputs, honest limitations section |
| LPIPS weights download blocked in some environments | LPIPS computed in CI/Colab; app shows PSNR/SSIM/ΔE live |
| Hosted CPU too slow | INT8 ONNX student, auto-downscale to ≤ 2 MP for interactive preview, tiling |
| Judge uploads odd input (huge, grayscale, CMYK, corrupt, X-Trans) | `InputError` with a friendly message; tested in `tests/test_io.py` |

## 12. Ethics and limitations (expanded in the design doc)
* Datasets are public research benchmarks; no faces or personal data are collected. Uploaded images are processed in memory and not stored.
* Synthetic RAW does not capture every real-sensor effect (fixed-pattern noise, lens shading, crosstalk). We report real-RAW results separately and do not over-claim.
* Colour "accuracy" is measured against a reference rendering, which is itself an aesthetic choice.

## 13. Open questions (need answers from the team)
1. Team ID and member USNs (first line of the eval file), plus member names for the ownership table.
2. Submission deadline (date and time).
3. Did the organisers provide a **design-document template**? The rubric says "every template section".
4. Is there a GPU (Colab/Kaggle/lab) for real-RAW fine-tuning?
5. Is there a Hugging Face account for hosting (Spaces)? The repo must be **public** at submission.
