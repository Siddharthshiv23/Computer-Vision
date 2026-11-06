---
title: Hybrid Smartphone ISP
sdk: docker
app_port: 7860
---

# Hybrid smartphone ISP

CV Hackathon 2026, Problem P2. A camera pipeline that turns noisy Bayer RAW into sRGB by combining classical stages with a small learned restoration network, compared against a classical baseline and an end-to-end network.

**Team ID:** `TXX` | **Live demo:** TBD-from-deployment | **Spec:** [docs/PRD.md](docs/PRD.md)

## Results

Kodak24, standard conditions. Metric order follows the brief. Values come from `results/` and are filled in once measured.

| Pipeline | PSNR | SSIM | DeltaE2000 | LPIPS | Params and size | Latency and memory |
|---|---|---|---|---|---|---|
| Classical | TBD-from-results | TBD-from-results | TBD-from-results | TBD-from-results | none | TBD-from-results |
| Learned | TBD-from-results | TBD-from-results | TBD-from-results | TBD-from-results | TBD-from-results | TBD-from-results |
| Hybrid | TBD-from-results | TBD-from-results | TBD-from-results | TBD-from-results | TBD-from-results | TBD-from-results |

## Run the app

Python 3.11. Without a GPU:

```
pip install -r requirements-torch-cpu.txt
pip install -r requirements-dev.txt
python scripts/run_app.py app
```

Open http://localhost:7860. Pipelines that have not landed, or whose weights are missing, say so in the interface; the app never substitutes a result.

| With make | Without make | What it does |
|---|---|---|
| `make app` | `python scripts/run_app.py app` | Starts the app on port 7860 |
| `make demo INPUT=photo.jpg` | `python scripts/run_app.py demo --input photo.jpg` | Runs every available pipeline on one file and writes the outputs to `runs/demo/` |
| `make test` | `python scripts/run_app.py test` | Runs the test suite |

With Docker:

```
docker build -t hybrid-isp .
docker run --rm -p 7860:7860 hybrid-isp
```

The image installs CPU-only torch and contains no secrets. ONNX weights are read from `weights/`; set `ISP_LEARNED_WEIGHTS` or `ISP_HYBRID_WEIGHTS` to use other paths.

## Reproduce the results

These commands are the interfaces in the [implementation plan](docs/superpowers/plans/2026-11-06-hybrid-isp.md). Each script lands with its workstream, so check that it exists at your commit. Training needs a CUDA machine.

```
pip install -r requirements-train.txt
python scripts/download_data.py
python scripts/train.py --model liteisp --epochs N
python scripts/train.py --model rawnet --epochs N
python scripts/train.py --model liteisp --teacher runs/<teacher>/best.pt --epochs N
python scripts/evaluate.py --team-id TXX
python scripts/robustness.py
python scripts/ablation.py
python scripts/benchmark.py
```

`evaluate.py` writes `results/*.json`, `results/*.png` and `eval_P2_<TeamID>.md`. Epoch counts and run names used for the reported numbers: TBD-from-results.

## Repository layout

| Path | Contents |
|---|---|
| `isp/` | Pipelines, models, data, evaluation; shared contract in `isp/types.py` |
| `app/` | Gradio app: `adapter.py` finds pipelines, `processing.py` handles input, `render.py` builds the results |
| `scripts/` | Entry points, including `run_app.py` and `gen_architecture.py` |
| `docs/` | [Design document](docs/design_document.md), [architecture](docs/architecture.md), [demo script](docs/demo_script.md), [viva prep](docs/viva_prep.md) |
| `weights/` | Final ONNX models |
| `results/` | Measured results |

## Limitations and ethics

- Synthetic RAW does not capture every real-sensor effect, such as fixed-pattern noise, lens shading and crosstalk. Real-RAW results are reported separately and are not over-claimed.
- Colour accuracy is measured against a reference rendering, which is itself an aesthetic choice.
- Datasets are public research benchmarks, and no faces or personal data are collected.
- Uploads are not kept. Gradio spools each upload to a temporary file, which the app deletes as soon as it has read it. A file that is uploaded but never developed is removed by a cache sweep within about 20 minutes.

More in the [design document](docs/design_document.md), sections 11 and 12.
