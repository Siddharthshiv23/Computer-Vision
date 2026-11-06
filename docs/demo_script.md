# Three-minute demo script

**Live URL:** TBD-from-deployment. **Presenter:** TBD. **Backup presenter:** TBD.

Rehearse with a stopwatch. Say only numbers that appear in `results/`, and replace every `TBD-from-results` below before the demo.

## Before you start

- Open the live URL on a laptop and on a phone. Free hosting sleeps, so load the page at least two minutes early.
- Keep three inputs ready: a noisy low-light RAW (`.dng`), an ordinary JPEG, and a Kodak-free image with a matching reference for the metrics table.
- Keep `eval_P2_T??.md` and the robustness plot open in another tab.

## Timeline

| Time | Do | Say |
|---|---|---|
| 0:00 to 0:20 | Show the app's first screen | A phone sensor records a noisy mosaic, not a picture. An ISP turns that into a photo. We built one that combines classical stages with a very small network, and we measured it against a classical baseline. |
| 0:20 to 1:10 | Ask a judge for an image, or use the low-light RAW. Upload, select Develop image | This is an image the system has not seen. All three pipelines run on the same input and appear as each one finishes. |
| 1:10 to 1:40 | Tap a detailed area to compare it at full size | Differences live at pixel level. Here is the same crop from all three. Point to noise and colour in the crop. |
| 1:40 to 2:05 | Open the Hybrid stages | Every stage is visible. The classical front end does the cheap linear work, the network restores, and the sharpening adapts to measured noise. This is the latency of each stage on this CPU. |
| 2:05 to 2:30 | Show the results table | On the Kodak24 test set, never used for training or selection, the hybrid scored TBD-from-results dB PSNR against TBD-from-results for classical, and DeltaE2000 of TBD-from-results against TBD-from-results. |
| 2:30 to 2:45 | Show the benchmark | The shipped model has TBD-from-results parameters and takes TBD-from-results MB as INT8. It runs 1 MP in TBD-from-results ms on 2 vCPUs. If a target was missed, say so here. |
| 2:45 to 3:00 | Show one failure case | It fails here, and this is why. Synthetic training does not capture every real-sensor effect. Name the case from the failure gallery: TBD-from-results. |

## If something goes wrong

| Problem | Response |
|---|---|
| Live URL is down or asleep | Run the local copy with `python scripts/run_app.py app` and say that you are doing so |
| The judge's file is rejected | The friendly error is part of the demo. Read it aloud, then use the backup JPEG |
| A pipeline shows as failed | Say what the panel says. Never describe a result that is not on the screen |
| The upload is slow on venue Wi-Fi | Use the phone's hotspot, or the smaller JPEG |

## Rules

- Do not quote a number that is not in `results/`.
- Metrics in the app are for a preview of at most 2 MP against the uploaded reference. Do not present them as the test-set results.
