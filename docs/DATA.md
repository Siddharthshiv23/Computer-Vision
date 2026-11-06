# Data

`python scripts/download_data.py` fetches 336 images (171 MB) into `<root>/<dataset>/`, for example `data/kodak24/`, which is where the evaluation scripts look by default. `isp/data/splits.py` says which dataset belongs to which split. The root is `--root`, else the environment variable `ISP_DATA_ROOT`, else the shared hackathon folder, else `data/` in the checkout. `data/` is gitignored: no dataset image is ever committed or redistributed by this repo.

Every file is pinned twice: the URLs name a fixed commit, and `configs/data_manifest.json` holds its SHA-256. The script rejects a file with the wrong hash or one that does not decode as an image, and exits 1 if a dataset ends up short. It never swaps in another dataset.

## Sources and licences

| Split | Dataset | Count | Downloaded from | Original source | Terms |
|---|---|---|---|---|---|
| train | Urban100 | 100 | [jbhuang0604/SelfExSR](https://github.com/jbhuang0604/SelfExSR) `data/Urban100/image_SRF_2/*_HR.png` | Huang et al., CVPR 2015 (images gathered from Flickr) | No licence stated for the images. Research benchmark use. |
| train | BSD100 | 100 | same repo, `data/BSD100/image_SRF_2/*_HR.png` | Martin et al., ICCV 2001 (test half of BSDS300) | [Berkeley's page](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) allows non-commercial research and educational use, and asks for a citation. |
| train | Set14 | 14 | same repo, `data/Set14/image_SRF_2/*_HR.png` | Zeyde et al., 2010 | No licence stated. Research benchmark use. |
| train | Sun-Hays80 | 80 | same repo, `data/Sun-Hays80/image_SRF_8/*_HR.png` | Sun and Hays, ICCP 2012 | No licence stated. Research benchmark use. |
| val | McMaster | 18 | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) `testsets/McMaster/*.tif` | Zhang et al., J. Electronic Imaging 2011 | No licence stated for the images. SwinIR's Apache-2.0 covers its code. Research benchmark use. |
| test | Kodak24 | 24 | [r0k.us/graphics/kodak](https://r0k.us/graphics/kodak/) | Eastman Kodak | The host understands them to be released for unrestricted usage. |

SelfExSR has no licence file, and its README asks users to cite the paper. The original McMaster page (`www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm`) and the Sun-Hays page (`cs.brown.edu/~lbsun/SRproj2012`) were unreachable on 2026-11-06, so they were not used and the terms above could not be confirmed there.

"No licence stated" means we found none, not that the images are free to reuse. We only use them locally for research and do not redistribute them. Anyone reusing this work commercially should check each source first.

## What is in the images

- Sizes run from 250 x 276 to 1024 x 1024 in train, 500 x 500 in val, and 768 x 512 in test (either orientation).
- 9 of the 294 training images have no colour (R = G = B), and one of those is stored as single-channel. They are kept, because the PRD fixes the set at 294, and are expanded to three channels on load.
- A crop larger than an image is filled by reflecting its borders.

## Split isolation

Each dataset belongs to one split, and no file's SHA-256 appears in two splits (`tests/test_datasets.py`). A one-off search for near-duplicates across splits, over whole images and sampled patches, found none. The closest whole-image pair (`bsd100/img_053` and `kodim20`, correlation 0.848) only shares a sky-over-ground layout: one shows giraffes, the other an aircraft. That search sampled patches, so it is evidence rather than proof.

## Synthetic sensor model

`isp/data/synthetic.py` follows Brooks et al. 2019, with three deliberate changes, listed at the top of that file. `sample_condition` deliberately covers every level in `configs/test_conditions.yaml`, including the neutral `[1, 1, 1]` illuminant, so the test conditions are not out of range for a trained model. For other workstreams:

- `synthesize` leaves `RawImage.wb_gains` as `None`, because estimating white balance is part of the task. `ccm` is set, as real RAW files carry one. The true parameters are in `meta["truth"]`, for evaluation only: pipelines must not read them.
- `unprocess_jpeg` does the opposite. It adds no noise and reports the gains and matrix it assumed, so a pipeline that honours them gets the photo's own colours back.
- The tone curve inverted here is Brooks' smoothstep followed by gamma 2.2. A pipeline that applies a different curve will not reproduce the target exactly.
- `target` is always the clean, normally exposed image. It does not change with `ev`, so at -3 EV a pipeline is scored on recovering the exposure.
- `illuminant` multiplies the camera RGB channels before the mosaic. `ccm` is camera RGB to linear sRGB, as in `RawImage.ccm`.
- Noise comes from `SensorCondition.seed`. If that is `None`, the seed is drawn from numpy's global generator, so `np.random.seed()` fixes it too. `sample_condition` draws a new seed each time.

`SyntheticRawDataset("test", None, seed, condition="ev_-3")` picks a named level from `configs/test_conditions.yaml`. Its `seed` argument is ignored, so the test set cannot be rerolled. The reported numbers come from `isp/eval/harness.py`, which reads the same file but derives its own per-image noise seeds, so the two give the same conditions with different noise.

## Input limits (`isp/io.py`)

- Below 32 pixels on a side, or above 120 MP: refused with `InputError`.
- Above 24 MP: reduced by a whole-number factor, recorded in `meta["downscaled_from"]`. Bayer data is binned per colour, so RGGB is kept. The app's smaller preview size is the app's job.
- Integer Bayer data with no stated depth is scaled by the smallest of 10, 12, 14 or 16 bits that fits its peak value, so a very dark 16-bit frame can come out brighter than it should. `.npy` and 16-bit PNG carry no black level, so it is assumed to be 0.
- Name a `.npy` or 16-bit PNG `*_bggr.*` (or `grbg`, `gbrg`) to state its layout. Otherwise RGGB is assumed and `meta["cfa_assumed"]` is True.

## Citations

- Huang, Singh, Ahuja. Single Image Super-Resolution from Transformed Self-Exemplars. CVPR 2015.
- Martin, Fowlkes, Tal, Malik. A Database of Human Segmented Natural Images. ICCV 2001.
- Zeyde, Elad, Protter. On Single Image Scale-Up Using Sparse-Representations. Curves and Surfaces 2010.
- Sun, Hays. Super-resolution from Internet-scale Scene Matching. ICCP 2012.
- Zhang, Wu, Buades, Li. Color Demosaicking by Local Directional Interpolation and Nonlocal Adaptive Thresholding. J. Electronic Imaging 2011.
- Brooks, Mildenhall, Xue, Chen, Sharlet, Barron. Unprocessing Images for Learned Raw Denoising. CVPR 2019. The camera matrices and noise line in `isp/data/synthetic.py` come from its [Apache-2.0 reference code](https://github.com/google-research/google-research/tree/master/unprocessing).
