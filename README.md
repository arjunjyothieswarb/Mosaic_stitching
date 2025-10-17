# Mosaic Stitching

This repository contains a simple image processing pipeline to stitch multiple images together into a single mosaic using feature feature matching and graph based optimization (GTSAM). 

<p align="center">
    <img src="output\gtsam_29imgs.png" alt="6Images_nogtsam" width="60%">
</p>
<p align="center">
    <em>29 Images using GTSAM</em>
</p>

## Features

The pipeline consists of the following steps.

- Image preprocessing (CLAHE, Gaussian blur)
- Feature detection and matching (SIFT feature extraction, BF matcher)
- Compute Affine Transform using the positive matches
- Pose graph construction with GTSAM's BetweenFactors for matched pairs
- Pose optimization using GTSAM (Gauss-Newton solver)
- Compose final mosaic from optimized poses

## Project layout

Files and directories you'll interact with:

- `src/` - main source code
  - `stitch_mosaic.py` - entry point: loads config, images, builds and optimizes the pose graph, then stitches the mosaic
  - `utils.py` - helper utilities (image loading, transforms, conversions)
  - `graph_utils.py` - functions to build/convert graph factors and transforms
  - `log_utils.py` - logging helpers
- `config/config.yaml` - main configuration (dataset selection, preprocessing, matching thresholds, etc.)
- `imgs/` - example image folders (datasets)
- `output/` - runtime output (stitched mosaics, debug images)
- `testYAML.py`, `testbool.py` - small tests / scripts

## Configuration

Most behavior is controlled via `config/config.yaml`. Important sections include:

- `Dataset` - select which dataset from `imgs/` to use
- `Preprocessing` - Gaussian blur kernel size and CLAHE parameters
- `FeatureMatching` - minimum match count and matching options

Open `config/config.yaml` to tune parameters (for example change dataset, or lower the minimum match count to accept weaker matches).

## Running the stitcher

From the repository root run the main script (PowerShell example):

```powershell
# Activate your virtual env first if needed
python .\src\stitch_mosaic.py
```

The script reads `config/config.yaml`, loads images from the configured dataset folder, runs preprocessing and feature matching, constructs a pose graph and runs GTSAM optimization. The final stitched image is displayed (and can be saved by editing the code in `stitch_mosaic.py`).

## Output

No GTSAM  | With GTSAM
:--------:|:--------:
<img src="output\nogtsam_6imgs.png" alt="6Images_nogtsam" width="100%">|<img src="output\gtsam_6imgs.png" alt="6Images_nogtsam" width="100%">
