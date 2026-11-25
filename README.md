# EPI-GS

EPI-GS is a dual-branch decoder for arbitrary-scale light field spatial super-resolution. It combines a spatial implicit image function with epipolar Gaussian sampling and disparity-guided frequency enhancement.

## Layout

- model_*.py: six LF encoder adapters from the SEIIF codebase.
- EPI_GS_Decoder.py: EPI-GS decoder and LR photometric warping loss.
- train.py, test.py: training and checkpoint evaluation.
- saved_models/: EPI-GS checkpoints grouped by encoder (2x, 4x, and arbitrary scale).
- reports/: benchmark JSON with per-dataset PSNR/SSIM and equal-weight dataset averages.

## Installation

The verified environment uses Python 3.10, PyTorch 2.3.1, torchvision 0.18.1, and CUDA 11.8. Install the remaining dependencies with `pip install -r requirements.txt`.

## Datasets

Following [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR), we use five datasets, including EPFL, HCInew, HCIold, INRIA and STFgantry for training and testing. Please download the datasets in the official repository of [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR).


## Data and evaluation

Prepare BasicLFSR-style HDF5 data for the central 5x5 views of EPFL, HCInew, HCIold, INRIA, and Stanford Gantry. Place each dataset's scene files in its own subdirectory under `datasets/test/SR_5x5_<scale>x/`, or pass its parent root with `--test_root`. Use the benchmark split and bicubic LR inputs. Metrics are computed on Y using the included evaluator. The default evaluation uses 64-pixel LR patches and stride 32. The verified test split is listed in `reports/test_scenes.json`. On the lab server, pass `--test_root ../EPI-GS/datasets/test` to reuse the prepared inputs.

    python test.py --encoder DistgSSR --variant Arbitrary --scales 2 3 4 6 8 --report reports/distg.json
    python test.py --encoder DistgSSR --variant 4x --report reports/distg_4x.json
    python test.py --encoder DistgSSR --variant 2x --check-only

## Acknowledgement

Our work and implementations are inspired and based on the following projects:

- [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)
- [SEIIF](https://github.com/Congrx/SEIIF)
- [LIIF](https://github.com/yinboc/liif)

We sincerely thank the authors for sharing their code and amazing research work!
