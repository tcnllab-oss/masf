# MASF: Measurement-Aware Score-based Filter

[![arXiv](https://img.shields.io/badge/arXiv-2604.02889-b31b1b.svg)](https://arxiv.org/abs/2604.02889)

This repo contains the official implementation of **[Rethinking Forward Processes for Score-Based Nonlinear Data Assimilation in High Dimensions](https://arxiv.org/abs/2604.02889)**.

> The code and documentation are currently being organized and will be updated progressively.

## Table of Contents

- [1. Overview](#1-overview)
- [2. Results](#2-results)
  - [2.1. Results at 128 × 128 Resolution](#21-results-at-128--128-resolution)
  - [2.2. High-Resolution Results with Grid Mask Measurements](#22-high-resolution-results-with-grid-mask-measurements)
- [3. Code Implementation](#3-code-implementation)
  - [3.1. Code Structure](#31-code-structure)
  - [3.2. Installation](#32-installation)
  - [3.3. Single Experiment](#33-single-experiment)
- [4. Tutorials](#4-tutorials)
- [5. Advanced Usage](#5-advanced-usage)
  - [5.1. Tuning](#51-tuning)
  - [5.2. Post Evaluation](#52-post-evaluation)
- [6. Citation](#6-citation)
- [7. Contact](#7-contact)

## 1. Overview

<p align="center">
  <img src="figures/main_figure.png" width="95%">
</p>

## 2. Results

### 2.1. Results at 128 × 128 Resolution

<p align="center">
  <img src="figures/masf_results_128.gif" width="95%">
</p>

### 2.2. High-Resolution Results with Grid Mask Measurements

<p align="center">
  <img src="figures/masf_results_highres.gif" width="95%">
</p>

## 3. Code Implementation

### 3.1. Code Structure

```text
masf/
├── configs/
├── dynamics/                       # Kolmogorov flow
├── measurements/                   # Grid mask, Center mask, Sigmoid, Speed
├── methods/                        # EnKF, LETKF, SF, SSLS, MASF
├── models/                         # UNet, dual-UNet
├── main.py                         # Run a single method
├── swap/
│   ├── run_tuning.py               # Run hyperparameter tuning phases
│   ├── run_post_eval.py            # Run post-evaluation and sensitivity analysis
│   ├── configs/
│   └── src/                        # Source code for tuning, post evaluation, and reports
├── utils/
├── README.md
└── requirements.txt
```

### 3.2. Installation

```bash
git clone git@github.com:tcnllab-oss/masf.git
cd masf
```

```bash
conda create -n masf python=3.10
conda activate masf
```

```bash
pip install -r requirements.txt
```

### 3.3. Single Experiment

```bash
python main.py \
    --dynamic_type kolmogorov_128 \
    --measurement_type grid_mask \
    --method_type ours \
    --seed 0 \
    --exp grid128_masf_seed0
```

## 4. Tutorials

A Colab tutorial will be available soon.

<!--
[Open in Colab](COLAB_LINK)
-->

## 5. Advanced Usage

### 5.1. Tuning

```bash
python swap/run_tuning.py \
    --phase finetuning \
    --dynamic_type kolmogorov \
    --measurement_type grid_mask \
    --method_type ours \
    --dim 128
```

### 5.2. Post Evaluation

```bash
python swap/run_post_eval.py \
    --phase num_sample \
    --root_dir outputs/kolmogorov_128/grid_mask/ours
```

## 6. Citation

```bibtex
@article{yoon2026masf,
  title={Rethinking Forward Processes for Score-Based Nonlinear Data Assimilation in High Dimensions},
  author={Yoon, Eunbi and Chang, Won and Kim, Donghan and Kim, Dae Wook},
  journal={arXiv preprint},
  year={2026}
}
```

## 7. Contact

For questions, please contact Eunbi Yoon at `eunbiyoon6286@kaist.ac.kr`.
