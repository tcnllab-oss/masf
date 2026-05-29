# MASF: Measurement-Aware Score-based Filter

[![arXiv](https://img.shields.io/badge/arXiv-2604.02889-b31b1b.svg)](https://arxiv.org/abs/2604.02889)

This repo contains the official implementation of **[Rethinking Forward Processes for Score-Based Nonlinear Data Assimilation in High Dimensions](https://arxiv.org/abs/2604.02889)**.

> The code and documentation are currently being organized and will be updated progressively.

## 0. Table of Contents

<details>
<summary><b>Contents</b></summary>
  
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

</details>

## 1. Overview

**MASF (Measurement-Aware Score-based Filter)** is a score-based data assimilation method for estimating the hidden state of a dynamical system from noisy measurements.

Given a dynamical model and a measurement operator, MASF performs Bayesian filtering by propagating prior samples through the dynamics and then correcting them using the current measurement. It is designed for challenging settings where the state is high-dimensional and the measurements are sparse or nonlinear.

The main component of MASF is a **measurement-aware forward process**. Instead of perturbing states toward standard Gaussian noise, MASF perturbs states toward the measurement space. This allows the learned score model to reflect the relationship between the state and the measurement.

During the measurement-update step, MASF uses guided reverse-time sampling to transport samples from the measurement space back to the state space, producing posterior state estimates. More details are provided in the figure below.

<p align="center">
  <img src="figures/main_figure.png" width="95%">
</p>

### Key Features

- **Measurement-aware forward process**: MASF incorporates the measurement equation directly into the forward process, allowing the learned prior score to reflect the relationship between states and measurements.
- **Principled likelihood score**: For linear measurements, MASF provides a closed-form likelihood score at perturbed states. For nonlinear measurements, it uses an endpoint Gaussian approximation derived from a Markovian projection of the forward process.
- **Scalable high-dimensional filtering**: MASF is evaluated on Kolmogorov flow with state dimensions up to `O(10^5)`, including `256 × 256` and `512 × 512` resolutions.
- **Efficient amortized inference**: The model is pretrained once on simulated dynamics and then fine-tuned during assimilation, reducing online computational cost.
- **Robustness under sparse and nonlinear measurements**: MASF improves performance under grid masks, center masks, element-wise sigmoid measurements, and channel-coupled speed measurements.
  

## 2. Results

We evaluate MASF on two-dimensional Kolmogorov flow, a high-dimensional nonlinear fluid system governed by the incompressible Navier--Stokes equations. The experiments include both linear and nonlinear measurement operators:

- **Grid mask**: spatially sparse point-wise measurements
- **Center mask**: measurements outside a central missing region
- **Sigmoid**: element-wise nonlinear measurements
- **Speed**: channel-coupled nonlinear measurements based on local velocity magnitude

Across measurement settings, MASF achieves the best filtering accuracy while maintaining favorable wall-clock time. See the results tables and animations below for quantitative and qualitative results.

### 2.1. Results at 128 × 128 Resolution

<p align="center">
  <img src="figures/masf_results_128.gif" width="95%">
</p>


### 2.2. High-Resolution Results with Grid Mask Measurements

<p align="center">
  <img src="figures/masf_results_highres.gif" width="95%">
</p>

### 2.3. Temporal RMSE Evolution with Speed Measurement
<p align="center">
  <img src="figures/vorticity_pairs_with_live_rmse_compressed.gif" width="95%">
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

dataset: https://huggingface.co/datasets/eunbii1/Kolmogorov_flow

model ckpt: It will be avaliable soon.

## 5. Advanced Usage

### 5.1. Tuning

```bash
python -m swap.run_tuning \
    --until finetuning \
    --dynamic_type kolmogorov \
    --measurement_type grid_mask \
    --method_type ours \
    --dim 128
```

### 5.2. Post Evaluation

```bash
python -m swap.run_post_eval \
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
