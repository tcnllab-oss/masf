
# MASF: Measurement-Aware Score-based Filter

[![arXiv](https://img.shields.io/badge/arXiv-2604.02889-b31b1b.svg)](https://arxiv.org/abs/2604.02889)

This repo contains the official implementation of **[Rethinking Forward Processes for Score-Based Nonlinear Data Assimilation in High Dimensions](https://arxiv.org/abs/2604.02889)**.


* The code and documentation are currently being organized and will be updated progressively.


## 1. Overview

<p align="center">
  <img src="figures/main_figure.png" width="95%">
</p>



## 2. Results

### 2.1. Results at $128^2$ Resolution

<p align="center">
  <img src="figures/masf_results_128.gif" width="95%">
</p>

### 2.2. High-Resolution Results with Grid Mask Measurements

<p align="center">
  <img src="figures/masf_results_highres.gif" width="95%">
</p>

## 3. Code Implementation

### 3.1. Code structure

```text
masf/
├── configs/                        # Configuration files
├── dynamics/                       # Dynamical system implementations: Kolmogorov flow
├── measurements/                   # Measurement operators: Grid mask, Center mask, Sigmoid, Speed
├── methods/                        # Filtering methods: EnKF, LETKF, SF, SSLS, MASF
├── models/                         # Neural network models: UNet, dual-UNet 
├── main.py                         # Run a single method
├── swap/                           # Experiment pipeline for tuning and post evaluation
│   ├── run_tuning.py               # Run hyperparameter tuning phases
│   ├── run_post_eval.py            # Run post-evaluation and sensitivity analysis
│   ├── configs/                    # Pipeline, tuning, and post-evaluation configs
│   └── src/                        # Source code for tuning, post evaluation, and reports
├── utils/                          # Utilities for training, loading, filtering, and saving
├──README.md
└── requirements.txt
```


