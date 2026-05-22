#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

SEEDS="42 43 44 45 46"
OUT_DIR="${PROJECT_ROOT}/swap_results"
BASE_DIR="${OUT_DIR}/letkf"


echo "============================================================"
echo "[POST-EVAL] LETKF saved results"
echo "============================================================"

###############################################################################
# 1. Grid mask / 128 / stride 10
###############################################################################

echo "============================================================"
echo "[1/6] LETKF / Grid mask / 128 / stride 10"
echo "============================================================"

python -m swap.run_post_eval \
  --phase num_sample \
  --scratch \
  --root_dir "${BASE_DIR}/kolmogorov_128/grid_mask/stride_10" \
  --seeds $SEEDS \
  --cleanup

python -m swap.run_post_eval \
  --phase measurement_sensitivity \
  --root_dir "${BASE_DIR}/kolmogorov_128/grid_mask/stride_10" \
  --seeds $SEEDS \
  --cleanup

python -m swap.run_post_eval \
  --phase temporal_sensitivity \
  --root_dir "${BASE_DIR}/kolmogorov_128/grid_mask/stride_10" \
  --seeds $SEEDS \
  --cleanup


###############################################################################
# 2. Center mask / 128 / hole 0.5
###############################################################################

echo "============================================================"
echo "[2/6] LETKF / Center mask / 128 / hole 0.5"
echo "============================================================"

python -m swap.run_post_eval \
  --phase num_sample \
  --root_dir "${BASE_DIR}/kolmogorov_128/center_mask/hole_0.5" \
  --seeds $SEEDS \
  --cleanup


###############################################################################
# 3. Nonlinear sigmoid / alpha 2
###############################################################################

echo "============================================================"
echo "[3/6] LETKF / Nonlinear sigmoid / alpha 2"
echo "============================================================"

python -m swap.run_post_eval \
  --phase num_sample \
  --root_dir "${BASE_DIR}/kolmogorov_128_nonlinear/nonlinear/sigmoid/alpha_2" \
  --seeds $SEEDS \
  --cleanup


###############################################################################
# 4. Nonlinear speed / alpha 2
###############################################################################

echo "============================================================"
echo "[4/6] LETKF / Nonlinear speed / alpha 2"
echo "============================================================"

python -m swap.run_post_eval \
  --phase num_sample \
  --root_dir "${BASE_DIR}/kolmogorov_128_nonlinear/nonlinear/speed/alpha_2" \
  --seeds $SEEDS \
  --cleanup


###############################################################################
# 5. Grid mask / 256 / stride 15
###############################################################################

echo "============================================================"
echo "[5/6] LETKF / Grid mask / 256 / stride 15"
echo "============================================================"

python -m swap.run_post_eval \
  --phase num_sample \
  --root_dir "${BASE_DIR}/kolmogorov_256/grid_mask/stride_15" \
  --seeds $SEEDS \
  --cleanup


###############################################################################
# 6. Grid mask / 512 / stride 20
###############################################################################

echo "============================================================"
echo "[6/6] LETKF / Grid mask / 512 / stride 20"
echo "============================================================"

python -m swap.run_post_eval \
  --phase num_sample \
  --root_dir "${BASE_DIR}/kolmogorov_512/grid_mask/stride_20" \
  --seeds $SEEDS \
  --cleanup


echo "============================================================"
echo "[DONE] LETKF post evaluation"
echo "============================================================"