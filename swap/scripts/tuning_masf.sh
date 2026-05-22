#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

SEEDS="42 43 44 45 46"
OUT_DIR="${PROJECT_ROOT}/swap_results"

echo "============================================================"
echo "[RUN] MASF tuning"
echo "============================================================"

###############################################################################
# 1. Grid mask / 128 / stride 10
###############################################################################

echo "============================================================"
echo "[1/6] MASF / Grid mask / 128 / stride 10"
echo "============================================================"

python -m swap.run_tuning \
  --until finetuning \
  --dynamic_type kolmogorov \
  --dim 128 \
  --measurement_type grid_mask \
  --method_type ours \
  --stride 10 \
  --out_dir "$OUT_DIR" \
  --cleanup


###############################################################################
# 2. Center mask / 128 / hole 0.5
###############################################################################

echo "============================================================"
echo "[2/6] MASF / Center mask / 128 / hole 0.5"
echo "============================================================"

python -m swap.run_tuning \
  --until finetuning \
  --dynamic_type kolmogorov \
  --dim 128 \
  --measurement_type center_mask \
  --method_type ours \
  --hole_ratio 0.5 \
  --out_dir "$OUT_DIR" \
  --cleanup


###############################################################################
# 3. Nonlinear sigmoid / alpha 2
###############################################################################

echo "============================================================"
echo "[3/6] MASF / Nonlinear sigmoid / alpha 2"
echo "============================================================"

python -m swap.run_tuning \
  --until finetuning \
  --dynamic_type kolmogorov \
  --dim 128 \
  --measurement_type nonlinear \
  --nonlinear_type sigmoid \
  --method_type nonlinear_ours \
  --alpha 2 \
  --out_dir "$OUT_DIR" \
  --cleanup


###############################################################################
# 4. Nonlinear speed / alpha 2
###############################################################################

echo "============================================================"
echo "[4/6] MASF / Nonlinear speed / alpha 2"
echo "============================================================"

python -m swap.run_tuning \
  --until finetuning \
  --dynamic_type kolmogorov \
  --dim 128 \
  --measurement_type nonlinear \
  --nonlinear_type speed \
  --method_type nonlinear_ours \
  --alpha 2 \
  --out_dir "$OUT_DIR" \
  --cleanup


###############################################################################
# 5. Grid mask / 256 / stride 15
###############################################################################

echo "============================================================"
echo "[5/6] MASF / Grid mask / 256 / stride 15"
echo "============================================================"

python -m swap.run_tuning \
  --until finetuning \
  --dynamic_type kolmogorov \
  --dim 256 \
  --measurement_type grid_mask \
  --method_type ours \
  --stride 15 \
  --out_dir "$OUT_DIR" \
  --cleanup


###############################################################################
# 6. Grid mask / 512 / stride 20
###############################################################################

echo "============================================================"
echo "[6/6] MASF / Grid mask / 512 / stride 20"
echo "============================================================"

python -m swap.run_tuning \
  --until finetuning \
  --dynamic_type kolmogorov \
  --dim 512 \
  --measurement_type grid_mask \
  --method_type ours \
  --stride 20 \
  --out_dir "$OUT_DIR" \
  --cleanup


echo "============================================================"
echo "[DONE] MASF tuning"
echo "============================================================"