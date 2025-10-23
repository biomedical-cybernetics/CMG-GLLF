#!/bin/bash
# Usage wrapper to reproduce tuned results for CMG and SReLU used by plot_results.ipynb
# This script runs CM-GLLF (CMG) and SReLU tuners with the Supra-linear scheduler (power_to_linear)
# across both CIFAR10 and CIFAR100, covering batch sizes {32,64,128}, learning rates {0.025,0.01,0.001},
# and seeds {0,1,2} as expected by the notebook.
#
# Arguments:
#   $1 (optional) - CUDA device index (default: 0)
#
# Example:
#   bash usage_run_tuner_power.sh 0

set -euo pipefail

DEVICE=${1:-0}

# Datasets and modes required by the notebook figures
DATASETS=(CIFAR10 CIFAR100)
MODES=(
  "CM-GLLF-all-valuewise-minmax-mapping"  # CMG
  "srelu_valuewise_positive"              # SReLU
  # Additional baselines explored in the notebook/paper
  "linear-valuewise-positive"
  "cubic-valuewise-positive"
  "adaptive-sigmoid-valuewise"
  "adaptive-tanh-valuewise"
)

# Iterate and launch runs
for ds in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "[usage_run_tuner_power] Running dataset=${ds} mode=${mode} device=${DEVICE}"
    bash ./run_tuner_power.sh "${ds}" "${mode}" "${DEVICE}"
  done
done

echo "[usage_run_tuner_power] All jobs submitted. Results will be saved under ./results/tuner/..."
