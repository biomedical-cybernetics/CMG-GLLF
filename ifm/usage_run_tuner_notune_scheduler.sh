#!/bin/bash
# Usage wrapper to reproduce no-tuner (Original MLP) baselines used by plot_results.ipynb
# This script runs the Original MLP (no tuner) across CIFAR10 and CIFAR100
# for the schedulers used in the figures: {power_to_linear, linear, exponential_to_linear}.
# It covers batch sizes {32,64,128}, learning rates {0.025,0.01,0.001}, and seeds {0,1,2}.
#
# Arguments:
#   $1 (optional) - CUDA device index (default: 0)
#
# Example:
#   bash usage_run_tuner_notune_scheduler.sh 0

set -euo pipefail

DEVICE=${1:-0}

DATASETS=(CIFAR100)
SCHEDULERS=(power_to_linear linear exponential_to_linear)

for ds in "${DATASETS[@]}"; do
  for sched in "${SCHEDULERS[@]}"; do
    echo "[usage_run_tuner_notune_scheduler] Running dataset=${ds} scheduler=${sched} device=${DEVICE}"
    bash ./run_tuner_notune_scheduler.sh "${ds}" "${sched}" "${DEVICE}"
  done
done

echo "[usage_run_tuner_notune_scheduler] All jobs submitted. Results will be saved under ./results/tuner/..."
