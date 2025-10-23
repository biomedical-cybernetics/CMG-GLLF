#!/bin/bash
# Usage wrapper to reproduce CM-GLLF (CMG) tuned results across schedulers used by plot_results.ipynb
# This script targets CIFAR100 (as in the notebook) and runs CM-GLLF with schedulers
# {linear, exponential_to_linear, power_to_linear} and batch sizes {32,64,128}.
# Learning rates {0.025,0.01,0.001} and seeds {0,1,2} are handled inside run_gllf_scheduler.sh.
#
# Arguments:
#   $1 (optional) - CUDA device index (default: 0)
#
# Example:
#   bash usage_run_gllf_scheduler.sh 0

set -euo pipefail

DEVICE=${1:-0}

SCHEDULERS=(linear exponential_to_linear power_to_linear)
BATCH_SIZES=(32 64 128)

for sched in "${SCHEDULERS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    echo "[usage_run_gllf_scheduler] Running scheduler=${sched} bs=${bs} device=${DEVICE}"
    bash ./run_gllf_scheduler.sh "${sched}" "${bs}" "${DEVICE}"
  done
done

echo "[usage_run_gllf_scheduler] All jobs submitted. Results will be saved under ./results/tuner/..."
