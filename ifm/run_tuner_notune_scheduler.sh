#!/bin/bash
# Configurations

# change
dataset=$1
scheduler=$2
# Set network based on dataset
if [ "$dataset" == "CIFAR10" ]; then
  network="MLP_CIFAR10_A"
elif [ "$dataset" == "CIFAR100" ]; then
  network="MLP_CIFAR100_A"
fi
device=$3
learning_rates=(0.025 0.01 0.001)
batch_sizes=(128 64 32)
seeds=(0 1 2)
# Fixed parameters
epochs=150
wd=5e-4

# train_tuner
for seed in "${seeds[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for lr_w in "${learning_rates[@]}"; do
          python train_tuner.py \
              --network $network \
              --dataset $dataset \
              --batch_size $bs \
              --lr_w $lr_w \
              --epochs $epochs \
              --seed $seed \
              --device $device \
              --weight_decay $wd \
	            --scheduler_w $scheduler &  
          done
    done
    wait
done
