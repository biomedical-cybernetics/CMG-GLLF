#!/bin/bash
# Configurations

# change
dataset="CIFAR100"
mode="CM-GLLF-all-valuewise-minmax-mapping"
scheduler=$1
batch_size=$2
device=$3
# learning_rates=(0.01 0.001 0.0001)
learning_rates=(0.025 0.01 0.001)
seeds=(0 1 2)
# Fixed parameters
epochs=150
wd=5e-4
init_method="uniform"
# Set network based on dataset
if [ "$dataset" == "CIFAR10" ]; then
  network="MLP_CIFAR10_A"
elif [ "$dataset" == "CIFAR100" ]; then
  network="MLP_CIFAR100_A"
fi
# train_tuner
for seed in "${seeds[@]}"; do
    # when batch size is 32 or 64, run all tasks in parallel and wait
      for lr_w in "${learning_rates[@]}"; do
      for lr_special in "${learning_rates[@]}"; do
              python train_tuner.py \
                  --network $network \
                  --dataset $dataset \
                  --batch_size $batch_size \
                  --lr_w $lr_w \
                  --lr_special $lr_special \
                  --epochs $epochs \
                  --seed $seed \
                  --use_tuner \
                  --mode $mode \
                  --init_method $init_method \
                  --device $device \
                  --train_2 \
                  --weight_decay $wd \
                  --scheduler_w $scheduler &
      done
      done
      wait
done
