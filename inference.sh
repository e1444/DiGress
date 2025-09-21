#!/bin/bash

BATCHES=20
BATCH_SIZE=512

LOG_DIR="gpu-logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p $LOG_DIR

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 > $LOG_DIR/gpu_usage.csv &
MONITOR_PID=$!

time python src/main.py \
    +experiment=moses.yaml \
    dataset=moses.yaml \
    general.gpus=1 \
    general.test_only=$CHECKPOINT_FILE \
    general.resume=$CHECKPOINT_FILE \
    general.final_model_samples_to_generate=$(($BATCHES * $BATCH_SIZE)) \
    train.batch_size=$BATCH_SIZE \
    train.num_workers=4 > log.out 2> log.err

kill $MONITOR_PID