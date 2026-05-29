#!/bin/sh

## This script is intended for recreating paper results reliabily.

### RUN sh examples/run_beam_predict_seed.sh - outside of examples folder

export DATA_NAME="flower_new_dataset"
export EXP_NAME="best_large_hyperparam"
export EMB_DIM=256
export RBF_HIGH=12
export RBF_GAP=0.1
export SIGMA=0.15

export MODEL_NAME="model.2940000_97.pt" # your trained checkpoint here


export TRAIN_BATCH_SIZE=4096
export VAL_BATCH_SIZE=4096
export TEST_BATCH_SIZE=4096

export NUM_WORKERS=4
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS_PER_NODE=1

export NUM_NODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=1235
export SCALE=4

export TRAIN_FILE=$PWD/data/$DATA_NAME/train.txt
export VAL_FILE=$PWD/data/$DATA_NAME/val.txt
export TEST_FILE=$PWD/data/$DATA_NAME/beam.txt


export MODEL_PATH=$PWD/checkpoints/$DATA_NAME/$EXP_NAME/
export RESULT_PATH=$PWD/results/$DATA_NAME/$EXP_NAME/


# beam-search #
export BEAM_SIZE=5
export NBEST=3
export MAX_DEPTH=15
export CHUNK_SIZE=50
# export SAMPLE_SIZE=64   # optional; defaults to 64 // SCALE when unset


python examples/beam_predict_seed.py
