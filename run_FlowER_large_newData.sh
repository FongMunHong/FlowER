#!/bin/sh


export DATA_NAME="flower_new_dataset"
export EXP_NAME="best_large_hyperparam"
export EMB_DIM=256
export RBF_HIGH=12
export RBF_GAP=0.1
export SIGMA=0.15

export MODEL_NAME="model.2760000_91.pt" # your trained checkpoint here

export TRAIN_BATCH_SIZE=2048
export VAL_BATCH_SIZE=2048
export TEST_BATCH_SIZE=4096

export NUM_WORKERS=4
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS_PER_NODE=2

export NUM_NODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=1235

export TRAIN_FILE=$PWD/data/$DATA_NAME/train.txt
export VAL_FILE=$PWD/data/$DATA_NAME/val.txt
export TEST_FILE=$PWD/data/$DATA_NAME/test.txt
#export TEST_FILE=$PWD/data/$DATA_NAME/beam.txt


export MODEL_PATH=$PWD/checkpoints/$DATA_NAME/$EXP_NAME/
export RESULT_PATH=$PWD/results/$DATA_NAME/$EXP_NAME/


# [ -f $TRAIN_FILE ] || { echo $TRAIN_FILE does not exist; exit; }
# [ -f $VAL_FILE ] || { echo $VAL_FILE does not exist; exit; }
# [ -f $TEST_FILE ] || { echo $TEST_FILE does not exist; exit; }


#export SCALE=4 # smaller sample size during training validation
#sh scripts/train.sh

export SCALE=1 # larger sample size during testing
sh scripts/eval_multiGPU.sh
#sh scripts/search.sh
# sh scripts/search_multiGPU.sh