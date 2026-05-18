import os


DATA_NAME = os.environ.get("DATA_NAME", "USPTO")
EXP_NAME = os.environ.get("EXP_NAME", "")

SCALE = int(os.environ.get("SCALE", 4)) # train & val
# SCALE = 1 # test
SAMPLE_SIZE = 64 // SCALE
NUM_GPU = int(os.environ.get("NUM_GPUS_PER_NODE", 1))


TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 4096 * NUM_GPU))
VAL_BATCH_SIZE = int(os.environ.get("VAL_BATCH_SIZE", 4096 * NUM_GPU))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", 512 * NUM_GPU * SCALE))

NUM_NODES = int(os.environ.get("NUM_NODES", 1))
ACCUMULATION_COUNT = int(os.environ.get("ACCUMULATION_COUNT", 1))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 16))

MODEL_NAME = os.environ.get("MODEL_NAME")

class Args:
    # train #
    model_name = MODEL_NAME
    exp_name = EXP_NAME
    train_path = os.environ.get("TRAIN_FILE") 
    val_path = os.environ.get("VAL_FILE")
    test_path = os.environ.get("TEST_FILE")
    model_path = os.environ.get("MODEL_PATH")
    result_path = os.environ.get("RESULT_PATH")
    data_name = f"{DATA_NAME}"
    log_file = f"FlowER"
    load_from = ""
    seed = int(os.environ.get("SEED", 42))
    resume = True
    load_from = f"{model_path}{MODEL_NAME}"

    backend = "nccl"
    num_workers = NUM_WORKERS
    emb_dim = int(os.environ.get("EMB_DIM"))
    enc_num_layers = 12
    post_processing_layers = 1
    enc_heads = 32
    enc_filter_size = 2048
    dropout = 0.0
    attn_dropout = 0.0
    rel_pos = "emb_only"
    shared_attention_layer = 0
    sigma = float(os.environ.get("SIGMA"))
    train_batch_size = (TRAIN_BATCH_SIZE // ACCUMULATION_COUNT // NUM_GPU // NUM_NODES)
    val_batch_size = (VAL_BATCH_SIZE // ACCUMULATION_COUNT // NUM_GPU // NUM_NODES)
    test_batch_size = TEST_BATCH_SIZE
    batch_type = "tokens_sum"
    lr = 0.0003
    beta1 = 0.9
    beta2 = 0.998
    eps = 1e-9
    weight_decay = 1e-2
    warmup_steps = 30000
    clip_norm = 200


    epoch = int(os.environ.get("EPOCH", 100))
    max_steps = 3000000
    accumulation_count = ACCUMULATION_COUNT
    save_iter = int(os.environ.get("SAVE_ITER", 30000))
    log_iter = int(os.environ.get("LOG_ITER", 100))
    eval_iter = int(os.environ.get("EVAL_ITER", 30000))


    sample_size = SAMPLE_SIZE
    rbf_low = 0
    rbf_high = float(os.environ.get("RBF_HIGH"))
    rbf_gap = float(os.environ.get("RBF_GAP"))

    # validation #
    # do_validate = True
    # steps2validate =  ["1050000", "1320000", "1500000", "930000", "1020000"]

    # inference # 
    do_validate = False

    # beam-search #
    beam_size = 10
    nbest = 10
    max_depth = 15
    chunk_size = 50
