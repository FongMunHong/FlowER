import sys
sys.path.append('.')
import os
import torch
import numpy as np
from beam_predict import Args, setup_logger, log_args, main, log_rank_0

if __name__ == "__main__":
    args = Args
    args.local_rank = int(os.environ["LOCAL_RANK"]) if os.environ.get("LOCAL_RANK") else -1
    logger = setup_logger(args, "beam")
    log_args(args, 'evaluation') 

    for i in range(100):
        seed = torch.seed()
        log_rank_0(f"Current_seed: {seed}")
        torch.manual_seed(seed)

        main(args, seed=seed)
