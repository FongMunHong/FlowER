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
        seed = 8353189939158121202 # 8and13  - [best_tanimoto=0.786] target=CN(CCCC1c2ccccc2C=Cc2ccccc21)N=O  closest=C[NH+](CCCC1C2=CC=[CH+]C=C2C=Cc2ccccc21)N=O
        # seed = 15445470622804109308 # 9and14 - [best_tanimoto=1.000] target=Cc1ccc(Sc2ccccc2N2CCN(N=O)CC2)c(C)c1  closest=Cc1ccc(Sc2ccccc2N2CC[NH+](N=O)CC2)c(C)c1

        log_rank_0(f"Current_seed: {seed}")
        torch.manual_seed(seed)

        main(args, seed=seed)
        print()
        raise
