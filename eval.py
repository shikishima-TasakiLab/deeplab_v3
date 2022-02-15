import os
import numpy as np
import torch
from deeplab_v3.evaluate import parse_args, main
from deeplab_v3.constant import ARG_SEED

if __name__ == '__main__':
    workdir = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()

    ###############
    # Random Seed #
    ###############
    np.random.seed(seed=args[ARG_SEED])
    torch.random.manual_seed(seed=args[ARG_SEED])

    main(args, workdir)
