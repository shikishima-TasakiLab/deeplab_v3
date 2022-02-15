import os
from deeplab_v3.train import parse_args, main

if __name__ == '__main__':
    workdir = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()
    main(args, workdir)
