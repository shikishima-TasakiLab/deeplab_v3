import argparse

import torch
from torch import Tensor
import torch.nn as nn
import os
import yaml
from ..constant import *
from typing import Dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', f'--{arg_hyphen(ARG_CHECK_POINT)}', type=str, metavar='PATH', help='Path of checkpoint file.')
    parser.add_argument('-x', f'--{arg_hyphen(ARG_WIDTH)}', type=int, default=512, help='Width of input images.')
    parser.add_argument('-y', f'--{arg_hyphen(ARG_HEIGHT)}', type=int, default=256, help='Height of input images.')
    parser.add_argument(f'--{arg_hyphen(ARG_TRAIN_CONFIG)}', type=str, metavar='PATH', default=None, help='PATH of "config.yaml"')
    return vars(parser.parse_args())

def main(args: Dict[str, str]):
    if isinstance(args[ARG_TRAIN_CONFIG], str) is True:
        if os.path.isfile(args[ARG_TRAIN_CONFIG]) is False:
            raise FileNotFoundError(f'{args[ARG_TRAIN_CONFIG]}')
        config_path: str = args[ARG_TRAIN_CONFIG]
    else:
        config_path: str = os.path.join(os.path.dirname(args[ARG_CHECK_POINT]), 'config.yaml')
    with open(config_path) as f:
        train_args: dict = yaml.safe_load(f)

    model: nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')
    model.classifier[4] = nn.Conv2d(256, train_args[ARG_NUM_CLASSES], 1)
    model.load_state_dict(torch.load(args[ARG_CHECK_POINT]))
    model.cpu()
    model.eval()

    sample_rgb: Tensor = torch.empty(1, 3, args[ARG_HEIGHT], args[ARG_WIDTH])

    traced_ts = torch.jit.trace(model, (sample_rgb), strict=False)
    traced_ts.save(f'{os.path.splitext(args[ARG_CHECK_POINT])[0]}.pt')
