import argparse
import codecs
from contextlib import redirect_stdout
import datetime
import os
from glob import glob
from typing import Dict, OrderedDict
from tqdm import tqdm
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.dataloader import DataLoader
from torch.utils import tensorboard

from .constant import *
from .dataloader import init_train_dataloader, init_val_dataloader
from .metric import *

def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    parser_train = parser.add_argument_group('Training')
    parser_train.add_argument(
        '-t', f'--{arg_hyphen(ARG_TAG)}',
        type=str, required=True,
        help='Training Tag.'
    )
    parser_train.add_argument(
        '-tdc', f'--{arg_hyphen(ARG_TRAIN_DL_CONFIG)}',
        type=str, metavar='PATH', required=True,
        help='PATH of JSON file of dataloader config for training.'
    )
    parser_train.add_argument(
        '-vdc', f'--{arg_hyphen(ARG_VAL_DL_CONFIG)}',
        type=str, metavar='PATH', default=None,
        help=f'PATH of JSON file of dataloader config for validation. If not specified, the same file as "--{arg_hyphen(ARG_TRAIN_DL_CONFIG)}" will be used.'
    )
    parser_train.add_argument(
        '-bs', f'--{arg_hyphen(ARG_BLOCK_SIZE)}',
        type=int, default=0,
        help='Block size of dataset.'
    )
    parser_train.add_argument(
        '-td', f'--{arg_hyphen(ARG_TRAIN_DATA)}',
        type=str, metavar='PATH', nargs='+', required=True,
        help='PATH of training HDF5 datasets.'
    )
    parser_train.add_argument(
        '-vd', f'--{arg_hyphen(ARG_VAL_DATA)}',
        type=str, metavar='PATH', nargs='*', default=[],
        help=f'PATH of validation HDF5 datasets. If not specified, the same files as "--{arg_hyphen(ARG_TRAIN_DATA)}" will be used.'
    )
    parser_train.add_argument(
        '-ae', f'--{arg_hyphen(ARG_AUTO_EVALUATION)}',
        action='store_true',
        help='Auto Evaluation.'
    )
    parser_train.add_argument(
        '-edc', f'--{arg_hyphen(ARG_EVAL_DL_CONFIG)}',
        type=str, metavar='PATH',
        help='PATH of JSON file of dataloader config.'
    )
    parser_train.add_argument(
        '-ed', f'--{arg_hyphen(ARG_EVAL_DATA)}',
        type=str, metavar='PATH', nargs='*', default=[],
        help='PATH of evaluation HDF5 datasets.'
    )
    # parser_train.add_argument(
    #     '-r', f'--{arg_hyphen(ARG_RESUME)}',
    #     type=str, metavar='PATH', default=None,
    #     help='PATH of checkpoint.'
    # )
    # parser_train.add_argument(
    #     '-se', f'--{arg_hyphen(ARG_START_EPOCH)}',
    #     type=int, default=1,
    #     help='start epoch.'
    # )
    parser_train.add_argument(
        f'--{arg_hyphen(ARG_EPOCHS)}',
        type=int, default=200,
        help='Epochs'
    )
    parser_train.add_argument(
        f'--{arg_hyphen(ARG_STEPS_PER_EPOCH)}',
        type=int, default=10000,
        help='Number of steps per epoch. If it is greater than the total number of datasets, then the total number of datasets is used.'
    )
    parser_train.add_argument(
        f'--{arg_hyphen(ARG_SEED)}',
        type=int, default=0,
        help='Random seed.'
    )

    parser_net = parser.add_argument_group('Network')
    parser_net.add_argument(
        '-b', f'--{arg_hyphen(ARG_BATCH_SIZE)}',
        type=int, default=2,
        help='Batch Size'
    )
    parser_net.add_argument(
        '-amp', f'--{arg_hyphen(ARG_AMP)}',
        action='store_true',
        help='Use AMP.'
    )

    parser_optim = parser.add_argument_group('Optimizer')
    parser_optim.add_argument(
        f'--{arg_hyphen(ARG_BASE_LR)}',
        type=float, default=0.0005,
        help='base learning rate'
    )
    parser_optim.add_argument(
        f'--{arg_hyphen(ARG_ASPP_MULT)}',
        type=float, default=1.0,
        help='learning rate multiplier for ASPP'
    )
    parser_optim.add_argument(
        f'--{arg_hyphen(ARG_CLIP_MAX_NORM)}',
        type=float, default=1.0,
        help='max_norm for clip_grad_norm.'
    )

    args: dict = vars(parser.parse_args())
    return args

def main(args: dict, workdir: str):
    ###############
    # Random Seed #
    ###############
    np.random.seed(seed=args[ARG_SEED])
    torch.random.manual_seed(seed=args[ARG_SEED])

    ##################
    # Device Setting #
    ##################
    if torch.cuda.is_available():
        device: torch.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed=args[ARG_SEED])
        torch.backends.cudnn.deterministic = True
        print(f'{"Device":11s}: {torch.cuda.get_device_name(device)}')
    else:
        device: torch.device = torch.device('cpu')
        print(f'{"Device":11s}: CPU')

    ######################
    # DataLoader Setting #
    ######################
    if args[ARG_VAL_DL_CONFIG] is None:
        args[ARG_VAL_DL_CONFIG] = args[ARG_TRAIN_DL_CONFIG]
    if len(args[ARG_VAL_DATA]) == 0:
        args[ARG_VAL_DATA] = args[ARG_TRAIN_DATA]
    if args.get(ARG_EVAL_DL_CONFIG) is None:
        args[ARG_EVAL_DL_CONFIG] = args[ARG_VAL_DL_CONFIG]
    if len(args[ARG_EVAL_DATA]) == 0:
        args[ARG_EVAL_DATA] = args[ARG_VAL_DATA]
    if args[ARG_BLOCK_SIZE] < 3:
        args[ARG_BLOCK_SIZE] = 0

    with redirect_stdout(open(os.devnull, 'w')):
        train_dataloader: DataLoader = init_train_dataloader(args)
        val_dataloader: DataLoader = init_val_dataloader(args)

    steps_per_epoch: int = args[ARG_STEPS_PER_EPOCH]
    train_data_len: int = len(train_dataloader)
    if train_data_len < steps_per_epoch:
        steps_per_epoch = train_data_len
        args[ARG_STEPS_PER_EPOCH] = steps_per_epoch

    ###################
    # Network Setting #
    ###################
    model: nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')
    model.classifier[4] = nn.Conv2d(256, args[ARG_NUM_CLASSES], 1)

    model.to(device)

    backbone_params = (
        list(model.backbone.parameters()) +
        list(model.classifier[1].parameters()) +
        list(model.classifier[2].parameters()) +
        list(model.classifier[3].parameters()) +
        list(model.classifier[4].parameters())
    )
    aspp_params = (
        list(model.classifier[0].parameters())
    )

    ################
    # Count Params #
    ################
    params = 0
    for p in model.parameters():
        if p.requires_grad is True:
            params += p.numel()
    args[ARG_PARAMS] = params

    #####################
    # Optimizer Setting #
    #####################
    optimizer = optim.SGD(
        [
            {'params': filter(lambda p: p.requires_grad, backbone_params)},
            {'params': filter(lambda p: p.requires_grad, aspp_params)},
        ],
        lr=args[ARG_BASE_LR], momentum=0.9, weight_decay=1e-4
    )
    scaler = GradScaler(enabled=args[ARG_AMP])

    ################
    # Loss Setting #
    ################
    lossCE: nn.Module = nn.CrossEntropyLoss().to(device)

    ##################
    # Metric Setting #
    ##################
    metricSumCE: nn.Module = nn.CrossEntropyLoss(reduction='sum').to(device)
    metricSumIntersection: nn.Module = MetricSumIntersection().to(device)
    metricSumUnion: nn.Module = MetricSumUnion().to(device)

    val_best_miou = METRIC_BEST(0.0, '', 0)
    eval_best_miou = METRIC_BEST(0.0, '', 0)

    ###############
    # Save Config #
    ###############
    dt_start = datetime.datetime.now()
    args[ARG_DATE] = dt_start.strftime('%Y%m%dT%H%M%S')
    train_name: str = f'{args[ARG_DATE]}-{args[ARG_TAG]}'
    checkpoint_dir: str = os.path.join(workdir, DIR_CHECKPOINTS, train_name)

    print(f'{"CheckPoint":11s}: "{checkpoint_dir}"')

    #######################
    # TensorBoard Setting #
    #######################
    tb_log_dir: str = os.path.join(workdir, DIR_LOGS, train_name)
    tb_writer = None

    print(f'{"Summary":11s}: "{tb_log_dir}"')

    try:
        loss_dict: Dict[str, Tensor] = {}
        max_iter: int = args[ARG_EPOCHS] * steps_per_epoch
        #################
        # Training Loop #
        #################
        for epoch_itr in range(args[ARG_EPOCHS]):
            epoch_num: int = epoch_itr + 1
            ############
            # Training #
            ############
            model.train()

            for batch_itr in tqdm(range(steps_per_epoch), desc=f'Epoch {epoch_num:5d}: {"Train":14s}'):
                step_itr: int = epoch_itr * steps_per_epoch + batch_itr

                lr = args[ARG_BASE_LR] * (1 - float(step_itr) / max_iter) ** 0.9
                optimizer.param_groups[0]['lr'] = lr
                optimizer.param_groups[1]['lr'] = lr * args[ARG_ASPP_MULT]

                batch: Dict[str, Tensor] = next(iter(train_dataloader))
                in_camera: Tensor = batch[DATASET_CAMERA].to(device, non_blocking=True)
                gt_label: Tensor = batch[DATASET_LABEL].to(device, non_blocking=True)

                with autocast(enabled=args[ARG_AMP]):
                    pred: OrderedDict[str, Tensor] = model(in_camera)

                loss_seg_ce: Tensor = lossCE(pred['out'], gt_label)
                loss_dict['CrossEntropy'] = loss_seg_ce.clone().detach()
                loss_dict['LearningRate'] = lr
                scaler.scale(loss_seg_ce).backward()
                del loss_seg_ce

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args[ARG_CLIP_MAX_NORM])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                ######################
                # Update Tensorboard #
                ######################
                if (step_itr % 50 == 49):
                    if tb_writer is None:
                        os.makedirs(tb_log_dir, exist_ok=True)
                        tb_writer = tensorboard.SummaryWriter(log_dir=tb_log_dir)
                    for h2, value in loss_dict.items():
                        if isinstance(value, Tensor):
                            scalar = value.item()
                        else:
                            scalar = value
                        tb_writer.add_scalar(f'Training/{h2}', scalar, global_step=step_itr)
                    tb_writer.flush()

            ##############
            # Validation #
            ##############
            model.eval()

            sum_intersection: float = 0.0
            sum_union: float = 0.0
            sum_ce: float = 0.0
            count_px: float = 0.0

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f'Epoch {epoch_num:5d}: {"Validation":14s}'):
                    in_camera: Tensor = batch[DATASET_CAMERA].to(device)
                    gt_label: Tensor = batch[DATASET_LABEL].to(device)
                    count_px += in_camera.numel()

                    with autocast(enabled=args[ARG_AMP]):
                        pred: OrderedDict[str, Tensor] = model(in_camera)

                    ce = metricSumCE(pred['out'], gt_label)
                    sum_ce += ce

                    intersection: Tensor = metricSumIntersection(pred['out'], gt_label)
                    sum_intersection += intersection.cpu().detach().numpy()

                    union: Tensor = metricSumUnion(pred['out'], gt_label)
                    sum_union += union.cpu().detach().numpy()

            tmp_miou: float = np.mean(sum_intersection / sum_union)
            tmp_ce: float = sum_ce / count_px

            val_metrics: Dict[str, float] = {}
            val_metrics['mIoU'] = tmp_miou
            val_metrics['CrossEntropy'] = tmp_ce

            ###############
            # Show Result #
            ###############
            print(f'Epoch {epoch_num:5d}: {"Metric":14s}: {"mIoU":14s}: {tmp_miou * 100.0:8.2f} [ % ]')
            print(f'{"":11s}: {"":14s}: {"CrossEntropy":14s}: {tmp_ce:8.4f}')

            ######################
            # Update TensorBoard #
            ######################
            if tb_writer is None:
                os.makedirs(tb_log_dir, exist_ok=True)
                tb_writer = tensorboard.SummaryWriter(log_dir=tb_log_dir)
            for h2, value in val_metrics.items():
                tb_writer.add_scalar(f'Validation/{h2}', value, epoch_num)
            tb_writer.flush()

            ###################
            # Save CheckPoint #
            ###################
            if os.path.isdir(checkpoint_dir) is False:
                os.makedirs(checkpoint_dir, exist_ok=True)
                with codecs.open(os.path.join(checkpoint_dir, 'config.yaml'), mode='w', encoding='utf-8') as f:
                    yaml.dump(args, f, encoding='utf-8', allow_unicode=True)

            epoch_str: str = f'{epoch_num:05d}'
            if (val_best_miou.value < tmp_miou):
                ckpt_path: str = os.path.join(checkpoint_dir, f'{epoch_num:06d}_DeepLabV3.pth')
                torch.save(model.state_dict(), ckpt_path)

                ###################
                # Auto Evaluation #
                ###################
                if args.get(ARG_AUTO_EVALUATION) is True:
                    # TODO
                    pass

            tmp_ckpts_epoch: set = {val_best_miou.epoch}
            if val_best_miou.value < tmp_miou:
                val_best_miou = METRIC_BEST(tmp_miou, epoch=epoch_num)
            rm_ckpts_epoch: set = tmp_ckpts_epoch - {val_best_miou.epoch}
            for rm_ckpt_epoch in rm_ckpts_epoch:
                # if {rm_ckpt_epoch} <= eval_best_epoch: continue
                for ckpt_path in glob(os.path.join(checkpoint_dir, f'{rm_ckpt_epoch:06d}_DeepLabV3.pth')):
                    if os.path.isfile(ckpt_path): os.remove(ckpt_path)

    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    finally:
        if tb_writer is not None:
            tb_writer.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
