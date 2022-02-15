from typing import NamedTuple
import torch
from torch import Tensor

class METRIC_BEST(NamedTuple):
    value: float
    path: str = None
    epoch: int = 0

class METRIC_RESULT:
    def __init__(self, name:str = '', use_class:bool = False, use_thresholds:bool = False, all_tag:str = 'All') -> None:
        self.name: str = name
        self.use_class: bool = use_class
        self.use_thresholds: bool = use_thresholds
        self.all_tag: str = all_tag

        self.all_metric: Tensor = None
        self.class_metric: Tensor = None
        self.threshold_metric: Tensor = None

        self.counts_metric: Tensor = None
        self.sum_class_metric: Tensor = None
        self.sum_threshold_metric: Tensor = None

    def add(self, count_metric: Tensor = None, class_metric: Tensor = None, threshold_metric: Tensor = None):
        if isinstance(count_metric, Tensor):
            if self.counts_metric is None:
                self.counts_metric = torch.zeros_like(count_metric)
            self.counts_metric = self.counts_metric + count_metric

        if isinstance(class_metric, Tensor):
            tmp_class_metric = class_metric
        elif isinstance(self.class_metric, Tensor):
            tmp_class_metric = self.class_metric
        else:
            tmp_class_metric = None
        if tmp_class_metric is not None:
            if self.sum_class_metric is None:
                self.sum_class_metric = torch.zeros_like(tmp_class_metric)
            self.sum_class_metric = self.sum_class_metric + tmp_class_metric

        if isinstance(threshold_metric, Tensor):
            tmp_threshold_metric = threshold_metric
        elif isinstance(self.threshold_metric, Tensor):
            tmp_threshold_metric = self.threshold_metric
        else:
            tmp_threshold_metric = None
        if tmp_threshold_metric is not None:
            if self.sum_threshold_metric is None:
                self.sum_threshold_metric = torch.zeros_like(tmp_threshold_metric)
            self.sum_threshold_metric = self.sum_threshold_metric + tmp_threshold_metric

def arg_hyphen(arg: str):
    return arg.replace('_', '-')

ARG_TAG                 = 'tag'
ARG_BATCH_SIZE          = 'batch_size'
ARG_BLOCK_SIZE          = 'block_size'
ARG_TRAIN_DATA          = 'train_data'
ARG_TRAIN_DL_CONFIG     = 'train_dl_config'
ARG_VAL_DATA            = 'val_data'
ARG_VAL_DL_CONFIG       = 'val_dl_config'
ARG_EPOCHS              = 'epochs'
ARG_STEPS_PER_EPOCH     = 'steps_per_epoch'
ARG_EVAL_DATA           = 'eval_data'
ARG_EVAL_DL_CONFIG      = 'eval_dl_config'
ARG_SEED                = 'seed'
ARG_AMP                 = 'amp'
ARG_CLIP_MAX_NORM       = 'clip_max_norm'
ARG_BASE_LR             = 'base_lr'
ARG_ASPP_MULT           = 'aspp_mult'
ARG_AUTO_EVALUATION     = 'auto_evaluation'
ARG_RESUME              = 'resume'
ARG_START_EPOCH         = 'start_epoch'

ARG_DATE                = 'date'
ARG_NUM_CLASSES         = 'num_classes'
ARG_PARAMS              = 'params'
ARG_LABEL_TAGS          = 'label_tags'

ARG_CHECK_POINT         = 'check_point'
ARG_TRAIN_CONFIG        = 'train_config'

ARG_HEIGHT              = 'height'
ARG_WIDTH               = 'width'

DATASET_CAMERA          = 'camera'
DATASET_LABEL           = 'label'

DIR_CHECKPOINTS         = 'checkpoints'
DIR_LOGS                = 'logs'
DIR_RESULTS             = 'results'

METRIC_IOU              = 'IoU'
