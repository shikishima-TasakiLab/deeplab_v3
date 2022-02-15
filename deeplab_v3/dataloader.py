from typing import Dict, List, Tuple, Union
from h5dataloader.pytorch.hdf5dataset import HDF5Dataset
from h5dataloader.common.structure import CONFIG_TAG_LABELTAG, CONFIG_TAG_LABEL, CONFIG_TAG_TAG
from torch.utils.data.dataloader import DataLoader
from .constant import *

def init_train_dataloader(args: dict) -> DataLoader:
    use_mods: Tuple[int, int] = None
    block_size: int = args[ARG_BLOCK_SIZE]

    if block_size > 2:
        use_mods = (0, block_size - 2)

    dataset = HDF5Dataset(h5_paths=args[ARG_TRAIN_DATA], config=args[ARG_TRAIN_DL_CONFIG], quiet=True, block_size=block_size, use_mods=use_mods)
    args[ARG_NUM_CLASSES] = len(dataset.label_color_configs[dataset.minibatch[DATASET_LABEL][CONFIG_TAG_LABELTAG]])

    return DataLoader(dataset, batch_size=args[ARG_BATCH_SIZE], shuffle=True)

def init_val_dataloader(args: dict) -> DataLoader:
    use_mods: Tuple[int, int] = None
    block_size: int = args[ARG_BLOCK_SIZE]

    if block_size > 2:
        use_mods = (block_size - 2, block_size - 1)

    dataset = HDF5Dataset(h5_paths=args[ARG_VAL_DATA], config=args[ARG_VAL_DL_CONFIG], quiet=True, block_size=block_size, use_mods=use_mods)

    return DataLoader(dataset, batch_size=args[ARG_BATCH_SIZE], shuffle=False)

def init_eval_dataloader(args: dict) -> DataLoader:
    use_mods: Tuple[int, int] = None
    block_size: int = args[ARG_BLOCK_SIZE]

    if block_size > 2:
        use_mods = (block_size - 1, block_size)

    dataset = HDF5Dataset(h5_paths=args[ARG_EVAL_DATA], config=args[ARG_EVAL_DL_CONFIG], quiet=True, block_size=block_size, use_mods=use_mods)

    label_color_configs: List[Dict[str, Union[str, int]]] = dataset.label_color_configs[dataset.minibatch[DATASET_LABEL][CONFIG_TAG_LABELTAG]]
    args[ARG_LABEL_TAGS] = {str(lcc[CONFIG_TAG_LABEL]): lcc[CONFIG_TAG_TAG] for lcc in label_color_configs}

    return DataLoader(dataset, batch_size=args[ARG_BATCH_SIZE], shuffle=False)
