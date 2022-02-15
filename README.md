# DeepLab-v3

## Requirement

- NVIDIA-Driver `>=418.81.07`
- Docker `>=19.03`
- NVIDIA-Docker2

## Docker Images

- build

    ```bash
    docker pull pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
    ```
    ```bash
    ./docker/build.sh -i pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
    ```

## Preparing Datasets

### KITTI-360

1. Store the KITTI-360 dataset in HDF5 using "[h5_kitti360](https://github.com/shikishima-TasakiLab/h5_kitti360)".

1. Use `./config/kitti360-5class.json` for the dataloader configuration file.

## Start a Docker Container

1. Start a Docker container with the following command.

    ```bash
    ./docker/run.sh -d path/of/the/dataset/dir
    ```
    ```
    Usage: run.sh [OPTIONS...]
    OPTIONS:
        -h, --help          Show this help
        -i, --gpu-id ID     Specify the ID of the GPU
        -d, --dataset-dir   Specify the directory where datasets are stored
    ```

## Training

1. Start training with the following command.

    ```bash
    python train.py -t TAG -tdc path/of/the/config.json \
      -td path/of/the/dataset1.hdf5 [path/of/the/dataset1.hdf5 ...] \
      -bs BLOCK_SIZE
    ```
    ```
    usage: train.py [-h] -t TAG -tdc PATH [-vdc PATH] [-bs BLOCK_SIZE] -td PATH
                    [PATH ...] [-vd [PATH [PATH ...]]] [-ae] [-edc PATH]
                    [-ed [PATH [PATH ...]]] [--epochs EPOCHS]
                    [--steps-per-epoch STEPS_PER_EPOCH] [--seed SEED]
                    [-b BATCH_SIZE] [-amp] [--base-lr BASE_LR]
                    [--aspp-mult ASPP_MULT] [--clip-max-norm CLIP_MAX_NORM]

    optional arguments:
      -h, --help            show this help message and exit

    Training:
      -t TAG, --tag TAG     Training Tag.
      -tdc PATH, --train-dl-config PATH
                            PATH of JSON file of dataloader config for training.
      -vdc PATH, --val-dl-config PATH
                            PATH of JSON file of dataloader config for validation.
                            If not specified, the same file as "--train-dl-config"
                            will be used.
      -bs BLOCK_SIZE, --block-size BLOCK_SIZE
                            Block size of dataset.
      -td PATH [PATH ...], --train-data PATH [PATH ...]
                            PATH of training HDF5 datasets.
      -vd [PATH [PATH ...]], --val-data [PATH [PATH ...]]
                            PATH of validation HDF5 datasets. If not specified,
                            the same files as "--train-data" will be used.
      -ae, --auto-evaluation
                            Auto Evaluation.
      -edc PATH, --eval-dl-config PATH
                            PATH of JSON file of dataloader config.
      -ed [PATH [PATH ...]], --eval-data [PATH [PATH ...]]
                            PATH of evaluation HDF5 datasets.
      --epochs EPOCHS       Epochs
      --steps-per-epoch STEPS_PER_EPOCH
                            Number of steps per epoch. If it is greater than the
                            total number of datasets, then the total number of
                            datasets is used.
      --seed SEED           Random seed.

    Network:
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch Size
      -amp, --amp           Use AMP.

    Optimizer:
      --base-lr BASE_LR     base learning rate
      --aspp-mult ASPP_MULT
                            learning rate multiplier for ASPP
      --clip-max-norm CLIP_MAX_NORM
                            max_norm for clip_grad_norm.
    ```

1. The checkpoints of the training will be stored in the "./checkpoints" directory.

    ```
    checkpoints/
    　├ YYYYMMDDThhmmss-TAG/
    　│　├ config.yaml
    　│　├ 00001_PMOD.pth
    　│　├ :
    　│　├ :
    　│　├ EPOCH_PMOD.pth
    　│　└ validation.xlsx
    ```

## Evaluation

1. Start evaluation with the following command.

    ```bash
    python eval.py -t TAG -cp path/of/the/checkpoint.pth \
      -edc path/of/the/config.json \
      -ed path/of/the/dataset1.hdf5 [path/of/the/dataset2.hdf5 ...]
    ```
    ```
    usage: eval.py [-h] -t TAG -cp PATH -edc PATH [-bs BLOCK_SIZE] -ed PATH
                   [PATH ...] [--train-config PATH] [--seed SEED] [-b BATCH_SIZE]
                   [-amp]

    optional arguments:
      -h, --help            show this help message and exit

    Evaluation:
      -t TAG, --tag TAG     Evaluation Tag.
      -cp PATH, --check-point PATH
                            PATH of checkpoint.
      -edc PATH, --eval-dl-config PATH
                            PATH of JSON file of dataloader config.
      -bs BLOCK_SIZE, --block-size BLOCK_SIZE
                            Block size of dataset.
      -ed PATH [PATH ...], --eval-data PATH [PATH ...]
                            PATH of evaluation HDF5 datasets.
      --train-config PATH   PATH of "config.yaml"
      --seed SEED           Random seed.

    Network:
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch Size
      -amp, --amp           Use AMP.
    ```

1. The results of the evaluation will be stored in the "./results" directory.

    ```
    results/
    　├ YYYYMMDDThhmmss-TRAINTAG/
    　│　├ YYYYMMDDThhmmss-TAG/
    　│　│　├ config.yaml
    　│　│　├ data.hdf5
    　│　│　└ result.xlsx
    ```

## data.hdf5 &rarr; Video (.avi)

1. Convert "data.hdf5" to video (.avi) with the following command.

    ```bash
    python data2avi.py -i path/of/the/data.hdf5
    ```
    ```
    usage: data2avi.py [-h] -i PATH [-o PATH] [-r]

    optional arguments:
      -h, --help            show this help message and exit
      -i PATH, --input PATH
                            Input path.
      -o PATH, --output PATH
                            Output path. Default is "[input dir]/data.avi"
      -r, --raw             Use raw codec
    ```

1. The converted video will be output to the same directory as the input HDF5 file.

## Checkpoint (.pth) → Torch Script model (.pt)

1. Convert checkpoint (.pth) to Torch Script model (.pt) with the following command.

    ```bash
    python ckpt2tsm.py -c path/of/the/checkpoint.pth
    ```
    ```
    usage: ckpt2tsm.py [-h] [-c PATH] [-x WIDTH] [-y HEIGHT] [--train-config PATH]

    optional arguments:
      -h, --help            show this help message and exit
      -c PATH, --check-point PATH
                            Path of checkpoint file.
      -x WIDTH, --width WIDTH
                            Width of input images.
      -y HEIGHT, --height HEIGHT
                            Height of input images.
      --train-config PATH   PATH of "config.yaml"
    ```

1. The converted Torch Script model will be output to the same directory as the input checkpoint.
