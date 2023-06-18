#!/usr/bin/env bash

sudo bash /mnt/common/jianshu/liquidio/common-dataset-mount/common_dateset_mount.sh
mount | grep s3_common_dataset


source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv

cd $(dirname $0)

NUM_GPUS=$1
set -x
PY_ARGS=${@:2}
# EXTRA_TAG=$3

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ./tools/train.py ${PY_ARGS}
