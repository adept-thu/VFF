#!/usr/bin/env bash


# sudo bash /mnt/common/jianshu/liquidio/common-dataset-mount/common_dateset_mount.sh
# mount | grep s3_common_dataset


source ~/miniconda3/etc/profile.d/conda.sh
conda activate vff
cd $(dirname $0)/..

set -x
NGPUS=$1
PY_ARGS=${@:2}

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}
