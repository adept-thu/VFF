#!/usr/bin/env bash


# sudo bash /mnt/common/jianshu/liquidio/common-dataset-mount/common_dateset_mount.sh
# mount | grep s3_common_dataset


source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv
cd $(dirname $0)/..

set -x

PY_ARGS=${@:1}
echo $RESOURCE_NUM_GPU
echo $DISTRIBUTED_NODE_COUNT
echo $DISTRIBUTED_NODE_RANK
echo $DISTRIBUTED_MASTER_HOSTS
echo $DISTRIBUTED_PYTORCH_PORT

python -m torch.distributed.launch \
--nproc_per_node=$RESOURCE_NUM_GPU \
--nnodes=$DISTRIBUTED_NODE_COUNT \
--node_rank=$DISTRIBUTED_NODE_RANK \
--master_addr=$DISTRIBUTED_MASTER_HOSTS \
--master_port=$DISTRIBUTED_PYTORCH_PORT \
train.py ${PY_ARGS}
