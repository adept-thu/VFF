#!/bin/sh
# 启动脚本，只能用sh编译器，很多bash的用不了，所以先包装下。

nvidia-smi

cd $(dirname $0)
set -x

# 定义变量
GPUS=""
PARAMS=""

# 解析命令行参数
while [ $# -gt 0 ]; do
    case "$1" in
        --GPUS)
            GPUS="$2"
            shift 2
            ;;
        *)
            PARAMS="$PARAMS $1"
            shift
            ;;
    esac
done

# 提取GPUS后的一位数字
GPU_NUMBER=$(echo "$GPUS" | sed 's/.*\([0-9]\)$/\1/')

# 拼接参数
PARAMS="$PARAMS"

# 输出结果
echo "GPUS: $GPU_NUMBER"
echo "PARAMS: $PARAMS"


# OpenPCDet自带的默认脚本 传入两个参数 GPU_NUMBER是显卡数量 PARAMS是pyhton的参数
scripts/dist_train.sh ${GPU_NUMBER} ${PARAMS}