#!/bin/bash
#SBATCh --mail-type=FAIL

if [ "${SLURM_JOB_NODELIST}" != "" ]; then
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    NNODES=$SLURM_NNODES
    GPUS_PER_NODE=2
else
    MASTER_ADDR=`hostname`
    NNODES=1
    GPUS_PER_NODE=2
fi
MASTER_PORT=25199

OUTPUT_DIR=/data/zhongz2/temp29/dinov2_output_v2_debug
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    train2.py --config-file ssl_default_config_vit_small2.yaml \
    --output-dir ${OUTPUT_DIR}







