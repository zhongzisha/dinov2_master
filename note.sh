
if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th21_ds
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11 
else
    source /data/zhongz2/anaconda3/bin/activate th21_ds
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0   
fi
export OMP_NUM_THREADS=4

OUTPUT_DIR=/data/zhongz2/temp29/dinov2_output
mkdir -p ${OUTPUT_DIR}
torchrun --nproc_per_node 4 \
    train1.py --config-file ssl_default_config_vit_small.yaml \
    --output-dir ${OUTPUT_DIR}


















