
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

DATA_DIR=/lscratch/$SLURM_JOB_ID/cache_dir/train
torchrun --nproc_per_node 2 \
    train_wds.py --config-file ssl_default_config_vit_small.yaml \
    --output-dir /lscratch/$SLURM_JOB_ID/cache_dir/output \
    train.dataset_path="${DATA_DIR}"


















