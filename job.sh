#!/bin/bash

#SBATCH --mail-type=FAIL

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th21_ds
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11 
else
    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0   
fi

cd ~/dinov2

srun python create_tcga_tarfiles.py

exit;


sbatch --partition=multinode --mem=100G --gres=lscratch:100 --time=108:00:00 --cpus-per-task=1 --nodes=64 --ntasks-per-node=1 \
    job.sh



















