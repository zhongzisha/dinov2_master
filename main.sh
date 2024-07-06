#!/bin/bash

#SBATCH --job-name=debug
#SBATCh --mail-type=ALL
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:2,lscratch:400
#SBATCH --time=200:00:00
#SBATCH --output=%x-%j.out
#SBATCH --export=ALL



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


srun --export ALL --jobid $SLURM_JOB_ID bash main_job.sh











