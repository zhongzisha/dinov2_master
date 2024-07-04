import os
import hostlist    # from python-hostlist

# get SLURM variables
if 'SLURM_PROCID' in os.environ:
    rank = int(os.environ['SLURM_PROCID'])
else:
    rank = 0

if 'SLURM_LOCALID' in os.environ:
    local_rank = int(os.environ['SLURM_LOCALID'])
else:
    local_rank = 0

if 'SLURM_NTASKS' in os.environ:
    world_size = int(os.environ['SLURM_NTASKS'])
else:
    world_size = 1

if 'SLURM_CPUS_PER_TASK' in os.environ:
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
else:
    cpus_per_task = 1

if 'SLURM_JOB_PARTITION' in os.environ and 'GPU' in os.environ['SLURM_JOB_PARTITION']:
    # get IDs of reserved GPU
    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
    has_gpu = True
else:
    gpu_ids = ['-1']
    has_gpu = False

if 'SLURM_JOB_NODELIST' in os.environ:
    # get node list from slurm
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])  # r[123,387], v[]
else:
    hostnames = ['127.0.0.1']

os.environ['MASTER_ADDR'] = hostnames[0]
os.environ['MASTER_PORT'] = str(23456 + int(min(gpu_ids)))  # to avoid port conflict on the same node
