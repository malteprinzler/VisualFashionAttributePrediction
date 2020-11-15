#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=5000
#SBATCH --partition=2080ti-long
#SBATCH --output slurm_logs/out.txt

# activate conda env
conda activate embedding

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
srun python3 train.py --gpus 4 --distributed_backend ddp --data_root /home/mprinzler/storage/iMaterialist --batch_size 16 $@
