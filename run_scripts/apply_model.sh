#!/bin/bash

#SBATCH --job-name=nsn
#SBATCH --output=apply_model.log
#SBATCH --partition=cs05r
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "Loading environment..."
module load python
conda activate nostripesnet
echo "Environment loaded"

model='/path/to/model.tar'
h5='/path/to/rawdata.hdf5'
mask='/path/to/stripe_mask.npz'
out='/path/to/output.hdf5'

echo "Running script"
srun python apply_model.py $model $h5 -m $mask -o $out --ddp
