#!/bin/sh
#SBATCH --job-name=ml_training
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=cs05r
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module load python
conda activate nostripesnet

root="/dls/i12/data/2022/nt33730-1/processing/NoStripesNet/data/pruned" # directory where input data is stored
model="patch" # type of model to train. must be one of 'base', 'mask', 'simple', 'patch', 'window' or 'full'
size=256 # number of sinograms per sample
shifts=1 # number of shifts in vertical height per sample
batchsize=32 # batch size to load data in
epochs=10 # number of epochs
lr=0.0002 # learning rate
beta1=0.5 # beta 1 for adam optimizer
beta2=0.999 # beta 2 for adam optimizer
lambda=100 # weight for L1 loss in generator
savedir="/dls/i12/data/2022/nt33730-1/processing/NoStripesNet/pretrained_models/checkpoints/" # directory in which to save the models during training
width=25 # width of windows
name=patch_test
# Other parameters, such as --lsgan, --subset, --metrics, etc. must be added to the commands below

echo "Beginning training..."
srun python -m network.training -r $root -m $model -N $size -s $shifts -B $batchsize -e $epochs -l $lr -b $beta1 $beta2 --lambda $lambda -d $savedir --force -w $width -v -n $name --ddp
