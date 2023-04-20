#!/bin/bash
# Parameters (for full description of each parameter, see ../network/README.md)
root="./data" # directory where input data is stored
model="patch" # type of model to train. must be one of 'base', 'mask', 'simple', 'patch', 'window' or 'full'
size=256 # number of sinograms per sample
shifts=1 # number of shifts in vertical height per sample
batchsize=16 # batch size to load data in
epochs=1 # number of epochs
lr=0.0002 # learning rate
beta1=0.5 # beta 1 for adam optimizer
beta2=0.999 # beta 2 for adam optimizer
lambda=100 # weight for L1 loss in generator
savedir="./pretrained_models" # directory in which to save the models during training
width=25 # width of windows
# Other parameters, such as --lsgan, --subset, --metrics, etc. must be added to the commands below

echo "Beginning training..."
python -m network.training -r $root -m $model -N $size -s $shifts -B $batchsize -e $epochs -l $lr -b $beta1 $beta2 --lambda $lambda -d $savedir --force -w $width -v
echo "Network has finished training"
echo "Beginning testing..."
python -m network.testing -r $root -m $model -N $size -s $shifts -B $batchsize -f $savedir/${model}_${epochs}.tar -w $width -v
echo "Network has finished testing"
