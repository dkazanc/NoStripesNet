#!/bin/bash
# Parameters
root="../data" # root where data was created
shifts=5 # number of shifts in vertical height of each sample
size=256 # grid size of sample (and therefore also height of 3D sinogram generated)
savedir="../images" # directory in which to save the models during training
model="mask" # model type. must be one of 'window' or 'base'
width=25 # width of windows
epochs=1 # number of epochs
lr=0.0002 # learning rate
beta1=0.5 # beta 1 for adam optimizer
beta2=0.999 # beta 2 for adam optimizer
batchsize=32 # batch size for minibatches

echo "Beginning training..."
python ../network/training.py -r $root -m $model -N $size -s $shifts -w $width -e $epochs -l $lr -b $beta1 $beta2 -B $batchsize -d $savedir -v --force
echo "Network has finished training"
echo "Beginning testing..."
python ../network/testing.py -r $root -m $model -N $size -s $shifts -w $width -B $batchsize -v -f $savedir/${model}_${epochs}.tar
echo "Network has finished testing"
