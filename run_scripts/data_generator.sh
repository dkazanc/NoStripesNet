#!/bin/bash
mkdir ../data -p
rm -rf ../data/*

# Parameters
samples=10 # number of samples generate sinograms from
shifts=5 # number of shifts in vertical height of each sample
size=256 # grid size of sample (and therefore also height of 3D sinogram generated)
objects=300 # number of objects to generate per sample
I0=40000 # full-beam photon flux intensity
flatsnum=20 # the number of the flat fields generated
shift_step=5 # the shift step of a sample in pixels

echo "Data generation has begun"
python ../simulator/data_generator.py -S $samples -s $shifts -N $size -o $objects -I $I0 -f $flatsnum -p $shift_step -v
echo "Data generation finished"
