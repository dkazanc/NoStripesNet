#!/bin/bash
mkdir ../data -p
rm -rf ../data/*

# Parameters
samples=1 # Number of samples generate sinograms from
shifts=5 # number of shifts in vertical height of each sample
size=256 # grid size of sample (and therefore also height of 3D sinogram generated)
objects=300 # number of objects to generate per sample

echo "Data generation has begun"
python ../simulator/data_generator.py -S $samples -s $shifts -N $size -o $objects -v
echo "Data generation finished"
