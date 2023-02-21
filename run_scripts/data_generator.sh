#!/bin/bash
# Parameters
mode=complex # sample generation mode
root='./data' # data generation root directory
samples=1 # number of samples generate sinograms from
shifts=5 # number of shifts in vertical height of each sample
size=256 # grid size of sample (and therefore also height of 3D sinogram generated)
objects=300 # number of objects to generate per sample
I0=40000 # full-beam photon flux intensity
flatsnum=20 # the number of the flat fields generated
shift_step=5 # the shift step of a sample in pixels
start=0 # number at which to start generating samples
pipeline='./tomo_pipeline.yml' # HTTomo YAML pipeline file
hdf='/path/to/hdf_file.nxs' # HDF file containing real-life data

echo "Data generation has begun"
python -m simulator.data_generator -m $mode -r $root -S $samples -s $shifts -N $size -o $objects -I $I0 -f $flatsnum -p $shift_step --start $start --pipeline $pipeline --hdf-file $hdf -v
echo "Data generation finished"
