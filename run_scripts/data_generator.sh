#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=cs05r
#SBATCH --job-name=nsn

echo "Loading environment..."
module load python
conda activate nostripesnet
echo "Environment loaded"

# Parameters (for full description of each parameter, see ../simulator/README.md)
mode=patch # type of data to generate
root='./data' # directory to save data in
samples=1 # number of samples generate
start=0 # sample number to begin counting at
shifts=1 # number of vertical shifts for each sample
shift_step=5 # step in pixels between each shift
size=256 # cubic size of sample generated
objects=300 # number of objects to generate per sample
flatsnum=20 # the number of flat fields to generate
I0=40000 # full-beam photon flux intensity
hdf='/path/to/hdf_file.nxs' # Nexus file containing HDF data
chunk=243 # no. of slices to load per chunk
flat='/path/to/flat_file.nxs' # Nexus file containg flats & darks
mask='/path/to/mask_file.npz' # Npz file containg mask of stripe locations
angles=900 # number of angles per 'frame' of a dynamic scan
patch_h=1801 # height of patches to split data into
patch_w=256 # width of patches to split data into

echo "Data generation has begun"
python -m simulator.data_generator -m $mode -r $root -S $samples --start $start -s $shifts -p $shift_step -N $size -o $objects -f $flatsnum -I $I0 --hdf-file $hdf -C $chunk --flats $flat --mask $mask --frame-angles $angles --patch-size $patch_h $patch_w -v
echo "Data generation finished"
