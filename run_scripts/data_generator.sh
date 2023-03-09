#!/bin/bash
# Parameters (for full description of each parameter, see ../simulator/README.md)
mode=complex # type of data to generate
root='./data' # directory to save data in
samples=1 # number of samples generate
start=0 # sample number to begin counting at
shifts=5 # number of vertical shifts for each sample
shift_step=5 # step in pixels between each shift
size=256 # cubic size of sample generated
objects=300 # number of objects to generate per sample
flatsnum=20 # the number of flat fields to generate
I0=40000 # full-beam photon flux intensity
pipeline='./tomo_pipeline.yml' # HTTomo YAML pipeline file
hdf='/path/to/hdf_file.nxs' # Nexus file containing HDF data
angles=900 # number of angles per 'frame' of a dynamic scan

echo "Data generation has begun"
python -m simulator.data_generator -m $mode -r $root -S $samples --start $start -s $shifts -p $shift_step -N $size -o $objects -f $flatsnum -I $I0 --pipeline $pipeline --hdf-file $hdf  --frame-angles $angles -v
echo "Data generation finished"
