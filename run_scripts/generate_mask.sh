#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=cs05r
#SBATCH --job-name=nsn

h5='/dls/i12/data/2022/nt33730-1/rawdata/119647.nxs'
archive='/dls/i12/data/2022/nt33730-1/processing/NoStripesNet/stripe_masks.npz'
file='m119647'

echo "Loading environment..."
module load python
conda activate nostripesnet
echo "Environment loaded"

echo "Running script"
python -m simulator.generate_mask --h5 $h5 -a $archive -f $file