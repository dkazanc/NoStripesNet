#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=cs05r
#SBATCH --job-name=nsn

h5='/path/to/h5_files/<file_name>.h5'
archive='/path/to/archive.npz'
file='m<file_name>'

echo "Loading environment..."
module load python
conda activate nostripesnet
echo "Environment loaded"

echo "Running script"
python -m simulator.generate_mask --h5 $h5 -a $archive -f $file