import argparse
import os

from data_io import *
from data_simulator import generateSample, simulateFlats


def makeDirectories(sampleNo, shifts):
    """Function to make sub-directories for data generation.
    These will have the following structure:
        .
        ├── data
        │   └── <sampleNo>
        │       ├── clean
        │       ├── shift00
        │       ├── shift01
                ...
    IMPORTANT: This function assumes it is being run from: NoStripesNet/run_scripts/
               Therefore it is important that this function is executed in the correct location from the terminal.
    """
    dataDir = os.path.join(os.pardir, 'data')
    samplePath = os.path.join(dataDir, str(sampleNo).zfill(4))
    os.mkdir(samplePath)
    cleanPath = os.path.join(samplePath, 'clean')
    os.mkdir(cleanPath)
    for shift in range(shifts):
        shiftPath = os.path.join(samplePath, 'shift' + str(shift).zfill(2))
        os.mkdir(shiftPath)
    return samplePath, cleanPath


def get_args():
    parser = argparse.ArgumentParser(description="Create directories and generate samples of data.")
    parser.add_argument('-S', "--samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument('-s', "--shifts", type=int, default=5, help="Number of vertical shifts to apply to each sample")
    parser.add_argument('-N', "--size", type=int, default=256,
                        help="Size of image generated (cubic). Also height of sinogram")
    parser.add_argument('-o', "--objects", type=int, default=300, help="Number of objects used to generate each sample")
    parser.add_argument('-v', "--verbose", action="store_true", help="Print some extra information when running")
    return parser.parse_args()


if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'run_scripts':
        raise RuntimeError(f"Current Working Directory should be '.../NoStripesNet/run_scripts'. Instead got '{os.getcwd()}'.\n"
                           f"If CWD is not 'NoStripesNet/run_scripts', file and directory creation will be incorrect.")

    args = get_args()
    samples = args.samples
    shifts = args.shifts
    size = args.size
    objects = args.objects
    verbose = args.verbose

    for sampleNo in range(samples):
        if verbose:
            print(f"Generating sample [{str(sampleNo).zfill(4)} / {str(samples-1).zfill(4)}]")
        samplePath, cleanPath = makeDirectories(sampleNo, shifts)
        sample_clean = generateSample(size, objects, cleanPath, sampleNo, verbose=verbose)
        sample_shifts = simulateFlats(sample_clean, size, samplePath, sampleNo, verbose=verbose)
