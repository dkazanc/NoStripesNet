import argparse
import os
import yaml
from .data_simulator import generateSample, simulateFlats, simulateStripes
from .realdata_loader import convertHDFtoTIFF


def makeDirectories(dataDir, sampleNo, shifts):
    """Function to make sub-directories for data generation.
    These will have the following structure:
        .
        ├── data
        │   └── <sampleNo>
        │       ├── clean
        │       ├── shift00
        │       ├── shift01
                ...
    IMPORTANT: This function assumes it is being run from: NoStripesNet/
               Therefore it is important that this function is executed in the correct location from the terminal.
    """
    samplePath = os.path.join(dataDir, str(sampleNo).zfill(4))
    os.makedirs(samplePath, exist_ok=True)
    cleanPath = os.path.join(samplePath, 'clean')
    os.makedirs(cleanPath, exist_ok=True)
    for shift in range(shifts):
        shiftPath = os.path.join(samplePath, 'shift' + str(shift).zfill(2))
        os.makedirs(shiftPath, exist_ok=True)
    return samplePath, cleanPath


def get_args():
    parser = argparse.ArgumentParser(description="Create directories and generate samples of data.")
    parser.add_argument('-m', '--mode', type=str, default='complex',
                        help="Type of data to generate. Must be one of: ['simple', 'complex', 'real']")
    parser.add_argument('-r', '--root', type=str, default=None, help="Data root to generate samples in")
    parser.add_argument('-S', "--samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument('-s', "--shifts", type=int, default=5, help="Number of vertical shifts to apply to each sample")
    parser.add_argument('-N', "--size", type=int, default=256,
                        help="Size of image generated (cubic). Also height of sinogram")
    parser.add_argument('-o', "--objects", type=int, default=300, help="Number of objects used to generate each sample")
    parser.add_argument('-I', "--I0", type=int, default=40000, help="Full-beam photon flux intensity")
    parser.add_argument('-f', "--flatsnum", type=int, default=20, help="Number of the flat fields generated")
    parser.add_argument('-p', "--shiftstep", type=int, default=2, help="Shift step of a sample in pixels")
    parser.add_argument("--start", type=int, default=0,
                        help="Sample number to begin at (useful if some data has already been generated)")
    parser.add_argument("--pipeline", type=str, default='tomo_pipeline.yml',
                        help="YAML pipeline file for loading HDF data using HTTomo. (only used when --mode is real)")
    parser.add_argument("--hdf-file", type=str, default=None, help="HDF file to load real data from."
                                                                   "(only used when --mode is real)")
    parser.add_argument('-v', "--verbose", action="store_true", help="Print some extra information when running")
    return parser.parse_args()


if __name__ == '__main__':
    current_dir = os.path.basename(os.path.abspath(os.curdir))
    if current_dir != 'NoStripesNet':
        raise RuntimeError(f"Current Directory should be '.../NoStripesNet/'. Instead got '{current_dir}'.\n"
                           f"If Current Directory is not 'NoStripesNet', file and directory creation will be incorrect.")
    args = get_args()
    root = args.root
    if root is None:
        root = os.path.join(os.curdir, 'data')
    samples = args.samples
    shifts = args.shifts
    size = args.size
    objects = args.objects
    I0 = args.I0
    flatsnum = args.flatsnum
    shift_step = args.shiftstep
    verbose = args.verbose
    start = args.start
    total_samples = start + samples
    for sampleNo in range(start, total_samples):
        if verbose:
            print(f"Generating sample [{str(sampleNo).zfill(4)} / {str(total_samples-1).zfill(4)}]")
        samplePath, cleanPath = makeDirectories(root, sampleNo, shifts)
        if args.mode == 'simple':
            sample_clean = generateSample(size, objects, output_path=cleanPath, sampleNo=sampleNo, verbose=verbose)
            # TO-DO: Turn all the parameters below into CLI arguments
            sample_shifts = simulateStripes(sample_clean, percentage=1.2, max_thickness=3.0, intensity=0.25,
                                            kind='mix', variability=0, output_path=samplePath, sampleNo=sampleNo,
                                            verbose=verbose)
        elif args.mode == 'complex':
            # don't save 'clean' sample after it's generated
            # instead save 'clean' sample after flat noise has been added
            sample_clean = generateSample(size, objects, sampleNo=sampleNo, verbose=verbose)
            sample_shifts = simulateFlats(sample_clean, size, I0=I0, flatsnum=flatsnum, shifted_positions_no=shifts,
                                      shift_step=shift_step, output_path=samplePath, sampleNo=sampleNo, verbose=verbose)
        elif args.mode == 'real':
            pipeline = yaml.safe_load(open(args.pipeline))
            if args.hdf_file is None:
                raise ValueError("HDF File is None. Please include '--hdf-file' option.")
            convertHDFtoTIFF(samplePath, args.hdf_file, pipeline, sampleNo=sampleNo, num_shifts=shifts,
                             shiftstep=shift_step)
        else:
            raise ValueError(f"Option '--mode' should be one of ['simple', 'complex', 'real']. "
                             f"Instead got '{args.mode}'.")
