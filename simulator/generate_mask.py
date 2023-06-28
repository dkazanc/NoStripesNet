import timeit
import argparse
import numpy as np
from pathlib import Path
from larix.methods.misc import STRIPES_DETECT, STRIPES_MERGE
from utils.tomography import TomoH5


def append_npz(archive_name, file_name, arr):
    archive_name = Path(archive_name)
    if archive_name.exists():
        archive = np.load(archive_name)
        archive_files = dict(archive)
    else:
        archive_files = {}
    archive_files[file_name] = arr
    np.savez_compressed(archive_name, **archive_files)


def larix_stripes_detect(tomo, archive, file_name, save_weights=False):
    print("Calculating stripe weights...", flush=True)
    start = timeit.default_timer()
    weights = STRIPES_DETECT(tomo, size=250, radius=3)
    print(f"Done in {timeit.default_timer() - start:.5f}s", flush=True)
    if save_weights:
        print(f"Saving stripe weights...")
        try:
            start = timeit.default_timer()
            append_npz(archive, f'{file_name}_weights', weights)
            print(f"Done in {timeit.default_timer() - start:.5f}s")
        except (OSError, IOError) as e:
            print(f"Failed: {e}")
    print("Merging weights to create stripe mask...", flush=True)
    start = timeit.default_timer()
    mask = STRIPES_MERGE(weights,
                         threshold=0.63,
                         min_stripe_length=600,
                         min_stripe_depth=30,
                         min_stripe_width=22,
                         sensitivity_perc=85.0)
    print(f"Done in {timeit.default_timer() - start:.5f}s", flush=True)
    print("Saving mask...")
    start = timeit.default_timer()
    append_npz(archive, file_name, mask)
    print(f"Done in {timeit.default_timer() - start:.5f}s")


def get_args():
    parser = argparse.ArgumentParser(description="Generate a stripe mask.")
    parser.add_argument("--h5", type=str, default=None,
                        help="HDF5 or Nexus file containing raw tomographic "
                             "data to generate mask from.")
    parser.add_argument("--flats", type=str, default=None,
                        help="HDF5 or Nexus file containing flats and darks.")
    parser.add_argument('-a', "--archive", type=str, default=None,
                        help="Path to archive file to save data. Will be "
                             "created if it doesn't exist.")
    parser.add_argument('-f', "--file-name", type=str, default=None,
                        help="File name to save data within archive.")
    parser.add_argument('-w', "--weights", action='store_true',
                        help="Save weights as well as mask to archive.")
    return parser.parse_args()


def generate_mask():
    args = get_args()

    begin = timeit.default_timer()
    # Load h5 file
    print(f"Initializing h5 file {args.h5}...", flush=True)
    start = timeit.default_timer()
    th5 = TomoH5(args.h5)
    print(f"Done in {timeit.default_timer() - start:.5f}s")
    print(f"Data Shape: {th5.shape}\n")

    # Get flats & darks
    flats, darks = None, None
    if args.flats is not None:
        print(f"Loading flats & darks...")
        start = timeit.default_timer()
        flat_h5 = TomoH5(args.flats)
        flats = flat_h5.get_flats()
        darks = flat_h5.get_darks()
        print(f"Done in {timeit.default_timer() - start:.5f}s\n")

    # Load normalized 3D tomogram
    print("Loading tomogram...", flush=True)
    start = timeit.default_timer()
    data = th5.get_normalized(np.s_[:], flats=flats, darks=darks)
    stop = timeit.default_timer()
    print(f"Done in: {stop - start:.5f}s", flush=True)
    print(f"Tomogram: {data.shape}, {data.dtype}, "
          f"[{data.min()}, {data.max()}]\n")

    # Test larix stripe detection
    print("Detecting stripes...", flush=True)
    larix_stripes_detect(data, args.archive, args.file_name, args.weights)
    print(f"Total time: {timeit.default_timer() - begin:.5f}s")


if __name__ == '__main__':
    generate_mask()
