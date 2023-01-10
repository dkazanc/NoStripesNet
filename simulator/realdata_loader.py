# Code to load real-data from HDF files and then save them to disk as tiffs
import os
import yaml
import csv
import numpy as np
from mpi4py import MPI
import multiprocessing
from h5py import File
from skimage.transform import resize
from utils import plot_images, getFlatsDarks, reconstruct, getMask_functional, loadHDF, save3DTiff
import timeit


class RealDataset:
    """Dataset to get sinograms from HDF5 files"""

    def __init__(self, root, pipeline, flats_file=None, num_shifts=5, shiftstep=13):
        self.root = os.path.dirname(root)
        self.file = os.path.basename(root)
        self.tomo_params = pipeline[0]['httomo.data.hdf.loaders']['standard_tomo']
        self.cor_params = pipeline[1]['tomopy.recon.rotation']['find_center_vo']
        self.rec_params = pipeline[2]['tomopy.recon.algorithm']['recon']
        self.num_shifts = num_shifts
        self.shiftstep = shiftstep
        # multi-processing stuff idk
        self.comm = MPI.COMM_WORLD
        if self.comm.size == 1:
            self.ncore = multiprocessing.cpu_count()  # use all available CPU cores if not an MPI run
        # get shape of dataset
        with File(os.path.join(self.root, self.file), "r", driver="mpio", comm=self.comm) as file:
            self.shape = file[self.tomo_params['data_path']].shape
        if flats_file is None:
            self.flats, self.darks = getFlatsDarks(os.path.join(self.root, self.file),
                                                   self.tomo_params, self.shape, self.comm)
        else:
            self.flats, self.darks = getFlatsDarks(os.path.join(self.root, flats_file),
                                                   self.tomo_params, self.shape, self.comm)

    def __len__(self):
        return self.shape[1]

    def __getitem__(self, item):
        """ Return a list of sinograms of the different vertical shifts of the sample."""
        self.setPreviewFromItem(item)
        file_num = int(self.file.split('.')[0])
        shifts = []
        for s in range(0, self.num_shifts):
            # Get correct file name
            self.file = str(file_num + s) + '.nxs'
            print(f"Loading shift {s} (file {self.file})...", end=' ', flush=True)
            # load HDF5 file
            data = loadHDF(os.path.join(self.root, self.file),
                           self.tomo_params, self.flats, self.darks, self.comm, self.ncore)[0]
            self.tomo_params['preview'][1]['start'] += self.shiftstep
            self.tomo_params['preview'][1]['stop'] += self.shiftstep
            # any NaN values will cause errors so must be set to 0
            data[np.isnan(data)] = 0
            # swap axes
            data = np.swapaxes(data, 0, 1)
            # re-size data
            data = resize(data, (data.shape[0], 402, 362), anti_aliasing=True)
            shifts.append(data)
            print("Done.")
        # Reset self.file
        self.file = str(file_num) + '.nxs'
        return shifts

    @staticmethod
    def getCleanStripe(shifts):
        clean = shifts[0].copy()
        stripe = shifts[0].copy()
        masks = [getMask_functional(s) for s in shifts]
        for col in range(stripe.shape[1]):
            for i, mask in enumerate(masks):
                vert_sum = np.sum(mask[:, col])
                if vert_sum == 0:
                    clean[:, col] = shifts[i][:, col]
                else:
                    stripe[:, col] = shifts[i][:, col]
        return clean, stripe

    def setPreviewFromItem(self, item):
        if type(item) == slice:
            self.tomo_params['preview'][1]['start'] = item.start
            self.tomo_params['preview'][1]['stop'] = item.stop
            self.tomo_params['preview'][1]['step'] = item.step
        else:
            self.tomo_params['preview'][1]['start'] = item
            self.tomo_params['preview'][1]['stop'] = item + 1


def convertHDFtoTIFF(tiff_root, hdf_root, pipeline, sampleNo=0, **kwargs):
    # Create Dataset
    pipeline = yaml.safe_load(open(pipeline))
    ds = RealDataset(hdf_root, pipeline, **kwargs)
    clean3D = np.ndarray((len(ds), 402, 362))  # is hard-coding the size a good idea?
    stripe3D = np.ndarray((len(ds), 402, 362))
    # Load clean & stripe data into 3D array
    print("\nLoading item 0")
    for i, (clean, stripe) in enumerate(ds):
        clean3D[i] = clean
        stripe3D[i] = stripe
        print(f"\nLoading item {i + 1}")
    # Save 3D array as a series of tiff images
    sampleRoot = os.path.join(tiff_root, str(sampleNo).zfill(4))
    # clean
    cleanPath = os.path.join(sampleRoot, 'clean')
    filename = os.path.join(cleanPath, str(sampleNo).zfill(4) + '_clean')
    save3DTiff(clean3D, filename, normalise=True)
    # stripe
    stripePath = os.path.join(sampleRoot, 'shift' + str(0).zfill(2))
    filename = os.path.join(stripePath, str(sampleNo).zfill(4) + '_shift' + str(0).zfill(2))
    save3DTiff(stripe3D, filename, normalise=True)
