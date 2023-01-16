# Code to load real-data from HDF files and then save them to disk as tiffs
import os
import math
import numpy as np
from mpi4py import MPI
import multiprocessing
from h5py import File
from skimage.transform import resize
from utils import getFlatsDarks, getMask_functional, loadHDF, save3DTiff, saveTiff


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
            print(f"\tLoading shift {s} (file {self.file})...", end=' ', flush=True)
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
            if data.shape[0] != 0:  # if data is empty, don't resize (as will cause error)
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
            self.tomo_params['preview'][1]['start'] = min(item.start, len(self))
            self.tomo_params['preview'][1]['stop'] = min(item.stop, len(self))
            self.tomo_params['preview'][1]['step'] = item.step
        else:
            self.tomo_params['preview'][1]['start'] = item
            self.tomo_params['preview'][1]['stop'] = item + 1


def convertHDFtoTIFF(tiff_root, hdf_root, pipeline, no_slices=243, sampleNo=0, backup_save=True, **kwargs):
    # Create pathnames
    sampleRoot = os.path.join(tiff_root, str(sampleNo).zfill(4))
    cleanPath = os.path.join(sampleRoot, 'clean')
    stripePath = os.path.join(sampleRoot, 'shift00')
    # Create Dataset
    ds = RealDataset(hdf_root, pipeline, **kwargs)
    inpt3D = np.ndarray((len(ds), 402, 362))
    target3D = np.ndarray((len(ds), 402, 362))
    # Load data in chunks to speed it up
    num_chunks = math.ceil(len(ds) / no_slices)
    for i in range(num_chunks):
        print(f"Loading Chunk {i+1}/{num_chunks}")
        # Load `no_slices` slices for each shift
        shifts = ds[i*no_slices:(i+1)*no_slices]
        # Pad shifts[1:] with data from shifts[0] so that all shifts are the same length
        # as otherwise (due the stepping between each shift) you get index-out-of-bounds errors
        new_shift = np.empty_like(shifts[0])
        for s in range(1, len(shifts)):
            if shifts[s].shape != shifts[0].shape:
                new_shift[:shifts[s].shape[0]] = shifts[s]
                new_shift[shifts[s].shape[0]:] = shifts[0][shifts[s].shape[0]:]
                shifts[s] = new_shift
        # Calculate input & target for each slice
        print("\tCreating input/target pairs for each slice & saving data...")
        for slc in range(shifts[0].shape[0]):
            current_slice = (i * no_slices) + slc
            inpt3D[current_slice], target3D[current_slice] = ds.getCleanStripe([shift[slc] for shift in shifts])
            # Save input and target to disk as TIF files
            # (this is done pre-normalization, so is mainly just a backup incase the program crashes mid-execution)
            if backup_save:
                filename = os.path.join(cleanPath, str(sampleNo).zfill(4) + '_clean_' + str(current_slice).zfill(4))
                saveTiff(inpt3D[current_slice], filename)  # each image will only be normalized w.r.t itself
                filename = os.path.join(stripePath, str(sampleNo).zfill(4) + '_shift00_' + str(current_slice).zfill(4))
                saveTiff(target3D[current_slice], filename)
        print(f"Chunk {i+1} saved to '{tiff_root}'")
    # Normalize 3D images & save to disk
    # clip so norm isn't skewed by anomalies in very low & very high slices
    inpt3D = np.clip(inpt3D, inpt3D[20:-20].min(), inpt3D[20:-20].max())
    target3D = np.clip(target3D, target3D[20:-20].min(), target3D[20:-20].max())
    filename = os.path.join(cleanPath, str(sampleNo).zfill(4) + '_clean')
    save3DTiff(inpt3D, filename, normalise=True)  # each image will be normalized w.r.t. the whole 3D sample
    filename = os.path.join(stripePath, str(sampleNo).zfill(4) + '_shift00')
    save3DTiff(target3D, filename, normalise=True)
    print(f"Full normalized dataset saved to '{tiff_root}'")
