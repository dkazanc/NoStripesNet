# Code to load real-data from HDF files and then save them to disk as tiffs
import os
import math
import numpy as np
from mpi4py import MPI
import multiprocessing
from h5py import File
from skimage.transform import resize
from httomo.data.hdf._utils import load
from utils import getFlatsDarks, getMask_functional, loadHDF, save3DTiff, saveTiff, load3DTiff


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
        """Get a single item from the dataset,
            where the preview is defined by `item`
            and all other parameters come from `self.tomo_params`"""
        print(f"\t\tLoading file {self.file}...", end=' ', flush=True)
        self.setPreviewFromItem(item)
        # load HDF5 file
        data = loadHDF(os.path.join(self.root, self.file),
                       self.tomo_params, self.flats, self.darks, self.comm, self.ncore)[0]
        # any NaN values will cause errors so must be set to 0
        data[np.isnan(data)] = 0
        print("Done.")
        return data

    def getShifts(self, item):
        """Get a list of vertical shifts of a particular item"""
        file_num = int(self.file.split('.')[0])
        shifts = []
        for s in range(0, self.num_shifts):
            # Get correct file name
            self.file = str(file_num + s) + '.nxs'
            print(f"\tLoading shift {s}")
            # load HDF5 file
            data = self.__getitem__(item)
            # increment slice by shiftstep
            item = slice(item.start + self.shiftstep, item.stop + self.shiftstep, item.step)
            # swap axes
            data = np.swapaxes(data, 0, 1)
            # re-size data
            if data.size == 0:  # if data is empty, don't resize (as will cause error)
                data = np.ndarray((0, 402, 362))
            else:
                data = resize(data, (data.shape[0], 402, 362), anti_aliasing=True)
            shifts.append(data)
        # Reset self.file
        self.file = str(file_num) + '.nxs'
        # Pad shifts[1:] with data from shifts[0] so that all shifts are the same length
        # as otherwise (due the stepping between each shift) you get index-out-of-bounds errors
        new_shift = np.empty_like(shifts[0])
        for s in range(1, len(shifts)):
            if shifts[s].shape != shifts[0].shape:
                new_shift[:shifts[s].shape[0]] = shifts[s]
                new_shift[shifts[s].shape[0]:] = shifts[0][shifts[s].shape[0]:]
                shifts[s] = new_shift
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



def convertHDFtoTIFF(tiff_root, hdf_root, pipeline, no_slices=243, sampleNo=0, **kwargs):
    # Create pathnames
    cleanPath = os.path.join(tiff_root, 'clean')
    stripePath = os.path.join(tiff_root, 'shift00')
    # Create Dataset
    ds = RealDataset(hdf_root, pipeline, **kwargs)
    # Load data in chunks to speed it up
    num_chunks = math.ceil(len(ds) / no_slices)
    for i in range(num_chunks):
        print(f"Loading Chunk {i+1}/{num_chunks}")
        # Load `no_slices` slices for each shift
        shifts = ds.getShifts(np.s_[i*no_slices:(i+1)*no_slices])
        # Calculate input & target for each slice
        print("\tCreating input/target pairs for each slice & saving data...")
        for slc in range(shifts[0].shape[0]):
            current_slice = (i * no_slices) + slc
            inpt, target = ds.getCleanStripe([shift[slc] for shift in shifts])
            # Save input and target to disk as TIF files
            # saving 2D images first means virtual memory can be saved. also acts as back-up in case program crashes
            filename = os.path.join(cleanPath, str(sampleNo).zfill(4) + '_clean_' + str(current_slice).zfill(4))
            saveTiff(inpt, filename)  # each image will only be normalized w.r.t itself
            filename = os.path.join(stripePath, str(sampleNo).zfill(4) + '_shift00_' + str(current_slice).zfill(4))
            saveTiff(target, filename)
        print(f"Chunk {i+1} saved to '{tiff_root}'")
    # Load 3D images, clip & normalize, then re-save to disk
    print("Loading full 3D dataset and normalizing...")
    inpt_file = os.path.join(cleanPath, str(sampleNo).zfill(4) + '_clean')
    inpt3D = load3DTiff(inpt_file, (len(ds), 402, 362))
    target_file = os.path.join(stripePath, str(sampleNo).zfill(4) + '_shift00')
    target3D = load3DTiff(target_file, (len(ds), 402, 362))
    # clip so norm isn't skewed by anomalies in very low & very high slices
    inpt3D = np.clip(inpt3D, inpt3D[20:-20].min(), inpt3D[20:-20].max())
    target3D = np.clip(target3D, target3D[20:-20].min(), target3D[20:-20].max())
    save3DTiff(inpt3D, inpt_file, normalise=True)  # each image will be normalized w.r.t. the whole 3D sample
    save3DTiff(target3D, target_file, normalise=True)
    print(f"Full normalized dataset saved to '{tiff_root}'")
