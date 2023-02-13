# Code to load real-data from HDF files and then save them to disk as tiffs
import os
import math
import numpy as np
from mpi4py import MPI
import multiprocessing
from h5py import File
from skimage.transform import resize
from skimage.exposure import match_histograms
from tomophantom.supp.artifacts import stripes as add_stripes
from utils.tomography import getFlatsDarks
from utils.data_io import loadHDF, loadTiff, load3DTiff, saveTiff, save3DTiff, rescale
from utils.stripe_detection import getMask_functional, getMask_morphological


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

    def setHDFFile(self, path):
        self.root = os.path.dirname(path)
        self.file = os.path.basename(path)

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
        stripes, masks = [], []
        # Get list of length `num_shifts` of just stripey sinograms
        for s in shifts:
            mask = getMask_morphological(s)
            # if sum of mask is 0 (i.e. sinogram has no stripes), artificially add in stripes
            if mask.sum() == 0:
                new_s = add_stripes(s, percentage=1, maxthickness=2, intensity_thresh=0.2,
                                    stripe_type=np.random.choice(['partial', 'full']), variability=0.005)
                new_s = np.clip(new_s, s.min(), s.max())
                stripes.append(new_s)
            else:
                stripes.append(s)
            masks.append(mask)
        # Get clean sinogram by combining all parts of shifts that don't contain stripes
        for col in range(clean.shape[1]):
            for i, mask in enumerate(masks):
                vert_sum = np.sum(mask[:, col])
                # if all masks have been looped through and none are stripe-free, set clean to last shift
                if vert_sum == 0 or i+1 == len(masks):
                    clean[:, col] = shifts[i][:, col]
                    break
        return clean, stripes

    def setPreviewFromItem(self, item):
        if type(item) == tuple:
            for i in range(len(item)):
                self.tomo_params['preview'][i]['start'] = item[i].start
                self.tomo_params['preview'][i]['stop'] = item[i].stop
                self.tomo_params['preview'][i]['step'] = item[i].step
        elif type(item) == slice:
            self.tomo_params['preview'][1]['start'] = min(item.start, len(self))
            self.tomo_params['preview'][1]['stop'] = min(item.stop, len(self))
            self.tomo_params['preview'][1]['step'] = item.step
        else:
            self.tomo_params['preview'][1]['start'] = item
            self.tomo_params['preview'][1]['stop'] = item + 1


def createDynamicDataset(tiff_root, hdf_file, pipeline, no_slices=243, sampleNo=0, sino_size=900):
    # Create pathname
    dynamicPath = os.path.join(tiff_root, 'dynamic')
    # Create Dataset
    ds = RealDataset(hdf_file, pipeline, flats_file='/dls/i12/data/2022/nt33730-1/rawdata/119675.nxs')
    num_chunks = math.ceil(len(ds) / no_slices)
    # Load each "frame" of the dynamic data
    for f in range(0, ds.shape[0], sino_size * 2):
        frame_no = f//(sino_size*2)
        print(f"Loading Frame {frame_no+1}/{ds.shape[0]//(sino_size*2)}")
        # Load data in chunks to speed it up
        for chunk in range(num_chunks):
            print(f"\tLoading Chunk {chunk+1}/{num_chunks}")
            item = np.s_[f:f+sino_size, chunk*no_slices:(chunk+1)*no_slices]
            frame = ds[item]  # 3D data with shape (sino_size, no_slices, ds.shape[2])
            # swap axes so shape is (no_slices, sino_size, ds.shape[2])
            frame = np.swapaxes(frame, 0, 1)
            # re-size data to shape (no_slices, 402, 362)
            if frame.size == 0:  # if data is empty, don't resize (as will cause error)
                frame = np.ndarray((0, 402, 362))
            else:
                frame = resize(frame, (frame.shape[0], 402, 362), anti_aliasing=True)
            # save each sinogram in frame
            for s in range(frame.shape[0]):
                sino = frame[s]
                filename = os.path.join(dynamicPath, str(sampleNo).zfill(4) + '_frame' + str(frame_no).zfill(2)
                                        + '_' + str(chunk * no_slices + s).zfill(4))
                saveTiff(sino, filename)
            print(f"\tChunk {chunk+1} loaded & saved to '{dynamicPath}'")
        print("Entire frame saved, now re-loading and normalizing...")
        # Once entire frame has been loaded, normalize w.r.t the whole 3D data
        filename = os.path.join(dynamicPath,  str(sampleNo).zfill(4) + '_frame' + str(frame_no).zfill(2))
        frame3D = load3DTiff(filename, (len(ds), 402, 362))
        # clip so norm isn't skewed by anomalies in very low & very high slices
        frame3D = np.clip(frame3D, frame3D[20:-20].min(), frame3D[20:-20].max())
        # match histogram?
        save3DTiff(frame3D, filename, normalise=True)
        print(f"Frame {frame_no+1} done.")


def convertHDFtoTIFF(tiff_root, hdf_root, pipeline, no_slices=243, sampleNo=0, num_shifts=20, **kwargs):
    # Create pathnames
    cleanPath = os.path.join(tiff_root, 'clean')
    stripePaths = [os.path.join(tiff_root, 'shift' + str(s).zfill(2)) for s in range(num_shifts)]
    # Create Dataset
    ds = RealDataset(hdf_root, pipeline, num_shifts=num_shifts, **kwargs)
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
            target, inpts = ds.getCleanStripe([shift[slc] for shift in shifts])
            # Save input and target to disk as TIF files
            # saving 2D images first means virtual memory can be saved. also acts as back-up in case program crashes
            filename = os.path.join(cleanPath, str(sampleNo).zfill(4) + '_clean_' + str(current_slice).zfill(4))
            saveTiff(target, filename)  # each image will only be normalized w.r.t itself
            for s in range(len(inpts)):
                filename = os.path.join(stripePaths[s], str(sampleNo).zfill(4) + '_shift' + str(s).zfill(2) + '_' +
                                        str(current_slice).zfill(4))
                saveTiff(inpts[s], filename)
        print(f"Chunk {i+1} saved to '{tiff_root}'")
    # Load 3D images, clip & normalize, then re-save to disk
    print("Loading full 3D dataset and normalizing...")
    target_file = os.path.join(cleanPath, str(sampleNo).zfill(4) + '_clean')
    target3D = load3DTiff(target_file, (len(ds), 402, 362))
    # clip so norm isn't skewed by anomalies in very low & very high slices
    target3D = np.clip(target3D, target3D[20:-20].min(), target3D[20:-20].max())
    save3DTiff(target3D, target_file, normalise=True)  # each image will be normalized w.r.t. the whole 3D sample
    # now do the same for each shift
    for s in range(num_shifts):
        inpt_file = os.path.join(stripePaths[s], str(sampleNo).zfill(4) + '_shift' + str(s).zfill(2))
        inpt3D = load3DTiff(inpt_file, (len(ds), 402, 362))
        inpt3D = np.clip(inpt3D, inpt3D[20:-20].min(), inpt3D[20:-20].max())
        # match histogram of input to target
        for sino in range(inpt3D.shape[0]):
            inpt3D[sino] = match_histograms(inpt3D[sino], target3D[sino])
        save3DTiff(inpt3D, inpt_file, normalise=True)
    print(f"Full normalized dataset saved to '{tiff_root}'")


def saveRawData(tiff_root, hdf_root, pipeline, no_slices=243, sampleNo=0, num_shifts=20, **kwargs):
    """Function to save the raw data from an HDF file.
    Doesn't apply any processing methods (other than re-sizing & normalizing)."""
    fileNum = int(os.path.basename(hdf_root).split('.')[0])
    # Create Dataset
    ds = RealDataset(hdf_root, pipeline, num_shifts=num_shifts, **kwargs)
    for s in range(num_shifts):
        print(f"Loading shift {s}/{num_shifts-1}...")
        currentFile = os.path.join(os.path.dirname(hdf_root), f'{fileNum + s}.nxs')
        ds.setHDFFile(currentFile)
        savePath = os.path.join(tiff_root, f'shift{s:02}')
        # Create array of min/max values for each slice (for normalizing later)
        minmax = np.ndarray((len(ds), 2))
        # Get tomogram in chunks
        num_chunks = math.ceil(len(ds) / no_slices)
        for c in range(num_chunks):
            print(f"\tLoading chunk {c+1}/{num_chunks}...")
            # Get current chunk of sinograms
            chunk = ds[:, c*no_slices:(c+1)*no_slices, :]  # shape (1801, <no_slices>, 2560)
            # swap axes
            chunk = np.swapaxes(chunk, 0, 1)  # shape (<no_slices>, 1801, 2560)
            # re-size chunk to shape (<no_slices>, 402, 362)
            if chunk.size == 0:  # if chunk is empty, move on to next iteration
                continue
            else:
                chunk = resize(chunk, (chunk.shape[0], 402, 362), anti_aliasing=True, preserve_range=True)
            # Loop through each sinogram in chunk & save to disk
            for sino in range(chunk.shape[0]):
                currentSlice = c * no_slices + sino
                # store min/max values of current slice (for un-scaling later)
                minmax[currentSlice, 0] = np.nanmin(chunk[sino])
                minmax[currentSlice, 1] = np.nanmax(chunk[sino])
                # Save current slice
                filename = f'{sampleNo:04}_shift{s:02}_{currentSlice:04}'
                saveTiff(chunk[sino], os.path.join(savePath, filename), normalise=True)
            print(f"\tChunk {c+1} saved to {savePath}.")
        print(f"\tAll chunks loaded, now re-loading and normalizing 3D sample...")
        # Once entire shift has been saved, re-load, un-scale and normalise again w.r.t whole 3D sample
        sino3D = np.ndarray((len(ds), 402, 362))
        # Load each slice
        for slc in range(len(ds)):
            filename = f'{sampleNo:04}_shift{s:02}_{slc:04}'
            sino = loadTiff(os.path.join(savePath, filename), normalise=True)
            # Un-scale sinogram w.r.t. min/max from before it was rescaled
            # (This retrieves the original range of the sinogram)
            sino = rescale(sino, a=minmax[slc, 0], b=minmax[slc, 1])
            sino3D[slc] = sino
        print(f"\t3D Sample loaded and un-scaled.")
        # Clip `sino3D` so that rescale is not skewed by outliers
        sino3D = np.clip(sino3D, sino3D[20:-20].min(), sino3D[20:-20].max())
        # Save & normalise again, this time w.r.t whole 3D sample
        filename = f'{sampleNo:04}_shift{s:02}'
        save3DTiff(sino3D, os.path.join(savePath, filename), normalise=True)
        print(f"\t3D Sample re-normalized and saved.")
        print(f"Shift {s} done.")
