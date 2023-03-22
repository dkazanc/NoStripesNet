import os
import itertools
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from h5py import File
from mpi4py import MPI
import multiprocessing
from utils.data_io import loadTiff, loadHDF
from utils.tomography import getFlatsDarks
from utils.stripe_detection import gradient_sum_tv, getMask_functional


class NewDataset(Dataset):
    """Dataset class to load tiff images from disk and return input/target
    pairs.
    Assumes data was generated according to the scripts in ../simulators.
    Named "new" because it works for the two different directory structures:
        1.
            <root>/
                clean/
                    ....tif
                stripe/
                    ....tif
        2.
            <root>/
                <sample number>/
                    clean/
                        ....tif
                    shift00/
                        ....tif
                    shift01/
                        ....tif
                    ...
    """

    def __init__(self, root, mode, tvt, transform=None):
        """Parameters:
            root : str
                Path to dataset root
            mode : str
                Mode for which data should be loaded. Should be either 'train',
                'validate' or 'test'.
            tvt : Tuple[int, int, int]
                Train/validate/test split of data. Interpreted as a ratio,
                i.e. (3, 1, 1) is equivalent to train:validate:test ratio 3:1:1
            transform : torch.nn.Module
                Transformations to apply to data.
        """
        self.root = root
        self.tvt = tvt
        # Create list of all images
        self.all_filepaths = []
        # `root` will contain either './clean' and './stripe' or './<sample_no>'
        sub_dirs = sorted(os.listdir(root))
        # check for sample number
        if '0000' in sub_dirs:
            # Loop through each sub-directory
            for sub_dir in sub_dirs:
                # Get list of sub-sub-directories
                sub_dir_path = os.path.join(root, sub_dir)
                subsub_dirs = sorted(os.listdir(sub_dir_path))
                # Get list of paths to target images
                targets_path = os.path.join(sub_dir_path, subsub_dirs[0])
                targets = sorted(os.listdir(targets_path))
                targets = [os.path.join(targets_path, t) for t in targets]
                # Loop through each shift
                for shift in subsub_dirs[1:]:
                    # Get list of paths to input images
                    inpts_path = os.path.join(sub_dir_path, shift)
                    inpts = sorted(os.listdir(inpts_path))
                    inpts = [os.path.join(inpts_path, i) for i in inpts]
                    # Add pair (target_path, input_path) to list of all paths
                    self.all_filepaths.extend(list(zip(targets, inpts)))
        elif 'clean' in sub_dirs and 'stripe' in sub_dirs:
            target_paths = sorted(os.listdir(os.path.join(root, 'clean')))
            targets = [os.path.join(root, 'clean', t) for t in target_paths]
            inpt_paths = sorted(os.listdir(os.path.join(root, 'stripe')))
            inpts = [os.path.join(root, 'stripe', i) for i in inpt_paths]
            self.all_filepaths = list(zip(targets, inpts))
        else:
            raise RuntimeError(
                f"Unrecognized Directory Structure: '{root}/{sub_dirs}'"
            )
        # Set current dataset to the dataset that corresponds to mode
        self.mode, self.datasets, self.filepaths = None, [], []
        self.setMode(mode)
        self.transform = transform

    def __len__(self):
        """Return length of dataset."""
        return len(self.filepaths)

    def __getitem__(self, item):
        """For a given item in a dataset, return a pair containing the clean
        sinogram and the same sinogram with stripes.
        Parameters:
            item : int
                Index of sinogram to return
        """
        cleanPath, stripePath = self.filepaths[item]
        # Get clean & stripe from path
        clean = loadTiff(cleanPath, dtype=np.float32)
        stripe = loadTiff(stripePath, dtype=np.float32)
        # Apply any transformations
        if self.transform is not None:
            clean = self.transform(clean)
            stripe = self.transform(stripe)
        return clean, stripe

    def getAbsTVT(self, i):
        """Get the absolute number of train/validate/test data from the ratio.
        Parameters:
            i : int
                Index of `self.tvt`
        """
        return int((self.tvt[i] / sum(self.tvt)) * len(self.all_filepaths))

    def setMode(self, mode):
        """Set the mode of the dataset.
        Parameters:
             mode : str
                Mode to set the dataset to. Must be either 'train', 'validate'
                or 'test'.
        """
        num_train = self.getAbsTVT(0)
        num_vldt = self.getAbsTVT(1)
        num_test = self.getAbsTVT(2)
        if mode == "train":
            self.filepaths = self.all_filepaths[:num_train]
        elif mode == "validate":
            self.filepaths = self.all_filepaths[num_train:num_vldt + num_train]
        elif mode == "test":
            self.filepaths = self.all_filepaths[-num_test:]
        else:
            raise ValueError(
                f"Mode should be one of 'train', 'validate' or 'test'. "
                f"Instead got {mode}"
            )
        self.mode = mode


class BaseDataset(Dataset):
    """Dataset class to load tiff images from disk and return a clean image &
    its associated shifts.
    Assumes data was generated according to the scripts in ../simulators.
    Can only load data stored in the following structure:
        <root>/
            <sample number>/
                clean/
                    ....tif
                shift00/
                    ....tif
                shift01/
                    ....tif
                ...
    Due to this limitation, it is preferrable to use class NewDataset over
    this class.
    """

    def __init__(self, root, mode, tvt, size=256, shifts=5, transform=None):
        """Parameters:
            root : str
                Path to dataset root
            mode : str
                Mode for which data should be loaded. Should be either 'train',
                'validate' or 'test'.
            tvt : Tuple[int, int, int]
                Train/validate/test split of data. Interpreted as a ratio,
                i.e. (3, 1, 1) is equivalent to train:validate:test ratio 3:1:1
            size : int
                Number of sinograms per sample. Default is 256.
            shifts : int
                Number of shifts per sample. Default is 5.
            transform : torch.nn.Module
                Transformations to apply to data.
        """
        self.root = root
        self.size = size
        self.shifts = shifts

        # Create list of all datasets
        # and lists for train/test/validate datasets
        self.all_datasets = sorted(os.listdir(self.root))
        # if there is only one dataset, set train/validate/test all to that one
        # dataset.
        # This will cause overfitting, but we have no choice
        if len(self.all_datasets) == 1:
            self.train_datasets = self.all_datasets
            self.validate_datasets = self.all_datasets
            self.test_datasets = self.all_datasets
        else:
            num_train = int((tvt[0] / sum(tvt)) * len(self.all_datasets))
            num_validate = int((tvt[1] / sum(tvt)) * len(self.all_datasets))
            num_test = int((tvt[2] / sum(tvt)) * len(self.all_datasets))
            self.train_datasets = self.all_datasets[:num_train]
            self.validate_datasets = self.all_datasets[
                                     num_train:num_train + num_validate]
            self.test_datasets = self.all_datasets[-num_test:]
        # Set current dataset to the dataset that corresponds to mode
        self.mode, self.datasets, self.filepaths = None, [], []
        self.setMode(mode)

        self.transform = transform

    def __len__(self):
        """Return length of dataset"""
        return len(self.filepaths)

    def __getitem__(self, item):
        """For a given item in a dataset, return a tuple containing the clean
        sinogram and all its associated shifts.
        Parameters:
            item : int
                Index of sinogram to return
        """
        cleanPath, *shiftPaths = self.filepaths[item]

        # Get clean & shifts from path
        clean = loadTiff(cleanPath, dtype=np.float32)
        shifts = []
        for shiftPath in shiftPaths:
            shifts.append(loadTiff(shiftPath, dtype=np.float32))

        # Apply any transformations
        if self.transform:
            clean = self.transform(clean)
            for i in range(len(shifts)):
                shifts[i] = self.transform(shifts[i])
        return clean, *shifts

    def setMode(self, mode):
        """Set the mode of the dataset.
        Parameters:
             mode : str
                Mode to set the dataset to. Must be either 'train', 'validate'
                or 'test'.
        """
        if mode == "train":
            self.datasets = self.train_datasets
        elif mode == "validate":
            self.datasets = self.validate_datasets
        elif mode == "test":
            self.datasets = self.test_datasets
        else:
            raise ValueError(f"Mode should be one of 'train', 'validate' or 'test'. Instead got {mode}")
        self.mode = mode
        self.genFilepaths()

    def genFilepaths(self):
        """Generate list of filepaths corresponding to current mode.
        List is created in ascending order of sinogram index.
        Should be called everytime mode gets changed.
        """
        self.filepaths = []
        for dataset in self.datasets:
            dsPath = os.path.join(self.root, dataset)
            # for each slice in a sample, append list containg locations of
            # clean & shifts
            for i in range(self.size):
                files = []
                fileName = f'{dataset}_clean_{i:04}.tif'
                filePath = os.path.join(dsPath, 'clean', fileName)
                files.append(filePath)
                # loop through shifts
                for shift in range(self.shifts):
                    shiftNo = f'shift{shift:02}'
                    fileName = f'{dataset}_{shiftNo}_{i:04}.tif'
                    filePath = os.path.join(dsPath, shiftNo, fileName)
                    files.append(filePath)
                self.filepaths.append(files)


class WindowDataset(BaseDataset):
    """Dataset class to load tiff images from disk and return a list of a clean
    image & its associated shifts.
    Assumes data was generated according to the scripts in ../simulators.
    Training on windowed sinograms did not give great results, and so this
    class might be deprecated soon.
    """

    def __init__(self, root, mode, tvt, windowWidth, size=256, shifts=5,
                 transform=None):
        """Parameters:
            root : str
                Path to dataset root
            mode : str
                Mode for which data should be loaded. Should be either 'train',
                'validate' or 'test'.
            tvt : Tuple[int, int, int]
                Train/validate/test split of data. Interpreted as a ratio,
                i.e. (3, 1, 1) is equivalent to train:validate:test ratio 3:1:1
            windowWidth : int
                Width of windows into which to split images
            size : int
                Number of sinograms per sample. Default is 256.
            shifts : int
                Number of shifts per sample. Default is 5.
            transform : torch.nn.Module
                Transformations to apply to data.
        """
        super().__init__(root, mode, tvt, size=size, shifts=shifts,
                         transform=transform)
        self.windowWidth = windowWidth

    def getWindows(self, image):
        """Split an input image into windows of width `self.windowWidth`.
        Parameters:
            image : torch.Tensor or np.ndarray
                The image to split into windows.
        """
        num_windows = image.shape[-1] // self.windowWidth
        if image.shape[-1] % self.windowWidth == 0:
            num_windows -= 1
        windows = []
        for i in range(num_windows+1):
            try:
                window = image[...,
                         i * self.windowWidth:(i + 1) * self.windowWidth]
            except IndexError:
                window = image[..., i * self.windowWidth:]
            windows.append(window)
        return windows

    def __getitem__(self, item):
        """For a given item in a dataset, return a list of lists containing
        the windowed clean sinogram and all its associated shifts.
        Parameters:
            item : int
                Index of sinogram to return
        """
        clean, *shifts = super().__getitem__(item)
        clean = self.getWindows(clean)
        for i in range(len(shifts)):
            shifts[i] = self.getWindows(shifts[i])

        return clean, *shifts

    def getFullSinograms(self, item):
        """Get the full (non-windowed) image for a given item.
        Parameters:
            item : int
                Index of sinogram to return
        """
        return super().__getitem__(item)

    @staticmethod
    def combineWindows(window_list):
        """Combine windows together to form one full image.
        Parameters:
            window_list : List[torch.Tensor] or List[np.ndarray] or List[list]
                The list of windows to combine together.
        """
        if isinstance(window_list[0], torch.Tensor):
            return torch.cat(window_list, dim=-1)
        elif isinstance(window_list[0], np.ndarray):
            return np.concatenate(window_list, axis=-1)
        elif isinstance(window_list[0], list):
            return list(itertools.chain.from_iterable(window_list))


class PairedWindowDataset(WindowDataset):
    """Dataset class to load tiff images from disk and return a thruple of
    lists: (clean windows, windows with stripes, windows without stripes)
    Assumes data was generated according to the scripts in ../simulators.
    Training on windowed sinograms did not give great results, and so this
    class might be deprecated soon.
    """
    def __init__(self, root, mode, tvt, windowWidth, size=256, shifts=5,
                 stripeMetric=gradient_sum_tv, transform=None):
        """Parameters:
            root : str
                Path to dataset root
            mode : str
                Mode for which data should be loaded. Should be either 'train',
                'validate' or 'test'.
            tvt : Tuple[int, int, int]
                Train/validate/test split of data. Interpreted as a ratio,
                i.e. (3, 1, 1) is equivalent to train:validate:test ratio 3:1:1
            windowWidth : int
                Width of windows into which to split images
            size : int
                Number of sinograms per sample. Default is 256.
            shifts : int
                Number of shifts per sample. Default is 5.
            stripeMetric : function
                Metric function to assess how many stripes an image contains.
                Default is gradient_sum_tv()
                For a full list of functions, see ../utils/stripe_detection.py
            transform : torch.nn.Module
                Transformations to apply to data.
        """
        super().__init__(root, mode, tvt, windowWidth, size=size,
                         shifts=shifts, transform=transform)
        self.metric = stripeMetric
        self.num_pairs = self.shifts // 2

    def __len__(self):
        """Return length of dataset."""
        return super().__len__() * self.num_pairs

    def __getitem__(self, item):
        """For a given item in a dataset, return a a thruple of lists:
            (clean windows, windows with stripes, windows without stripes)
        Parameters:
            item : int
                Index of sinogram to return
        """
        pair_idx = item % self.num_pairs
        clean, *shifts = super().__getitem__(item // self.num_pairs)
        plain_windows, stripe_windows = [], []
        scores_dict = {}
        # loop through each window
        for w in range(len(clean)):
            # calculate metric for each shift
            for i in range(self.shifts):
                m = self.metric(shifts[i][w])
                scores_dict[m] = i
            # pair minimum score (plain) with maximum score (stripe)
            sorted_scores = sorted(scores_dict)
            plain_idx = scores_dict[sorted_scores[pair_idx]]  # min score
            stripe_idx = scores_dict[sorted_scores[-pair_idx-1]]  # max score
            plain_windows.append(shifts[plain_idx][w])
            stripe_windows.append(shifts[stripe_idx][w])
        return clean, stripe_windows, plain_windows

    def getFullSinograms(self, item):
        """Get the full (non-windowed) image for a given item.
        Parameters:
            item : int
                Index of sinogram to return
        """
        clean, stripe, plain = self.__getitem__(item)
        clean_full = self.combineWindows(clean)
        stripe_full = self.combineWindows(stripe)
        plain_full = self.combineWindows(plain)
        return clean_full, stripe_full, plain_full


class PairedFullDataset(PairedWindowDataset):
    """Dataset class to load tiff images from disk and return a thruple of
        images: (clean, with stripes, without stripes)
        Basically the same as PairedWindowDataset but combines windows before
        returning them.
        Assumes data was generated according to the scripts in ../simulators.
        Training on windowed sinograms did not give great results, and so this
        class might be deprecated soon.
        """
    def __init__(self, root, mode, tvt, windowWidth, size=256, shifts=5,
                 stripeMetric=gradient_sum_tv, transform=None):
        """Parameters:
            root : str
                Path to dataset root
            mode : str
                Mode for which data should be loaded. Should be either 'train',
                'validate' or 'test'.
            tvt : Tuple[int, int, int]
                Train/validate/test split of data. Interpreted as a ratio,
                i.e. (3, 1, 1) is equivalent to train:validate:test ratio 3:1:1
            windowWidth : int
                Width of windows into which to split images
            size : int
                Number of sinograms per sample. Default is 256.
            shifts : int
                Number of shifts per sample. Default is 5.
            stripeMetric : function
                Metric function to assess how many stripes an image contains.
                Default is gradient_sum_tv()
                For a full list of functions, see ../utils/stripe_detection.py
            transform : torch.nn.Module
                Transformations to apply to data.
        """
        super().__init__(root, mode, tvt, windowWidth, size=size,
                         shifts=shifts, stripeMetric=stripeMetric,
                         transform=transform)

    def __getitem__(self, item):
        """For a given item in a dataset, return a a thruple of images:
            (clean, with stripes, without stripes)
        Parameters:
            item : int
                Index of sinogram to return
        """
        clean, stripe, plain = super().__getitem__(item)
        clean_full = self.combineWindows(clean)
        stripe_full = self.combineWindows(stripe)
        plain_full = self.combineWindows(plain)
        return clean_full, stripe_full, plain_full


class MaskedDataset(NewDataset):
    """Dataset class to load tiff images from disk and return a thruple of:
        (clean sinogram, stripey sinogram, mask)
    Can be instantiated in two modes:
        simple=False
            Creates masks using only the stripey sinogram
        simple=True
            Creates masks using both clean & stripey sinograms
    Assumes data was generated according to the scripts in ../simulators.
    """
    def __init__(self, root, mode, tvt, size=256, shifts=5, transform=None,
                 kernel_width=3, min_width=2, max_width=25, threshold=0.01,
                 filter_size=10, simple=False):
        """Parameters:
            root : str
                Path to dataset root
            mode : str
                Mode for which data should be loaded. Should be either 'train',
                'validate' or 'test'.
            tvt : Tuple[int, int, int]
                Train/validate/test split of data. Interpreted as a ratio,
                i.e. (3, 1, 1) is equivalent to train:validate:test ratio 3:1:1
            size : int
                Number of sinograms per sample. Only exists to be compatible
                with BaseDataset, not needed with this class.
            shifts : int
                Number of shifts per sample. Only exists to be compatible
                with BaseDataset, not needed with this class.
            kernel_width : int
                Width of kernel used in stripe detection method
                Default is 3
            min_width : int
                Minimum width of stripes (for stripe detection method)
                Default is 2
            max_width : int
                Maximum width of stripes (for stripe detection method)
                Default is 25
            threshold : float
                Minimum intensity of stripe, relative to smoothed mean curve
                of image (for stripe detection method).
                Default is 0.01
            filter_size : int
                Size of median filter (for stripe detection method).
                Default is 10
            transform : torch.nn.Module
                Transformations to apply to data.
        """
        super().__init__(root, mode, tvt, transform=transform)
        self.size = size
        self.shifts = shifts
        self.k = kernel_width
        self.min_width = min_width
        self.max_width = max_width
        self.eta = threshold
        self.filter_size = filter_size
        self.simple = simple

    def __len__(self):
        """Return length of dataset."""
        return super().__len__()

    def __getitem__(self, item):
        """For a given item in a dataset, return a thruple containing the clean
        sinogram, sinogram with stripes, and the mask of the locations of those
        stripes.
        Parameters:
            item : int
                Index of sinogram to return
        """
        clean, stripe = super().__getitem__(item)
        if self.simple:
            mask = self.getSimpleMask(clean, stripe)
        else:
            mask = self.getMask(stripe)
        return clean, stripe, mask

    @staticmethod
    def getSimpleMask(clean, stripe):
        """Get binary mask of locations of stripes in a sinogram.
        Requires a 'clean' ground truth as a reference point, and so is not
        applicable in the real world.
        Parameters:
            clean : torch.Tensor or np.ndarray
                The input sinogram containing no stripes.
            stripe : torch.Tensor or np.ndarray
                The input sinogram containing stripes. Should be identical to
                `clean` apart from stripes.
        """
        mask = np.zeros_like(clean, dtype=np.bool_)
        diff = np.abs(clean - stripe)
        mask[diff > diff.min()] = 1

        # expand mask widths by one pixel
        zero_prepend = [0 for _ in range(mask.ndim - 1)]
        stripe_idxs = np.where(mask[zero_prepend, :] == 1)[0]
        for stripe_idx in stripe_idxs:
            mask[..., stripe_idx-1:stripe_idx+1] = 1

        if isinstance(clean, torch.Tensor):
            mask = torch.tensor(mask)
        return mask

    def getMask(self, sinogram):
        """Get binary mask of locations of stripes in a sinogram.
        Parameters:
            sinogram : torch.Tensor or np.ndarray
                The input sinogram containing stripes to be detected.
        """
        mask = getMask_functional(sinogram,
                                   kernel_width=self.k,
                                   min_width=self.min_width,
                                   max_width=self.max_width,
                                   threshold=self.eta,
                                   filter_size=self.filter_size)
        return mask


class RandomSubset(Subset):
    """Class to get random subsets of a dataset.
    Enables a user to train with any number of training images (less than the
    total length of the dataset), and still have the stochasticity required to
    train a model effectively.
    """
    def __init__(self, dataset, length):
        """Parameters:
            dataset : torch.utils.data.Dataset
                Dataset from which to create a subset
            length : int
                Length of subset
        """
        self.length = length
        indices = random.sample(range(0, len(dataset)), self.length)
        super().__init__(dataset, indices)

    def __len__(self):
        """Return length of dataset."""
        return self.length

    def setMode(self, mode):
        """Set the mode of the dataset.
        Parameters:
             mode : str
                Mode to set the dataset to. Must be either 'train', 'validate'
                or 'test'.
        """
        self.dataset.setMode(mode)
        indices = random.sample(range(0, len(self.dataset)), self.length)
        super().__init__(self.dataset, indices)
