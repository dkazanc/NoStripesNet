import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from utils.data_io import loadTiff
from utils.stripe_detection import getMask_functional
from pathlib import Path


def getPairedFilepaths(root):
    """Given the root of a dataset, return the list of target/input pairs for
    each item in the dataset.
    Only works for simple, complex, and paired data.
    Returns empty list for dynamic and raw data. This is because these modes
    have no target images.
    Parameters:
        root : str
            Path to root of dataset.
    Returns:
        List[Tuple[str, str]]
            List of pairs of target & input paths.
    """
    target_filepaths = []
    input_filepaths = []
    for root, sub_dirs, images in os.walk(root):
        sub_dirs[:] = sorted(sub_dirs)
        if 'real_artifacts' in root:
            sub_dirs[:] = []
        if 'clean' in sub_dirs:
            clean_path = Path(root) / 'clean'
            clean_images = sorted(os.listdir(clean_path))
            if 'shift00' in sub_dirs:
                del sub_dirs[0]  # relies on sub_dirs being sorted
                for s in range(len(sub_dirs)):
                    target_filepaths.extend([str(clean_path/i)
                                             for i in clean_images])
            elif 'stripe' in sub_dirs:
                # os.walk() takes forever if there's a large number of files
                # so instead we prune sub_dirs and run os.listdir()
                sub_dirs[:] = []
                target_filepaths.extend([str(clean_path/i)
                                         for i in clean_images])
                stripe_path = Path(root)/'stripe'
                stripe_images = sorted(os.listdir(stripe_path))
                input_filepaths.extend([str(stripe_path/i)
                                        for i in stripe_images])
        if root.endswith('shift', 0, -2):
            input_filepaths.extend([str(Path(root)/i)
                                    for i in sorted(images)])
    return list(zip(target_filepaths, input_filepaths))


class BaseDataset(Dataset):
    """Dataset class to load tiff images from disk and return input/target
    pairs.
    Assumes data was generated according to the scripts in ../simulators.
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
        # Create list of all target/input image pairs
        self.all_filepaths = getPairedFilepaths(root)
        rng = np.random.default_rng(0)
        rng.shuffle(self.all_filepaths)
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


class MaskedDataset(BaseDataset):
    """Dataset class to load tiff images from disk and return a thruple of:
        (clean sinogram, stripey sinogram, mask)
    Can be instantiated in two modes:
        simple=False
            Creates masks using only the stripey sinogram
        simple=True
            Creates masks using both clean & stripey sinograms
    Assumes data was generated according to the scripts in ../simulators.
    """
    def __init__(self, root, mode, tvt, transform=None,
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
            transform : torch.nn.Module
                Transformations to apply to data.
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
            simple : bool
                Whether to use calculate masks using clean and stripe
                sinograms. Default is False.
        """
        super().__init__(root, mode, tvt, transform=transform)
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
