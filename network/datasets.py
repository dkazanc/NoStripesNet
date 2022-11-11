import sys
sys.path.append('..')  # crap solution to an even worse problem
import os
import itertools
import functools
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from scipy.ndimage import uniform_filter1d
from simulator.data_io import loadTiff
from metrics import *


class BaseDataset(Dataset):
    """Dataset to get sinograms from directories (provided data was generated according to the scripts in ../simulators)
    Data is split into train:validate:test according to parameter `tvt`, a tuple of the ratios"""

    def __init__(self, root, mode, tvt, size=256, shifts=5, transform=None):
        self.root = root
        self.size = size
        self.shifts = shifts

        # Create list of all datasets
        # and lists for train/test/validate datasets
        self.all_datasets = os.listdir(self.root)
        num_train = int((tvt[0] / sum(tvt)) * len(self.all_datasets))
        num_validate = int((tvt[1] / sum(tvt)) * len(self.all_datasets))
        num_test = int((tvt[2] / sum(tvt)) * len(self.all_datasets))
        self.train_datasets = self.all_datasets[:num_train]
        self.validate_datasets = self.all_datasets[num_train:num_train + num_validate]
        self.test_datasets = self.all_datasets[-num_test:]
        # Set current dataset to the dataset that corresponds to mode
        self.mode, self.datasets, self.filepaths = None, [], []
        self.setMode(mode)

        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, item):
        """For each item in a dataset, return the clean sinogram and all its associated shifts"""
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
        """Function to generate list of filenames in order of selection.
        Should be called everytime mode gets changed."""
        self.filepaths = []
        for dataset in self.datasets:
            # for each slice in a sample, append list containg locations of clean & shifts
            for i in range(self.size):
                files = []
                fileString = dataset + '_clean_' + str(i).zfill(4) + '.tif'
                filePath = os.path.join(self.root, dataset, 'clean', fileString)
                files.append(filePath)
                # loop through shifts
                for shift in range(self.shifts):
                    shiftNo = 'shift' + str(shift).zfill(2)
                    fileString = dataset + '_' + shiftNo + '_' + str(i).zfill(4) + '.tif'
                    filePath = os.path.join(self.root, dataset, shiftNo, fileString)
                    files.append(filePath)
                self.filepaths.append(files)


class WindowDataset(BaseDataset):
    def __init__(self, root, mode, tvt, windowWidth, size=256, shifts=5, transform=None):
        super().__init__(root, mode, tvt, size=size, shifts=shifts, transform=transform)
        self.windowWidth = windowWidth

    def getWindows(self, image):
        num_windows = image.shape[-1] // self.windowWidth
        if image.shape[-1] % self.windowWidth == 0:
            num_windows -= 1
        windows = []
        for i in range(num_windows+1):
            try:
                window = image[..., i * self.windowWidth:(i + 1) * self.windowWidth]
            except IndexError:
                window = image[..., i * self.windowWidth:]
            windows.append(window)
        return windows

    def __getitem__(self, item):
        clean, *shifts = super().__getitem__(item)
        clean = self.getWindows(clean)
        for i in range(len(shifts)):
            shifts[i] = self.getWindows(shifts[i])

        return clean, *shifts

    def getFullSinograms(self, item):
        return super().__getitem__(item)

    @staticmethod
    def combineWindows(window_list):
        if isinstance(window_list[0], torch.Tensor):
            return torch.cat(window_list, dim=-1)
        elif isinstance(window_list[0], np.ndarray):
            return np.concatenate(window_list, axis=-1)
        elif isinstance(window_list[0], list):
            return list(itertools.chain.from_iterable(window_list))


class PairedWindowDataset(WindowDataset):
    def __init__(self, root, mode, tvt, windowWidth, size=256, shifts=5, stripeMetric=gradient_sum_tv, transform=None):
        super().__init__(root, mode, tvt, windowWidth, size=size, shifts=shifts, transform=transform)
        self.metric = stripeMetric
        self.num_pairs = self.shifts // 2

    def __len__(self):
        return super().__len__() * self.num_pairs

    def __getitem__(self, item):
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
        clean, stripe, plain = self.__getitem__(item)
        clean_full = self.combineWindows(clean)
        stripe_full = self.combineWindows(stripe)
        plain_full = self.combineWindows(plain)
        return clean_full, stripe_full, plain_full


class PairedFullDataset(PairedWindowDataset):
    def __init__(self, root, mode, tvt, windowWidth, size=256, shifts=5, stripeMetric=gradient_sum_tv, transform=None):
        super().__init__(root, mode, tvt, windowWidth, size=size, shifts=shifts, stripeMetric=stripeMetric,
                         transform=transform)

    def __getitem__(self, item):
        clean, stripe, plain = super().__getitem__(item)
        clean_full = self.combineWindows(clean)
        stripe_full = self.combineWindows(stripe)
        plain_full = self.combineWindows(plain)
        return clean_full, stripe_full, plain_full


class MaskedDataset(BaseDataset):
    def __init__(self, root, mode, tvt, size=256, shifts=5, transform=None,
                 kernel_width=3, min_width=2, max_width=25, threshold=0.01):
        super().__init__(root, mode, tvt, size=size, shifts=shifts, transform=transform)
        self.k = 3
        self.min_width = min_width
        self.max_width = max_width
        self.eta = threshold

    def __len__(self):
        return super().__len__() * self.shifts

    def __getitem__(self, item):
        clean, *shifts = super().__getitem__(item // self.shifts)
        shift_idx = item % self.shifts
        stripe = shifts[shift_idx]
        mask = self.getMask(stripe)
        return clean, stripe, mask

    def getMask(self, sinogram):
        if isinstance(sinogram, np.ndarray):
            mean = functools.partial(np.mean, axis=-2)
            abs = np.abs
            mask = np.zeros_like(sinogram, dtype=np.bool_)
        elif isinstance(sinogram, torch.Tensor):
            mean = functools.partial(torch.mean, dim=-2)
            abs = torch.abs
            mask = torch.zeros_like(sinogram, dtype=torch.bool)
        else:
            raise TypeError(f"Expected type {np.ndarray} or {torch.Tensor}. Instead got {type(sinogram)}")

        # calculate mean curve, smoothed mean curve, and difference between the two
        mean_curve = mean(sinogram)
        mean_curve = normalise(mean_curve)
        smooth_curve = uniform_filter1d(mean_curve, size=5)
        if isinstance(sinogram, torch.Tensor):
            smooth_curve = torch.tensor(smooth_curve)
        diff_curve = abs(smooth_curve - mean_curve).squeeze()
        mask[..., diff_curve > self.eta] = True

        convolutions = np.lib.stride_tricks.sliding_window_view(mask.squeeze(), (sinogram.shape[-2], self.k)).squeeze()
        for i, conv in enumerate(convolutions):
            if conv[0, 0] and conv[0, -1]:
                mask[..., i:i + self.k] = True

        # get thickness of stripes in mask
        # if thickness is within certain threshold, remove stripe
        in_stripe = False
        for i in range(mask.shape[-1]):
            if mask[..., 0, i] and not in_stripe:
                start = i
                in_stripe = True
            if not mask[..., 0, i] and in_stripe:
                stop = i
                in_stripe = False
                width = stop - start
                if width < self.min_width or width > self.max_width:
                    mask[..., start:stop] = 0
        return mask


class RandomSubset(Subset):
    def __init__(self, dataset, length):
        self.length = length
        indices = random.sample(range(0, len(dataset)), self.length)
        super().__init__(dataset, indices)

    def __len__(self):
        return self.length

    def setMode(self, mode):
        self.dataset.setMode(mode)
        indices = random.sample(range(0, len(self.dataset)), self.length)
        super().__init__(self.dataset, indices)
