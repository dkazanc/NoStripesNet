import os
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset
from simulator.data_io import loadTiff


class BaseDataset(Dataset):
    """Dataset to get sinograms from directories (provided data was generated according to the scripts in ../simulators)
    Data is split into train:validate:test according to parameter `tvt`, a tuple of the ratios"""

    def __init__(self, root, transform, mode, tvt, size=256, shifts=5):
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
    def __init__(self, root, transform, mode, tvt, windowWidth, size=256, shifts=5):
        super().__init__(root, transform, mode, tvt, size=size, shifts=shifts)
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
    def __init__(self, root, transform, mode, tvt, windowWidth, size=256, shifts=5):
        super().__init__(root, transform, mode, tvt, windowWidth, size=size, shifts=shifts)

    def __getitem__(self, item):
        clean, *shifts = super().__getitem__(item)
        centre = shifts[self.shifts // 2]

        # to get best of both, loop through each shift and use one with lowest tv
        bob = []
        # loop through each window
        for i in range(len(centre)):
            centre_tv = self.gradient_sum_tv(centre[i])
            min_tv = centre_tv
            min_idx = self.shifts // 2  # centre index
            # loop through each shift
            for j in range(self.shifts):
                shift_tv = self.gradient_sum_tv(shifts[j][i])
                if shift_tv < min_tv:
                    min_tv = shift_tv
                    min_idx = j
            bob.append(shifts[min_idx][i])

        # return (clean, centre, best-of-both)
        return clean, centre, bob

    def getFullSinograms(self, item):
        clean, centre, bob = self.__getitem__(item)
        clean_full = self.combineWindows(clean)
        centre_full = self.combineWindows(centre)
        bob_full = self.combineWindows(bob)
        return clean_full, centre_full, bob_full

    def gradient_sum_tv(self, data):
        grad_sum = np.sum(np.gradient(data, axis=1), axis=0)
        return self.total_variation(grad_sum)

    @staticmethod
    def total_variation(data):
        return np.sum(np.abs(data[1:] - data[:-1]))
