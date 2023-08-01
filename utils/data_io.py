import numpy as np
import pickle
from PIL import Image

from utils.misc import rescale


def savePickle(data, path):
    """Save a pickle file.
    Parameters:
        data : object
            Data to save.
        path : str
            File path to save data to. Must include file extension, e.g.:
                '/path/to/data.p'
    """
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.DEFAULT_PROTOCOL)
    print(f"Saved data to {path}.")


def loadPickle(path):
    """Load a pickle file.
    Parameters:
        path : str
            File path to save data to. Must include file extension, e.g.:
                '/path/to/data.p'
    Returns:
        data : object
            Object loaded from pickle file.
    """
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def saveTiff(data, path, dtype=np.uint16, normalise=True):
    """Save 2D image data as a tiff.
    By default, images are rescaled to range [0, 65535] and saved with 16
    bits per pixel.
    Parameters:
        data : np.ndarray
            The 2D image to save
        path : str
            The filepath where the image will be saved. If it doesn't have a
            file extension, one will be added.
        dtype : np.dtype
            The datatype to convert the data to before saving.
            Default is np.uint16
        normalise : bool
            Whether to normalise the data before saving.
            Default is True.
    """
    if normalise:
        data = rescale(data)
        if dtype == np.uint16:
            data *= 65535
    img = Image.fromarray(data.astype(dtype))
    if not path.endswith('.tif'):
        path += '.tif'
    img.save(path)


def save3DTiff(data, path, dtype=np.uint16, normalise=True):
    """Save 3D image as a series of 2D slices.
    Image data must be in form (z, y, x) where slices are selected along axis z
    Slices are all saved in the same directory, with names like:
        '<path>_0000.tif', '<path>_0001.tif', ..., '<path>_9999.tif'
    where the number after <path> identifies the index of the slice of the 3D
    image.
    By default, images are rescaled to range [0, 65535] and saved with 16
    bits per pixel.
    Parameters:
        data : np.ndarray
            The 3D image to save.
        path : str
            The filepath where the image will be saved. Should not contain a
            file extension, otherwise you will get names like this:
                '/path/to/image.tif_0000.tif'
        dtype : np.dtype
            The datatype to convert the data to before saving.
            Default is np.uint16
        normalise : bool
            Whether to normalise the data before saving. Slices are normalised
            w.r.t the entire 3D image.
            Default is True.
    """
    if normalise:
        data = rescale(np.abs(data))
        if dtype == np.uint16:
            data *= 65535
    for z in range(data.shape[0]):
        filename = path + '_' + str(z).zfill(4)
        saveTiff(data[z, ...], filename, dtype=dtype, normalise=False)
    print(f"Saved .tif image to {path}.")


def loadTiff(path, dtype=np.uint16, normalise=True):
    """Load tiff image from disk.
    Parameters:
        path : str
            The filepath to load the image from. If it does not contain a file
            extension, one will be added.
        dtype : np.dtype
            The datatype to convert the data to when loading. Should match the
            datatype it was saved with.
            Default is np.uint16
        normalise : bool
            Whether to normalise the data after loading. Data is rescaled to
            range [0, 65535].
            Default is True.
    """
    if not path.endswith('.tif'):
        path += '.tif'
    img = Image.open(path)
    data = np.array(img, dtype=dtype)
    if normalise:
        data = rescale(data, imin=0, imax=65535)
    return data


def load3DTiff(path, shape, dtype=np.uint16, normalise=True):
    """Load a 3D tiff image from disk. Assumes images were saved according to
    the function `save3DTiff`.
    In other words, 2D slices must all be located in the same directory, with
    names like:
        '<path>_0000.tif', '<path>_0001.tif', ..., '<path>_9999.tif'
    where the number after <path> indicates the index of the slice of the 3D
    image.
    Parameters:
        path : str
            The filepath to load the image from. Should not contain a file
            extension, otherwise loading will fail.
        shape : Tuple[int, int, int]
            Shape of data to load from disk. Should contain 3 integers in form
            (z, y, x) i.e. (depth, height, width).
        dtype : np.dtype
            The datatype to convert the data to when loading. Should match the
            datatype it was saved with.
            Default is np.uint16
        normalise : bool
            Whether to normalise the data after loading. Data is rescaled to
            range [0, 1] w.r.t the entire 3D image.
            Default is True.
    """
    data = np.empty(shape)
    for z in range(shape[0]):
        filename = path + '_' + str(z).zfill(4) + '.tif'
        data[z, ...] = loadTiff(filename, dtype=dtype, normalise=False)
    if normalise:
        data = rescale(data, 0, 1)
    return data
