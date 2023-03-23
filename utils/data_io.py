import numpy as np
import pickle
from PIL import Image
from mpi4py import MPI
from tomopy import normalize, minus_log
from httomo.data.hdf.loaders import standard_tomo
from httomo.utils import _parse_preview
from httomo.data.hdf._utils.load import get_slice_list_from_preview


def rescale(data, a=0, b=1, imin=None, imax=None):
    """Rescale data to range [a, b] w.r.t. bounds [imin, imax].
    Parameters:
        data : np.ndarray
            The data to be rescaled
        a : float
            The minimum value of the range the data will be scaled to.
            Default is 0
        b : float
            The maximum value of the range the data will be scaled to.
            Default is 1
        imin :  float
            The lower bound with respect to which the data will be scaled.
            Default is minimum value of `data`.
        imax: float
            The upper bound with respect to which the data will be scaled.
            Default is maximum value of `data`.
    Returns:
        out : np.ndarray
            The rescaled data. Has the same size & shape as input `data`.
    """
    data = data.astype(np.float32, copy=False)
    if imin is None:
        imin = data.min()
    if imax is None:
        imax = data.max()
    # if imin == imax, then the data is a constant value,
    # and so normalising will have no effect.
    # this also avoids a Divide By Zero error.
    if imin == imax:
        return data
    out = a + ((data - imin)*(b - a)) / (imax - imin)
    return out


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


def loadHDF(file, tomo_params, flats=None, darks=None, comm=MPI.COMM_WORLD,
            ncore=None):
    """Load the data from an HDF file.
    Parameters:
        file : str
            Path to hdf file to load data from.
        tomo_params : dict
            Dictionary that contains loading parameters. Same convention
            as an HTTomo yaml pipeline file.
            Must contain the following keys:
                {'name', 'data_path', 'image_key_path', 'dimension', 'preview',
                'pad'}
        flats : np.ndarray
            Numpy array containing flat fields for the data.
            If none are provided, will try to load flats from the HDF file.
            If the HDF file contains no flats, an error will be raised.
            Default is None.
        darks : np.ndarray
            Numpy array containing dark fields for the data.
            If none are provided, will try to load flats from the HDF file.
            If the HDF file contains no flats, an error will be raised.
            Default is None.
        comm : MPI.Comm
            MPI Communicator for parallel execution.
            Default is MPI.COMM_WORLD
        ncore : int
            Number of cores that will be assigned to jobs.
            Default is None.
    """
    # Load raw data
    data, maybe_flats, maybe_darks, angles, *shape = standard_tomo(
        tomo_params['name'],
        file,
        tomo_params['data_path'],
        tomo_params['image_key_path'],
        tomo_params['dimension'],
        tomo_params['preview'],
        tomo_params['pad'],
        comm
    )
    # Process flats and darks
    if flats is None and darks is None:
        # If none were passed in, try and use those loaded from HDF file
        if maybe_flats.size == 0 and maybe_darks.size == 0:
            raise RuntimeError("File contains no flat or dark field data, "
                               "and no flat or dark fields were passed as "
                               "parameters. Data cannot be normalized.")
        else:
            flats = maybe_flats
            darks = maybe_darks
    else:
        # make sure flats & darks are cropped correctly
        preview = _parse_preview(tomo_params['preview'], shape,
                                 [0, shape[0] - 1])
        slices = get_slice_list_from_preview(preview)
        if flats.shape[-2:] != data.shape[-2:]:
            flats = flats[tuple(slices)]
        if darks.shape[-2:] != data.shape[-2:]:
            darks = darks[tuple(slices)]
    # Normalize raw data with flats and darks
    data = normalize(data, flats, darks, ncore=ncore, cutoff=10)
    data = np.clip(data, 1e-09, 1)
    data = minus_log(data, ncore=ncore)
    return data
