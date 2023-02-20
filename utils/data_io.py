import numpy as np
import pickle
from PIL import Image
from mpi4py import MPI
from tomopy import normalize, minus_log
from httomo.data.hdf.loaders import standard_tomo
from httomo.utils import _parse_preview
from httomo.data.hdf._utils.load import get_slice_list_from_preview


def rescale(data, a=0, b=1, imin=None, imax=None):
    """Function to normalise data in range [a, b].
    Had to call it 'rescale' because Python got confused with other functions' parameters."""
    data = data.astype(np.float32, copy=False)
    if imin is None:
        imin = data.min()
    if imax is None:
        imax = data.max()
    # if imin == imax, then the data is a constant value, and so normalising will have no effect
    # this also avoids a Divide By Zero error
    if imin == imax:
        return data
    out = a + ((data - imin)*(b - a)) / (imax - imin)
    return out


def savePickle(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.DEFAULT_PROTOCOL)
    print(f"Saved data to {path}.")


def loadPickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def saveTiff(data, path, dtype=np.uint16, normalise=True):
    """Function to save 2D image data as a Tiff file. Default datatype is `np.uint16`.
    This means by default images are saved with 16 bits per pixel."""
    if normalise:
        data = rescale(data)
        if dtype == np.uint16:
            data *= 65535
    img = Image.fromarray(data.astype(dtype))
    if not path.endswith('.tif'):
        path += '.tif'
    img.save(path)


def save3DTiff(data, path, dtype=np.uint16, normalise=True):
    """Function that saves 3D images as a series of 2D axial slices.
    Image data must be in form (z, y, x) where slices are selected along axis z.
    Slices are all saved in the same directory, with names like '<path>_0000.tif'
    where the number after <path> identifies the slice of the 3D image."""
    if normalise:
        data = rescale(np.abs(data))
        if dtype == np.uint16:
            data *= 65535
    for z in range(data.shape[0]):
        filename = path + '_' + str(z).zfill(4)
        saveTiff(data[z, ...], filename, dtype=dtype, normalise=False)
    print(f"Saved .tif image to {path}.")


def loadTiff(path, dtype=np.uint16, normalise=True):
    if not path.endswith('.tif'):
        path += '.tif'
    img = Image.open(path)
    data = np.array(img, dtype=dtype)
    if normalise:
        data = rescale(data, imin=0, imax=65535)
    return data


def load3DTiff(path, shape, dtype=np.uint16, normalise=True):
    """Function that loads images in accordance with the convention in `save3DTiff()`
    i.e. 2D slices must be all located in one directory, with names like '<path>_0000.tif'
    where the number after <path> identifies the slice of the 3D image."""
    data = np.empty(shape)
    for z in range(shape[0]):
        filename = path + '_' + str(z).zfill(4) + '.tif'
        data[z, ...] = loadTiff(filename, dtype=dtype, normalise=False)
    if normalise:
        data = rescale(data, 0, 1)
    return data


def loadHDF(file, tomo_params, flats=None, darks=None, comm=MPI.COMM_WORLD, ncore=None):
    # load raw data
    data, maybe_flats, maybe_darks, angles, *shape = standard_tomo(tomo_params['name'],
                                                                   file,
                                                                   tomo_params['data_path'],
                                                                   tomo_params['image_key_path'],
                                                                   tomo_params['dimension'],
                                                                   tomo_params['preview'],
                                                                   tomo_params['pad'],
                                                                   comm)
    if flats is None and darks is None:
        if maybe_flats.size == 0 and maybe_darks.size == 0:
            raise RuntimeError("File contains no flat or dark field data, and no flat or dark fields were passed as "
                               "parameters. Data cannot be normalized.")
        else:
            flats = maybe_flats
            darks = maybe_darks
    else:
        # make sure flats & darks are cropped correctly
        slices = get_slice_list_from_preview(_parse_preview(tomo_params['preview'], shape, [0, shape[0] - 1]))
        if flats.shape[-2:] != data.shape[-2:]:
            flats = flats[tuple(slices)]
        if darks.shape[-2:] != data.shape[-2:]:
            darks = darks[tuple(slices)]
    # normalize with flats and darks
    data = normalize(data, flats, darks, ncore=ncore, cutoff=10)
    data = np.clip(data, 1e-09, 1)
    data = minus_log(data, ncore=ncore)
    return data, angles
