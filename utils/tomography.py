import numpy as np
import torch
import tomopy as tp
from mpi4py import MPI
from h5py import File
from .misc import rescale


def reconstruct(sinogram, angles=None, rot_center=None, comm=MPI.COMM_WORLD,
                ncore=None):
    """Reconstruct a sinogram. Reconstruction algorithm used is gridrec.
    Parameters:
        sinogram : nd.ndarray or torch.Tensor
            Sinogram to reconstruct.
        angles : np.ndarray
            Angles to reconstruct with. Default is evenly distributed radians
            between 0 and pi, with length the same as first dimension of
            sinogram.
        rot_center : float
            Center of rotation to use when reconstructing. Default is None.
            If not given, the center will be found automatically.
        comm : MPI.Comm
            MPI Communicator for parallel execution.
            Default is MPI.COMM_WORLD
        ncore : int
            Number of cores that will be assigned to jobs.
            Default is None.
    """
    if type(sinogram) == np.ndarray or type(sinogram) == torch.Tensor:
        sino_np = np.asarray(sinogram)
        if sino_np.ndim == 2:
            sino_np = sino_np[:, None, :]
        elif sino_np.ndim == 3:
            sino_np = np.swapaxes(sino_np, 0, 1)
    else:
        raise TypeError(f"Type of item should be one of "
                        f"['np.ndarray', 'torch.Tensor']. "
                        f"Instead got '{type(sinogram)}'")
    if angles is None:
        angles = tp.angles(sino_np.shape[0])
    if sino_np.min() < 0:
        sino_np = rescale(sino_np, a=0, b=sino_np.max())
    # Find Centre of Rotation
    if rot_center is None:
        rot_center = 0
        mid_rank = int(round(comm.size / 2) + 0.1)
        if comm.rank == mid_rank:
            mid_slice = int(np.size(sino_np, 1) / 2)
            rot_center = tp.find_center_vo(sino_np,
                                           mid_slice,
                                           smin=-50,
                                           smax=50,
                                           srad=6,
                                           step=0.25,
                                           ratio=0.5,
                                           drop=20,
                                           ncore=ncore)
        rot_center = comm.bcast(rot_center, root=mid_rank)
    # Reconstruct
    reconstruction = tp.recon(sino_np,
                              angles,
                              rot_center,
                              sinogram_order=False,
                              algorithm='gridrec',
                              ncore=ncore)
    reconstruction = tp.scale(reconstruction)[0]
    if sinogram.ndim == 2:
        reconstruction = reconstruction.squeeze()
    return reconstruction


class TomoH5:
    """Class to wrap around a .nxs file containing tomography scan data."""

    def __init__(self, nexus_file):
        """Parameters:
            nexus_file : str
                .nxs or .h5 file containing tomographic data.
        """
        self.file = File(nexus_file, 'r')
        self.data = self.file['entry1/tomo_entry/data/data']
        self.angles = self.file['entry1/tomo_entry/data/rotation_angle']
        image_key_path = 'entry1/tomo_entry/instrument/detector/image_key'
        if image_key_path in self.file:
            self.image_key = self.file[image_key_path]
        else:
            print("Warning: file contains no image key. Assuming no flats.")
            self.image_key = np.zeros(self.data.shape[0])
        self.data_indices = np.where(self.image_key[:] == 0)
        self.shape = self.data.shape
        self.flats = self.get_flats()
        self.darks = self.get_darks()

    def contains_flats(self):
        """Returns: bool
            Whether the HDF5 file contains flat fields.
        """
        return 1 in self.image_key

    def contains_darks(self):
        """Returns: bool
            Whether the HDF5 file contains dark fields.
        """
        return 2 in self.image_key

    def get_angles(self):
        """Returns: np.ndarray
            Array containing the angle of rotation for each projection.
        """
        return self.angles[self.data_indices]

    def get_flats(self):
        """Returns: np.ndarray
            Array containing the flat fields of the HDF5 file.
            If file contains no flat fields, None is returned.
        """
        if not self.contains_flats():
            return None
        return self.data[self.image_key[:] == 1]

    def get_darks(self):
        """Returns: np.ndarray
            Array containing the dark fields of the HDF5 file.
            If file contains no dark fields, None is returned.
        """
        if not self.contains_darks():
            return None
        return self.data[self.image_key[:] == 2]

    def _get_norm_from_idx(self, norm, idx):
        if type(idx) == int or type(idx) == slice:
            return norm
        elif type(idx) == tuple:
            return norm[:, idx[1], idx[2]]
        else:
            raise ValueError("Unrecognized index.")

    def get_normalized(self, item, flats=None, darks=None, ncore=None):
        """Get a slice of data and normalise it with flats and darks.
        Parameters:
            item : int or slice
                The index or slice of the data to retrieve from the HDF5 file.
            flats : np.ndarray
                Flat fields to use when normalising the data. Default is None.
                If not given, then flats from the HDF5 file will be used.
                If the HDF5 file contains no flats, an error is raised.
            darks : np.ndarray
                Dark fields to use when normalising the data. Default is None.
                If not given, then darks from the HDF5 file will be used.
                If the HDF5 file contains no darks, an error is raised.
            ncore : None
                Number of cores to use when normalising the data.
                Default is None.
        Returns:
            np.ndarray
                The normalised data.
        """
        if flats is None:
            if self.flats is None:
                raise ValueError(
                    "Self contains no flats, and none were passed in. "
                    "Please pass an ndarray containing flat fields.")
            else:
                flats = self.flats
        if darks is None:
            if self.darks is None:
                raise ValueError(
                    "Self contains no darks, and none were passed in. "
                    "Please pass an ndarray containing dark fields.")
            else:
                darks = self.darks
        # Make sure flats and darks are cropped correctly
        flats = self._get_norm_from_idx(flats, item)
        darks = self._get_norm_from_idx(darks, item)
        # Get data and remove any nan values
        raw = self[item]
        raw[np.isnan(raw)] = 0
        # Normalise with flats & darks
        norm = tp.normalize(raw, flats, darks, ncore=ncore)
        # Minus Log
        norm[norm <= 0] = 1e-9
        tp.minus_log(norm, out=norm, ncore=ncore)
        return norm

    def __len__(self):
        """Return the length of the data"""
        return self.shape[0]

    def __getitem__(self, item):
        """Get an un-normalised slice from the HDF5 data.
        Parameters:
            item : int or slice
                The index or slice of the data to retrieve from the HDF5 file.
        Returns:
            np.ndarray
                Raw data from the HDF5 file.
        """
        # Item should be adjusted so that flats & darks are ignored.
        if not (self.contains_flats() or self.contains_darks()):
            return self.data[item]
        # assumes data_indices is in ascending order
        lo = self.data_indices[0][0]
        hi = self.data_indices[0][-1]
        if type(item) == int:
            new_item = item + lo
        elif type(item) == slice:
            new_item = slice(item.start + lo if item.start is not None else lo,
                             item.stop + lo if item.stop is not None else None,
                             item.step)
        elif type(item) == tuple:
            if type(item[0]) == int:
                new_item = [item[0] + lo]
            elif type(item[0]) == slice:
                start = item[0].start + lo if item[0].start is not None else lo
                stop = item[0].stop + lo if item[0].stop is not None else None
                new_item = [slice(start,
                                  stop,
                                  item[0].step)]
            else:
                raise ValueError("Unrecognized index.")
            for i in range(1, len(item)):
                new_item.append(item[i])
            new_item = tuple(new_item)
        else:
            raise ValueError("Unrecognized index.")
        return self.data[new_item]
