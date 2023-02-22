import numpy as np
import torch
import tomopy as tp
from httomo.data.hdf._utils import load
from httomo.data.hdf.loaders import _parse_preview
from mpi4py import MPI
from h5py import File
from tomobar.methodsDIR import RecToolsDIR
from .misc import Rescale


def reconstruct(sinogram, angles=None, comm=MPI.COMM_WORLD, ncore=None):
    """Reconstruct a sinogram. Reconstruction algorithm used is gridrec.
    Parameters:
        sinogram : nd.ndarray or torch.Tensor
            Sinogram to reconstruct.
        angles : np.ndarray
            Angles to reconstruct with. Default is evenly distributed radians
            between 0 and pi, with length the same as first dimension of
            sinogram.
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
        sino_np = Rescale(a=0, b=sino_np.max())(sino_np)
    # Find Centre of Rotation
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


def getFlatsDarks(file, tomo_params, shape=None, comm=MPI.COMM_WORLD):
    """Load flats and darks from an HDF file.
    Parameters:
        file : str
            Path to HDF file.
        tomo_params : dict
            Dictionary that contains loading parameters. Same convention
            as an HTTomo yaml pipeline file.
            Must contain the following keys:
                {'data_path', 'image_key_path', 'dimension', 'preview',
                'pad'}
            shape : Tuple
                Shape of the data. If not provided, will be retrieved from
                the HDF file.
        comm : MPI.Comm
            MPI Communicator for parallel execution.
            Default is MPI.COMM_WORLD
    """
    if shape is None:
        with File(file, "r", driver="mpio", comm=comm) as f:
            shape = f[tomo_params['data_path']].shape
    data_indices = load.get_data_indices(file, tomo_params['image_key_path'],
                                         comm=comm)
    darks, flats = load.get_darks_flats(
        file,
        tomo_params['data_path'],
        tomo_params['image_key_path'],
        tomo_params['dimension'],
        tomo_params['pad'],
        _parse_preview(tomo_params['preview'], shape, data_indices),
        comm
    )
    return np.asarray(flats), np.asarray(darks)


def getRectools2D(size, device='cpu'):
    """Get reconstruction tools from tomobar.
    Parameters:
        size : int
            Height of sinogram.
        device : str
            Device to use when reconstructing.
    """
    total_angles = int(0.5 * np.pi * size)
    angles = np.linspace(0, 179.9, total_angles, dtype='float32')
    angles_rad = angles * (np.pi / 180.0)
    p = int(np.sqrt(2) * size)
    rectools = RecToolsDIR(DetectorsDimH=p,
                           DetectorsDimV=None,
                           CenterRotOffset=0.0,
                           AnglesVec=angles_rad,
                           ObjSize=size,
                           device_projector=device)
    return rectools
