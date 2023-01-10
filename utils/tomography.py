import numpy as np
import torch
from tomopy import find_center_vo, scale, recon as recon_fn
from httomo.data.hdf._utils import load
from httomo.data.hdf.loaders import _parse_preview
from mpi4py import MPI
from h5py import File


def reconstruct(sinogram, cor_params, rec_params, comm=MPI.COMM_WORLD, ncore=None):
    if type(sinogram) == np.ndarray or type(sinogram) == torch.Tensor:
        sino_np = np.asarray(sinogram)
        if sino_np.ndim == 2:
            sino_np = sino_np[:, None, :]
        elif sino_np.ndim == 3:
            sino_np = np.swapaxes(sino_np, 0, 1)
        angles = np.linspace(0, np.pi, sino_np.shape[0])
    else:
        raise TypeError(f"Type of item should be one of ['np.ndarray', 'torch.Tensor']. "
                        f"Instead got '{type(sinogram)}'")
    # Find Centre of Rotation
    rot_center = 0
    mid_rank = int(round(comm.size / 2) + 0.1)
    if comm.rank == mid_rank:
        mid_slice = int(np.size(sino_np, 1) / 2)
        rot_center = find_center_vo(sino_np,
                                    mid_slice,
                                    cor_params['smin'],
                                    cor_params['smax'],
                                    cor_params['srad'],
                                    cor_params['step'],
                                    cor_params['ratio'],
                                    cor_params['drop'],
                                    ncore=ncore)
    rot_center = comm.bcast(rot_center, root=mid_rank)
    # Reconstruct
    reconstruction = recon_fn(sino_np,
                              angles,
                              rot_center,
                              rec_params['sinogram_order'],
                              rec_params['algorithm'],
                              ncore=ncore)
    reconstruction = scale(reconstruction)[0]
    return reconstruction


def getFlatsDarks(file, tomo_params, shape=None, comm=MPI.COMM_WORLD):
    """Load flats and darks."""
    if shape is None:
        with File(file, "r", driver="mpio", comm=comm) as file:
            shape = file[tomo_params['data_path']].shape
    data_indices = load.get_data_indices(file, tomo_params['image_key_path'], comm=comm)
    darks, flats = load.get_darks_flats(file,
                                        tomo_params['data_path'],
                                        tomo_params['image_key_path'],
                                        tomo_params['dimension'],
                                        tomo_params['pad'],
                                        _parse_preview(tomo_params['preview'], shape, data_indices),
                                        comm)
    return np.asarray(flats), np.asarray(darks)
