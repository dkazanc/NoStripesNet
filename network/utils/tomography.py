import sys
sys.path.append('..')
from metrics import detect_stripe_larix, detect_stripe_mean, detect_stripe_vo
import numpy as np
import torch
from tomopy import find_center_vo, scale, normalize, minus_log, recon as recon_fn
from httomo.data.hdf._utils import load
from httomo.data.hdf.loaders import standard_tomo, _parse_preview
from mpi4py import MPI
from h5py import File


def reconstruct(sinogram, cor_params, rec_params, comm=MPI.COMM_WORLD, ncore=None):
    if type(sinogram) == np.ndarray or type(sinogram) == torch.Tensor:
        sinogram = np.asarray(sinogram.squeeze())
        angles = np.linspace(0, np.pi, sinogram.shape[0])
        if sinogram.ndim == 2:
            sinogram = sinogram[:, None, :]
    else:
        raise TypeError(f"Type of item should be one of ['np.ndarray', 'torch.Tensor']. "
                        f"Instead got '{type(sinogram)}'")
    # Find Centre of Rotation
    rot_center = 0
    mid_rank = int(round(comm.size / 2) + 0.1)
    if comm.rank == mid_rank:
        mid_slice = int(np.size(sinogram, 1) / 2)
        rot_center = find_center_vo(sinogram,
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
    reconstruction = recon_fn(sinogram,
                              angles,
                              rot_center,
                              rec_params['sinogram_order'],
                              rec_params['algorithm'],
                              ncore=ncore)
    reconstruction = scale(reconstruction)[0]
    return reconstruction.squeeze(), sinogram.squeeze()


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


def loadHDF(file, tomo_params, flats=None, darks=None, comm=MPI.COMM_WORLD, ncore=None):
    # load raw data
    data, maybe_flats, maybe_darks, angles, *_ = standard_tomo(tomo_params['name'],
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
    # normalize with flats and darks
    data = normalize(data, flats, darks, ncore=ncore, cutoff=10)
    data[data == 0.0] = 1e-09
    data = minus_log(data, ncore=ncore)
    return data, angles


def getMask_functional(sinogram, kernel_width=3, min_width=2, max_width=25, threshold=0.01, filter_size=10):
    if isinstance(sinogram, torch.Tensor):
        sino_np = sinogram.detach().numpy().squeeze()
    else:
        sino_np = sinogram
    mask_vo = detect_stripe_vo(sino_np, filter_size=filter_size).astype(int)
    mask_mean = detect_stripe_mean(sino_np, eta=threshold, kernel_width=kernel_width, min_width=min_width,
                                   max_width=max_width).astype(int)
    mask_larix = detect_stripe_larix(sino_np).astype(int)
    mask_sum = mask_vo + mask_mean + mask_larix
    mask_sum[mask_sum < 2] = 0

    # if there is a 3 pixel gap or less between stripes, merge them
    convolutions = np.lib.stride_tricks.sliding_window_view(mask_sum, (sinogram.shape[-2], kernel_width+2)).squeeze()
    for i, conv in enumerate(convolutions):
        if conv[0, 0] and conv[0, -1]:
            mask_sum[..., i:i + kernel_width+2] = True

    if isinstance(sinogram, torch.Tensor):
        mask_sum = torch.tensor(mask_sum, dtype=torch.bool).unsqueeze(0)
    elif isinstance(sinogram, np.ndarray):
        mask_sum = mask_sum.astype(np.bool_)
    else:
        raise TypeError(f"Expected type {np.ndarray} or {torch.Tensor}. Instead got {type(sinogram)}")
    return mask_sum
