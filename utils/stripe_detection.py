import warnings
import numpy as np
import torch
from larix.methods.misc import STRIPES_DETECT, STRIPES_MERGE
from tomopy.prep.stripe import _detect_stripe
from tomopy import normalize, minus_log, recon as recon_fn
from scipy.ndimage import median_filter, uniform_filter1d, binary_dilation


def normalise(data):
    return (data - data.min()) / (data.max() - data.min())


def detect_stripe_vo(sinogram, drop_ratio=0.1, snr=3, filter_size=10):
    drop_ratio = np.clip(drop_ratio, 0.0, 0.8)
    (nrow, ncol) = sinogram.shape[-2:]
    ndrop = int(0.5 * drop_ratio * nrow)
    sinosort = np.sort(sinogram, axis=0)
    sinosmooth = median_filter(sinosort, (1, filter_size))
    list1 = np.mean(sinosort[ndrop:nrow - ndrop], axis=0)
    list2 = np.mean(sinosmooth[ndrop:nrow - ndrop], axis=0)
    listfact = np.divide(list1, list2,
                         out=np.ones_like(list1), where=list2 != 0)
    # Locate stripes
    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    # Make mask 2D
    mask = np.zeros_like(sinogram, dtype=np.bool_)
    mask[:, listmask.astype(np.bool_)] = 1
    return mask


def detect_stripe_mean(sinogram, eta=0.01, kernel_width=3, min_width=2, max_width=25):
    mask = np.zeros_like(sinogram, dtype=np.bool_)
    # calculate mean curve, smoothed mean curve, and difference between the two
    mean_curve = np.mean(sinogram, axis=-2)
    mean_curve = normalise(mean_curve)
    smooth_curve = uniform_filter1d(mean_curve, size=5)
    diff_curve = np.abs(smooth_curve - mean_curve)
    mask[..., diff_curve > eta] = True

    convolutions = np.lib.stride_tricks.sliding_window_view(mask, (sinogram.shape[-2], kernel_width)).squeeze()
    for i, conv in enumerate(convolutions):
        if conv[0, 0] and conv[0, -1]:
            mask[..., i:i + kernel_width] = True

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
            if width < min_width or width > max_width:
                mask[..., start:stop] = 0
    return mask


def detect_stripe_larix(sinogram, threshold=0.1):
    (stripe_weights, grad_stats) = STRIPES_DETECT(sinogram, search_window_dims=(1, 7, 1), vert_window_size=5,
                                                  gradient_gap=3)
    # threshold weights to get a initialisation of the mask
    mask = np.zeros_like(stripe_weights, dtype="uint8")
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    mask[stripe_weights > grad_stats[3] / threshold] = 1

    # merge edges that are close to each other
    mask = STRIPES_MERGE(np.ascontiguousarray(mask, dtype=np.uint8), stripe_width_max_perc=25, dilate=3)
    return mask.astype(np.bool_)


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
    convolutions = np.lib.stride_tricks.sliding_window_view(mask_sum, (sino_np.shape[-2], kernel_width+2)).squeeze()
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


##############################################
# Other less-useful stripe detection methods #
##############################################
def residual(data):
    return data - np.mean(data)


def zscore(data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return residual(data) / np.std(data)


def total_variation(data):
    return np.sum(np.abs(data[1:] - data[:-1]))


def total_variation2D(data):
    ytv = np.sum(np.abs(data[1:, ...] - data[:-1, ...]))
    xtv = np.sum(np.abs(data[..., 1:] - data[..., :-1]))
    return xtv + ytv


def gradient_tv(data):
    return total_variation2D(np.gradient(data, axis=1))


def sum_max(data):
    return np.max(np.sum(data, axis=0))


def gradient_sum_max(data):
    return sum_max(np.gradient(data, axis=1))


def gradient_sum_tv(data):
    grad_sum = np.sum(np.gradient(data, axis=1), axis=0)
    return total_variation(grad_sum)


def grad_sum_res_max(data):
    return np.max(residual(np.sum(np.gradient(data, axis=1), axis=0)))


def grad_sum_res(data):
    return residual(np.sum(np.gradient(data, axis=1), axis=0))


def grad_sum_z_max(data):
    return np.max(zscore(np.sum(np.gradient(data, axis=1), axis=0)))


stripe_detection_metrics = [sum_max, total_variation2D, gradient_sum_max, gradient_tv, gradient_sum_tv,
                            grad_sum_res_max, grad_sum_z_max]
