# Module that holds all stripe detection metrics
import numpy as np
import warnings
import torch
import torch.nn as nn
from scipy.ndimage import median_filter, uniform_filter1d, binary_dilation
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio
from tomopy.prep.stripe import _detect_stripe
from larix.methods.misc import STRIPES_DETECT, STRIPES_MERGE

# not a good place for this but don't know where else to put it
class Rescale(object):
    """Rescale the image in a sample to a given range."""

    def __init__(self, a=0, b=1, imin=None, imax=None):
        self.a = a
        self.b = b
        self.imin = imin
        self.imax = imax

    def __call__(self, data):
        if self.imin is None:
            tmp_min = data.min()
        else:
            tmp_min = self.imin
        if self.imax is None:
            tmp_max = data.max()
        else:
            tmp_max = self.imax
        # if imin == imax, then the data is a constant value, and so normalising will have no effect
        # this also avoids a Divide By Zero error
        if self.imin == self.imax:
            return data
        return self.a + ((data - tmp_min) * (self.b - self.a)) / (tmp_max - tmp_min)


def normalise(data):
    return (data - data.min()) / (data.max() - data.min())


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


stripe_detection_metrics = [sum_max, total_variation2D, gradient_sum_max, gradient_tv, gradient_sum_tv,
                            grad_sum_res_max, grad_sum_z_max]


def l1(data1, data2):
    return np.mean(np.abs(data1 - data2))


def l2(data1, data2):
    return np.linalg.norm(data1 - data2)


def mean_squared_error(data1, data2):
    return l2(data1, data2) ** 2


def sum_diff_grad(data1, data2):
    """Sum the difference between the magnitude of the gradients of data1 and data2"""
    return np.sum([np.linalg.norm(np.gradient(data1, axis=1)), np.linalg.norm(np.gradient(data2, axis=1))])


def diceCoef(data1, data2):
    intersect = np.count_nonzero(data1 == data2)
    total_pixels = data1.size + data2.size
    f1 = (2 * intersect) / total_pixels
    return f1


def IoU(data1, data2):
    intersect = np.count_nonzero(data1 == data2)
    union = data1.size + data2.size - intersect
    iou = intersect / union
    return iou


def histogram_intersection(data1, data2):
    bin_edges = np.histogram_bin_edges(np.concatenate([data1, data2]))
    h1, _ = np.histogram(data1, bins=bin_edges)
    h2, _ = np.histogram(data2, bins=bin_edges)
    return np.sum(np.minimum(h1, h2)) / data1.size


def structural_similarity(data1, data2):
    range = np.amax((data1, data2)) - np.amin((data1, data2))
    return ssim(data1, data2, data_range=range)


def psnr(data1, data2):
    range = np.amax((data1, data2)) - np.amin((data1, data2))
    return peak_signal_noise_ratio(data1, data2, data_range=range)


test_metrics = [l1, l2, mean_squared_error, sum_diff_grad, diceCoef, IoU, histogram_intersection, structural_similarity, psnr]
