# Module that holds all stripe detection metrics
import numpy as np
import warnings
import torch
import torch.nn as nn
from scipy.ndimage import median_filter, uniform_filter
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio


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


def mask_1d(data, R):
    # calculate low-pass component
    low_pass = uniform_filter(data, size=4)

    # calculate abs difference between intensity and low-pass COLUMN-WISE
    data_1d = np.mean(data - low_pass, axis=0)

    # apply median filter
    data_filter = median_filter(data_1d, size=12)

    # normalize by dividing sino with filtered sino
    data_norm = np.square(data_1d - data_filter)

    # locate stripe artifact using Nghia's method
    npoint = len(data_norm)
    list_sort = np.sort(data_norm)
    listx = np.arange(0, npoint, 1.0)
    ndrop = np.int16(0.25 * npoint)
    (slope, intercept) = np.polyfit(listx[ndrop:-ndrop - 1], list_sort[ndrop:-ndrop - 1], 1)
    y_end = intercept + slope * listx[-1]
    noise_level = np.abs(y_end - intercept)
    noise_level = np.clip(noise_level, 1e-6, None)
    val1 = np.abs(list_sort[-1] - y_end) / noise_level
    val2 = np.abs(intercept - list_sort[0]) / noise_level
    list_mask = np.zeros(npoint, dtype=np.float32)
    if val1 >= R:
        upper_thresh = y_end + noise_level * R * 0.5
        list_mask[data_norm > upper_thresh] = 1.0
    if val2 >= R:
        lower_thresh = intercept - noise_level * R * 0.5
        list_mask[data_norm <= lower_thresh] = 1.0
    return list_mask


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
    intersect = np.intersect1d(data1, data2)
    total_pixels = data1.size + data2.size
    f1 = (2 * intersect.size) / total_pixels
    return f1


def IoU(data1, data2):
    intersect = np.intersect1d(data1, data2)
    union = np.union1d(data1, data2)
    iou = intersect.size / union.size
    return iou


def histogram_intersection(data1, data2, bins=10):
    hist1, _ = np.histogram(data1, bins=bins)
    hist2, _ = np.histogram(data2, bins=bins)
    return np.mean([min(h1, h2) for h1, h2 in zip(hist1, hist2)])


def structural_similarity(data1, data2):
    range = np.amax((data1, data2)) - np.amin((data1, data2))
    return ssim(data1, data2, data_range=range)


def psnr(data1, data2):
    range = np.amax((data1, data2)) - np.amin((data1, data2))
    return peak_signal_noise_ratio(data1, data2, data_range=range)


test_metrics = [l1, l2, mean_squared_error, sum_diff_grad, diceCoef, IoU, histogram_intersection, structural_similarity, psnr]
