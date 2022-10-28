# Module that holds all stripe detection metrics
import numpy as np
import warnings
import torch
import torch.nn as nn


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


def l1(data1, data2):
    return np.sum(np.abs(data1 - data2))


def l2(data1, data2):
    return np.linalg.norm(data1 - data2)


def sum_square_diff(data1, data2):
    return np.linalg.norm(data1 - data2) ** 2


def sum_diff_grad(data1, data2):
    """Sum the difference between the magnitude of the gradients of data1 and data2"""
    return np.sum([np.linalg.norm(np.gradient(data1, axis=1)), np.linalg.norm(np.gradient(data2, axis=1))])


def diceCoef(data1, data2):
    intersect = np.logical_and(data1, data2)
    total_pixels = data1.size + data2.size
    f1 = (2 * np.sum(intersect)) / total_pixels
    return f1


def IoU(data1, data2):
    intersect = np.logical_and(data1, data2)
    union = np.logical_or(data1, data2)
    iou = np.sum(intersect) / np.sum(union)
    return iou


def BCELoss(data1, data2):
    target = torch.tensor(data1)
    inp = torch.tensor(data2)
    out_tensor = nn.BCELoss(reduction='mean')(inp, target)
    return out_tensor.numpy()


def histogram_intersection(data1, data2, bins=10):
    hist1, _ = np.histogram(data1, bins=bins)
    hist2, _ = np.histogram(data2, bins=bins)
    return np.sum([min(h1, h2) for h1, h2 in zip(hist1, hist2)])


test_metrics = [l1, l2, sum_square_diff, sum_diff_grad, diceCoef, IoU, BCELoss, histogram_intersection]
