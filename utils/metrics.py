import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim, \
    peak_signal_noise_ratio

from .misc import toNumpy


def batch_metric(metric, data1_batch, data2_batch, reduction='mean'):
    """Apply a test metric to an entire batch of images.
    Parameters:
        metric : function
            Test metric to apply to the batch
        data1_batch : np.ndarray
            Batch of inputs to metric
        data2_batch : np.ndarray
            Batch of targets for metric
        reduction : str
            String specifying how the metrics for every image in the batch will
            be reduced to a single value.
            Must be either 'mean' or 'sum'. Default is 'mean'.
    Returns:
        float
            Reduction of `metric` applied to every image pair in batch.
    """
    metric_scores = []
    for data1, data2 in zip(data1_batch, data2_batch):
        metric_scores.append(metric(data1, data2))
    if reduction == 'mean':
        return np.mean(metric_scores)
    elif reduction == 'sum':
        return np.sum(metric_scores)
    else:
        raise ValueError(f"Reduction should be 'mean' or 'sum'. "
                         f"Instead got '{reduction}'.")


def apply_metrics(data1, data2, metrics):
    """
    Apply a list of metrics to two batches of data.
    Parameters:
        data1 : torch.Tensor or np.ndarray
            Batch of input images to calculate metric on.
        data2 : torch.Tensor or np.ndarray
            Batch of target images to calculate metric on.
        metrics : List[function]
            List of test metrics to apply on the two input batches.
    Returns:
        dict
            Dictionary of same length as `metrics`. The keys are the names of
            each metric, and the values are the result of each metric applied
            to the pair of batches `data1` and `data2`.
    """
    metric_scores = {}
    # All metric functions are vectorized, so if both input batches are tensors
    # we can directly apply each metric function.
    if type(data1) == torch.Tensor and type(data2) == torch.Tensor:
        for metric in metrics:
            metric_scores[metric.__name__] = metric(data1, data2)
    else:
        # Otherwise, we have to call `batch_metric`
        for metric in metrics:
            metric_scores[metric.__name__] = batch_metric(metric,
                                                          toNumpy(data1),
                                                          toNumpy(data2))
    return metric_scores


def l1(data1, data2):
    """Calculate the L1 score (mean absolute error) for two batches.
    Parameters:
        data1 : torch.Tensor or np.ndarray
            Input batch
        data2 : torch.Tensor or np.ndarray
            Target batch
    Returns:
        np.ndarray
            L1 score of `data1` and `data2`
    """
    if type(data1) == torch.Tensor and type(data2) == torch.Tensor:
        return toNumpy(nn.functional.l1_loss(data1, data2))
    else:
        return np.mean(np.abs(data1 - data2))


def l2(data1, data2):
    """Calculate the L2 norm for two batches.
    Parameters:
        data1 : torch.Tensor or np.ndarray
            Input batch
        data2 : torch.Tensor or np.ndarray
            Target batch
    Returns:
        np.ndarray
            L2 norm of `data1` and `data2`
    """
    if type(data1) == torch.Tensor and type(data2) == torch.Tensor:
        return toNumpy(torch.mean(torch.linalg.matrix_norm(data1 - data2)))
    else:
        return np.linalg.norm(data1 - data2)


def mean_squared_error(data1, data2):
    """Calculate the Mean Squared Error for two batches.
    Parameters:
        data1 : torch.Tensor or np.ndarray
            Input batch
        data2 : torch.Tensor or np.ndarray
            Target batch
    Returns:
        np.ndarray
            MSE norm of `data1` and `data2`
    """
    if type(data1) == torch.Tensor and type(data2) == torch.Tensor:
        return toNumpy(nn.functional.mse_loss(data1, data2))
    else:
        return np.mean(np.square(data1 - data2))


def sum_diff_grad(data1, data2):
    """Sum the difference between the magnitude of the gradients of two batches
    Parameters:
        data1 : torch.Tensor or np.ndarray
            Input batch
        data2 : torch.Tensor or np.ndarray
            Target batch
    Returns:
        np.ndarray
            Sum of the difference between the gradients of `data1` and `data2`
    """
    if type(data1) == torch.Tensor and type(data2) == torch.Tensor:
        return toNumpy(torch.mean(torch.sum(torch.abs(
            torch.gradient(data1, dim=-1)[0] -
            torch.gradient(data2, dim=-1)[0]),
            dim=[-1, -2])))
    else:
        return np.sum(np.abs(np.gradient(data1, axis=1) -
                             np.gradient(data2, axis=1)))


def diceCoef(data1, data2):
    """Calculate the Dice Coefficient (or F1 score) between two batches.
    Parameters:
        data1 : torch.Tensor or np.ndarray
            Input batch
        data2 : torch.Tensor or np.ndarray
            Target batch
    Returns:
        np.ndarray
            Dice Coefficient of `data1` and `data2`
    """
    if type(data1) == torch.Tensor and type(data2) == torch.Tensor:
        intersect = torch.count_nonzero(data1 == data2, dim=(-1, -2))
        total_pixels = data1.shape[-1] * data1.shape[-2] + \
                       data2.shape[-1] * data2.shape[-2]
        return toNumpy(torch.mean((2 * intersect) / total_pixels))
    else:
        intersect = np.count_nonzero(data1 == data2)
        total_pixels = data1.size + data2.size
        f1 = (2 * intersect) / total_pixels
        return f1


def IoU(data1, data2):
    """Calculate the Intersection over Union between two batches.
    Parameters:
        data1 : torch.Tensor or np.ndarray
            Input batch
        data2 : torch.Tensor or np.ndarray
            Target batch
    Returns:
        np.ndarray
            IoU of `data1` and `data2`
    """
    if type(data1) == torch.Tensor and type(data2) == torch.Tensor:
        intersect = torch.count_nonzero(data1 == data2, dim=(-1, -2))
        union = data1.shape[-1] * data1.shape[-2] + \
                data2.shape[-1] * data2.shape[-2] - intersect
        return toNumpy(torch.mean(intersect / union))
    else:
        intersect = np.count_nonzero(data1 == data2)
        union = data1.size + data2.size - intersect
        iou = intersect / union
        return iou


def histogram_intersection(data1, data2):
    """Calculate the Histogram Intersection between two batches.
    Parameters:
        data1 : torch.Tensor or np.ndarray
            Input batch
        data2 : torch.Tensor or np.ndarray
            Target batch
    Returns:
        np.ndarray
            Histogram Intersection of `data1` and `data2`
    """
    # No obvious way of vectorizing
    if type(data1) == torch.Tensor and type(data2) == torch.Tensor:
        return batch_metric(histogram_intersection, toNumpy(data1),
                            toNumpy(data2))
    else:
        bin_edges = np.histogram_bin_edges(np.concatenate([data1, data2]))
        h1, _ = np.histogram(data1, bins=bin_edges)
        h2, _ = np.histogram(data2, bins=bin_edges)
        return np.sum(np.minimum(h1, h2)) / data1.size


def structural_similarity(data1, data2):
    """Calculate the Structural Similarity between two batches.
    Parameters:
        data1 : torch.Tensor or np.ndarray
            Input batch
        data2 : torch.Tensor or np.ndarray
            Target batch
    Returns:
        np.ndarray
            Structural Similarity of `data1` and `data2`
    """
    # No obvious way of vectorizing
    if type(data1) == torch.Tensor and type(data2) == torch.Tensor:
        return batch_metric(structural_similarity, toNumpy(data1),
                            toNumpy(data2))
    else:
        drange = np.amax((data1, data2)) - np.amin((data1, data2))
        return ssim(data1, data2, data_range=drange)


def psnr(data1, data2):
    """Calculate the Peak Signal-to-Noise Ratio between two batches.
    Parameters:
        data1 : torch.Tensor or np.ndarray
            Input batch
        data2 : torch.Tensor or np.ndarray
            Target batch
    Returns:
        np.ndarray
            PSNR of `data1` and `data2`
    """
    # No obvious way of vectorizing
    if type(data1) == torch.Tensor and type(data2) == torch.Tensor:
        return batch_metric(psnr, toNumpy(data1), toNumpy(data2))
    else:
        drange = np.amax((data1, data2)) - np.amin((data1, data2))
        if (data1 == data2).all():
            return 0
        else:
            return peak_signal_noise_ratio(data1, data2, data_range=drange)


test_metrics = [l1, l2, mean_squared_error, sum_diff_grad, diceCoef, IoU,
                histogram_intersection, structural_similarity, psnr]
