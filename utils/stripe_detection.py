import warnings
import numpy as np
import torch
from larix.methods.misc import STRIPES_DETECT, STRIPES_MERGE
from tomopy.prep.stripe import _detect_stripe
from scipy.ndimage import median_filter, uniform_filter1d, binary_dilation
import skimage.morphology as mm
from sklearn.mixture import GaussianMixture
from utils.misc import rescale


def detect_stripe_vo(sinogram, drop_ratio=0.1, snr=3, filter_size=10):
    """Detect stripes using the method described in:
        Vo, N.T., Atwood, R.C. and Drakopoulos, M., 2018. Superior techniques
        for eliminating ring artifacts in X-ray micro-tomography. Optics
        express, 26(22), pp.28396-28412.
    Takes in a sinogram and returns a binary mask of the same shape.
    Parameters:
        sinogram : np.ndarray
            Sinogram to detect stripes in.
        drop_ratio : float
            Percentage of pixels to 'drop' at top and bottom of sinogram.
            Helps to reduce false-positives. Default is 0.1
        snr : float
            Ratio between stripe value and background value.
            Greater is less sensitive. Default is 3.
        filter_size : int
            Size of median filter used to smooth sinogram. Default is 10.
    Returns:
        np.ndarray
            Binary (boolean) mask, with True where a stripe has been detected,
            and False everywhere else.
    """
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


def detect_stripe_mean(sinogram, eta=0.01, kernel_width=3, min_width=2,
                       max_width=25):
    """Detect stripes using the mean curve method described in:
        Ashrafuzzaman, A.N.M., Lee, S.Y. and Hasan, M.K., 2011. A self-adaptive
        approach for the detection and correction of stripes in the sinogram:
        suppression of ring artifacts in CT imaging. EURASIP Journal on
        Advances in Signal Processing, 2011, pp.1-13.
    Takes in a sinogram and returns a binary mask of the same shape.
    Parameters:
        sinogram : np.ndarray
            Sinogram to detect stripes in.
        eta : float
            Minimum intensity of stripe, relative to smoothed mean curve
            of image.
            Default is 0.01
        kernel_width : int
            Width of kernel used to merge stripes together.
            In other words, if there is a gap between two stripes with width
            less than `kernel_width` - 1, the gap will be filled.
            Default is 3
        min_width : int
            Minimum width of stripes. Stripes of width less than `min_width`
            will be removed.
            Default is 2
        max_width : int
            Maximum width of stripes. Stripes of width greater than `max_width`
            will be removed.
            Default is 25
    Returns:
        np.ndarray
            Binary (boolean) mask, with True where a stripe has been detected,
            and False everywhere else.
    """
    mask = np.zeros_like(sinogram, dtype=np.bool_)
    # calculate mean curve, smoothed mean curve, and difference between the two
    mean_curve = np.mean(sinogram, axis=-2)
    mean_curve = rescale(mean_curve)
    smooth_curve = uniform_filter1d(mean_curve, size=5)
    diff_curve = np.abs(smooth_curve - mean_curve)
    mask[..., diff_curve > eta] = True

    convolutions = np.lib.stride_tricks.sliding_window_view(
        mask, (sinogram.shape[-2], kernel_width)
    ).squeeze()
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
    """Detect stripes using the method created by Daniil Kazantsev and
    implemented in Larix.
    Takes in a sinogram and returns a binary mask of the same shape.
    Parameters:
        sinogram : np.ndarray
            Sinogram to detect stripes in.
        threshold : float
            Minimum intensity of stripe, relative to median of sinogram.
            Default is 0.1
    Returns:
        np.ndarray
            Binary (boolean) mask, with True where a stripe has been detected,
            and False everywhere else.
    """
    (stripe_weights, grad_stats) = STRIPES_DETECT(sinogram,
                                                  search_window_dims=(1, 7, 1),
                                                  vert_window_size=5,
                                                  gradient_gap=3)
    # threshold weights to get a initialisation of the mask
    mask = np.zeros_like(stripe_weights, dtype="uint8")
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    mask[stripe_weights > grad_stats[3] / threshold] = 1

    # merge edges that are close to each other
    mask = STRIPES_MERGE(np.ascontiguousarray(mask, dtype=np.uint8),
                         stripe_width_max_perc=25, dilate=3)
    return mask.astype(np.bool_)


def getMask_functional(sinogram, kernel_width=3, min_width=2, max_width=25,
                       threshold=0.01, filter_size=10):
    """Get binary mask of locations of stripes in a sinogram.
    Combination of three methods:
        1. detect_stripe_vo
        2. detect_stripe_mean
        3. detect_stripe_larix
    If two or more of these methods agree on the location of a stripe, it is
    added to the mask. Otherwise, it is discarded.
    Parameters:
        sinogram : torch.Tensor or np.ndarray
            The input sinogram containing stripes to be detected.
        kernel_width : int
            Width of kernel used in detect_stripe_mean
            Default is 3
        min_width : int
            Minimum width of stripes for detect_stripe_mean
            Default is 2
        max_width : int
            Maximum width of stripes for detect_stripe_mean
            Default is 25
        threshold : float
            Minimum intensity of stripe, relative to smoothed mean curve
            of image for detect_stripe_mean.
            Default is 0.01
        filter_size : int
            Size of median filter for detect_stripe_vo.
            Default is 10
    Returns:
        np.ndarray
            Binary (boolean) mask, with True where a stripe has been detected,
            and False everywhere else.
    """
    if isinstance(sinogram, torch.Tensor):
        sino_np = sinogram.detach().numpy().squeeze()
    else:
        sino_np = sinogram
    mask_vo = detect_stripe_vo(sino_np, filter_size=filter_size).astype(int)
    mask_mean = detect_stripe_mean(sino_np, eta=threshold,
                                   kernel_width=kernel_width,
                                   min_width=min_width,
                                   max_width=max_width).astype(int)
    mask_larix = detect_stripe_larix(sino_np).astype(int)
    mask_sum = mask_vo + mask_mean + mask_larix
    mask_sum[mask_sum < 2] = 0

    # if there is a 3 pixel gap or less between stripes, merge them
    convolutions = np.lib.stride_tricks.sliding_window_view(
        mask_sum, (sino_np.shape[-2], kernel_width+2)
    ).squeeze()
    for i, conv in enumerate(convolutions):
        if conv[0, 0] and conv[0, -1]:
            mask_sum[..., i:i + kernel_width+2] = True

    if isinstance(sinogram, torch.Tensor):
        mask_sum = torch.tensor(mask_sum, dtype=torch.bool).unsqueeze(0)
    elif isinstance(sinogram, np.ndarray):
        mask_sum = mask_sum.astype(np.bool_)
    else:
        raise TypeError(f"Expected type {np.ndarray} or {torch.Tensor}. "
                        f"Instead got {type(sinogram)}")
    return mask_sum


###############################################
# Morphological & Clustering stripe detection #
###############################################
def black_tophat_mask(sino, num_clusters=6, iterations=3, max_width=10,
                      min_height=25):
    """Detect stripes by calculating the black tophat, then clustering this
    result.
    Parameters:
        sino : np.ndarray
            Sinogram to detect stripes in.
        num_clusters : int
            Number of clusters to use for Gaussian Mixture Model.
            Default is 6
        iterations : int
            Number of iterations for Gaussian Mixture Model.
            Default is 3
        max_width : int
            Maximum width of stripes; any wider than this will be discarded.
            Default is 10.
        min_height : int
            Minimum height of stripes; any shorter than this will be discarded.
            Default is 25.
    Returns:
        np.ndarray
            Binary mask, with 1 where a stripe has been detected,
            and 0 everywhere else.
    """
    bt = mm.black_tophat(sino)
    m = np.zeros_like(bt)
    m[bt > 0.01] = 1
    m[:, :33] = 0
    m[:, -33:] = 0
    if m[m == 1].size < num_clusters:
        # if no. of True values in mask is < n_components, GMM will throw error
        cm = m
    else:
        cm = clusterMask(m, num_clusters, iterations, max_width,
                         min_height).astype(np.bool_)
    return cm


def white_tophat_mask(sino, num_clusters=6, iterations=3, max_width=10,
                      min_height=25):
    """Detect stripes by calculating the white tophat, then clustering this
    result.
    Parameters:
        sino : np.ndarray
            Sinogram to detect stripes in.
        num_clusters : int
            Number of clusters to use for Gaussian Mixture Model.
            Default is 6
        iterations : int
            Number of iterations for Gaussian Mixture Model.
            Default is 3
        max_width : int
            Maximum width of stripes; any wider than this will be discarded.
            Default is 10.
        min_height : int
            Minimum height of stripes; any shorter than this will be discarded.
            Default is 25.
    Returns:
        np.ndarray
            Binary mask, with 1 where a stripe has been detected,
            and 0 everywhere else.
    """
    wt = mm.white_tophat(sino)
    m = np.zeros_like(wt)
    m[wt > 0.01] = 1
    m[:, :33] = 0
    m[:, -33:] = 0
    if m[m == 1].size < num_clusters:
        # if no. of True values in mask is < n_components, GMM will throw error
        cm = m
    else:
        cm = clusterMask(m, num_clusters, iterations, max_width,
                         min_height).astype(np.bool_)
    return cm


def clusterMask(binary_mask, num_clusters=6, iterations=3, max_width=10,
                min_height=25):
    """Cluster the results of a binary mask; any clusters that do not follow
    the expected shape of a stripe (long and thin) are discarded.
    Parameters:
        binary_mask : np.ndarray
            Binary mask indicating the locations of stripes.
        num_clusters : int
            Number of clusters to use for Gaussian Mixture Model.
            Default is 6
        iterations : int
            Number of iterations for Gaussian Mixture Model.
            Default is 3
        max_width : int
            Maximum width of stripes; any wider than this will be discarded.
            Default is 10.
        min_height : int
            Minimum height of stripes; any shorter than this will be discarded.
            Default is 25.
    Returns:
        np.ndarray
            Binary mask with un-shapely stripes discarded. Same shape as input.
    """
    # Get vector of coordinates where mask > 0
    # This needs to be in form (x, y), but numpy lists elements in form (y, x),
    # so we have to get elements in reverse, hence the slice [::-1]
    X = np.stack(np.where(binary_mask > 0)[::-1], axis=-1)
    # Fit a Gaussian Mixture model to cluster the coordinate vector
    cluster_labels = GaussianMixture(n_components=num_clusters,
                                     n_init=iterations).fit_predict(X)
    mask = np.zeros_like(binary_mask)
    # Loop through each cluster
    for label in np.unique(cluster_labels):
        # get coordinates of every point in `X` that belongs to cluster `l`
        c = X[cluster_labels == label]
        # If width and height are within bounds,
        # assume the cluster represents a stripe
        width = np.max(c[:, 0]) - np.min(c[:, 0])
        height = np.max(c[:, 1]) - np.min(c[:, 1])
        if width < max_width and height > min_height:
            # Set values of mask to 1 that correspond to coordinates in cluster
            # Needs to be a tuple for indexing to work properly.
            # `c` has shape (<cluster_size>, 2) so must be transposed to have
            # shape (2, <cluster_size>)
            # `c` is also in form (x, y) but numpy needs form (y, x) so we have
            # to reverse elements, hence slice [::-1]
            mask[tuple(c.T[::-1])] = 1
    return mask


def getMask_morphological(sinogram, num_clusters=6, iterations=3, max_width=10,
                          min_height=25):
    """Get binary mask of locations of stripes in a sinogram.
    Uses morphological processing and clustering to detect stripes.
    Parameters:
        sinogram : np.ndarray or torch.Tensor
            The input sinogram containing stripes to be detected.
        num_clusters : int
            Number of clusters to use for Gaussian Mixture Model.
            Default is 6
        iterations : int
            Number of iterations for Gaussian Mixture Model.
            Default is 3
        max_width : int
            Maximum width of stripes; any wider than this will be discarded.
            Default is 10.
        min_height : int
            Minimum height of stripes; any shorter than this will be discarded.
            Default is 25.
    Returns:
        np.ndarray
            Binary (boolean) mask, with True where a stripe has been detected,
            and False everywhere else.
    """
    if isinstance(sinogram, torch.Tensor):
        sino_np = sinogram.detach().numpy().squeeze()
    else:
        sino_np = sinogram
    # Calculate black & white tophat masks
    m1 = black_tophat_mask(sino_np)
    m2 = white_tophat_mask(sino_np)
    m = m1 + m2
    # Dilate using a long and thin rectangle (10% of input height by 2 pixels)
    footprint = mm.footprints.rectangle(int(0.1 * sino_np.shape[0]), 2)
    m = mm.binary_dilation(m, footprint=footprint)
    if isinstance(sinogram, torch.Tensor):
        m = torch.tensor(m, dtype=torch.bool).unsqueeze(0)
    elif isinstance(sinogram, np.ndarray):
        m = m.astype(np.bool_)
    else:
        raise TypeError(f"Expected type {np.ndarray} or {torch.Tensor}. "
                        f"Instead got {type(sinogram)}")
    return m


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


stripe_detection_metrics = [sum_max, total_variation2D, gradient_sum_max,
                            gradient_tv, gradient_sum_tv, grad_sum_res_max,
                            grad_sum_z_max]
