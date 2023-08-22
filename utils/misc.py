import matplotlib.pyplot as plt
import numpy as np
import torch


def toTensor(array, device=torch.device('cpu')):
    """Convert an array-like to a tensor.
    Tensors have dtype torch.float32.
    Parameters:
        array : np.ndarray
            Array to convert to tensor
        device : torch.device
            Device to convert tensor to. Default is CPU.
    Return:
        torch.Tensor
            Tensor of input array
    """
    return torch.tensor(array, dtype=torch.float32).unsqueeze(0).to(device)


def toNumpy(tensor):
    """Convert a tensor to a numpy array.
    Disables gradient calculation & converts input tensor to CPU.
    Also reduces any singleton dimensions, so that channels are removed.
    Parameters:
        tensor : torch.Tensor
            Tensor to be converted to a numpy array.
    Returns:
        np.ndarray
            Numpy array of input tensor.
    """
    return tensor.detach().squeeze().cpu().numpy()


def rescale(data, a=0, b=1, imin=None, imax=None):
    """Rescale data to range [a, b] w.r.t. bounds [imin, imax].
    Parameters:
        data : np.ndarray
            The data to be rescaled
        a : float
            The minimum value of the range the data will be scaled to.
            Default is 0
        b : float
            The maximum value of the range the data will be scaled to.
            Default is 1
        imin :  float
            The lower bound with respect to which the data will be scaled.
            Default is minimum value of `data`.
        imax: float
            The upper bound with respect to which the data will be scaled.
            Default is maximum value of `data`.
    Returns:
        out : np.ndarray
            The rescaled data. Has the same size & shape as input `data`.
    """
    data = data.astype(np.float32, copy=False)
    if imin is None:
        imin = data.min()
    if imax is None:
        imax = data.max()
    # if imin == imax, then the data is a constant value,
    # and so normalising will have no effect.
    # this also avoids a Divide By Zero error.
    if imin == imax:
        return data
    out = a + ((data - imin)*(b - a)) / (imax - imin)
    return out


class Rescale:
    """Rescale data to a given range."""

    def __init__(self, a=0, b=1, imin=None, imax=None):
        """Parameters:
            a : float
                The minimum value of the range the data will be scaled to.
                Default is 0
            b : float
                The maximum value of the range the data will be scaled to.
                Default is 1
            imin :  float
                The lower bound with respect to which the data will be scaled.
                Default is minimum value of `data`.
            imax: float
                The upper bound with respect to which the data will be scaled.
                Default is maximum value of `data`.
        """
        self.a = a
        self.b = b
        self.imin = imin
        self.imax = imax

    def __call__(self, data):
        """Rescale data to range [a, b] w.r.t. bounds [imin, imax].
        Returns:
            out : np.ndarray
                The rescaled data. Has the same size & shape as input `data`.
        """
        if self.imin is None:
            tmp_min = data.min()
        else:
            tmp_min = self.imin
        if self.imax is None:
            tmp_max = data.max()
        else:
            tmp_max = self.imax
        # if min == max, then the data is a constant value,
        # and so normalising will have no effect.
        # this also avoids a Divide By Zero error.
        if tmp_min == tmp_max:
            return data
        return self.a + ((data - tmp_min) * (self.b - self.a)
                         ) / (tmp_max - tmp_min)


def plot_images(*images, titles=None, subplot_size=None, axis='off',
                clim=(None, None)):
    """Plot a series of images using subplots.
    Images are plotted with colourmap 'gray'.
    Parameters:
        *images : np.ndarray
            Argument list of images to be plotted.
        titles : List[str]
            List containing the titles for each subplot. Default is None.
        subplot_size : Tuple[int, int]
            Tuple containing size of subplots. First item is number of rows,
            second item is number of columns.
            Default is (1, len(images)).
        axis : str
            String specifying whether axes should be shown on subplots.
            Must be either 'off' or 'on'. Default is 'off'.
        clim : Tuple[float, float]
            Colour limits for plotting images.
    """
    if subplot_size is None:
        subplot_size = (1, len(images))
    for i, img in enumerate(images):
        if img is None:
            continue
        plt.subplot(*subplot_size, i + 1)
        plt.imshow(img, cmap='gray', vmin=clim[0], vmax=clim[1])
        plt.axis(axis)
        if type(titles) == list and i < len(titles):
            plt.title(str(titles[i]))
    plt.show()


def plot_data(*data, titles=None, subplot_size=None, axis='on',
              lim=(None, None)):
    """Plot a series of data using subplots.
    Parameters:
        *data : np.ndarray
            Argument list of data to be plotted.
        titles : List[str]
            List containing the titles for each subplot. Default is None.
        subplot_size : Tuple[int, int]
            Tuple containing size of subplots. First item is number of rows,
            second item is number of columns.
            Default is (1, len(images)).
        axis : str
            String specifying whether axes should be shown on subplots.
            Must be either 'off' or 'on'. Default is 'on'.
        lim : Tuple[float, float]
            Y-axis limits for plotting data.
    """
    if subplot_size is None:
        subplot_size = (1, len(data))
    for i, d in enumerate(data):
        if d is None:
            continue
        plt.subplot(*subplot_size, i + 1)
        plt.plot(d)
        if type(titles) == list and i < len(titles):
            plt.title(str(titles[i]))
        plt.ylim(lim)
        plt.axis(axis)
    plt.show()


def plot_mix(*mixed_data, subplot_size=None, titles=None, axis='off',
             clim=(None, None), lim=(None, None)):
    """Plot a mixed series of images and data using subplots.
    Images are plotted with colourmap 'gray'.
    Parameters:
        *mixed_data : np.ndarray
            Argument list of images and data to be plotted.
        titles : List[str]
            List containing the titles for each subplot. Default is None.
        subplot_size : Tuple[int, int]
            Tuple containing size of subplots. First item is number of rows,
            second item is number of columns.
            Default is (1, len(images)).
        axis : str
            String specifying whether axes should be shown on subplots.
            Must be either 'off' or 'on'. Default is 'off'.
        clim : Tuple[float, float]
            Colour limits for plotting data.
        lim : Tuple[float, float]
            Y-axis limits for plotting data.
    """
    if subplot_size is None:
        subplot_size = (1, len(mixed_data))
    for i, data in enumerate(mixed_data):
        if data is None:
            continue
        plt.subplot(*subplot_size, i+1)
        if type(titles) == list and i < len(titles):
            plt.title(str(titles[i]))
        if type(data) == tuple:
            plt.scatter(data[0], data[1])
        elif data.ndim == 2:
            plt.imshow(data, cmap='gray')
            plt.clim(*clim)
            plt.axis(axis)
        else:
            plt.plot(data)
            plt.ylim(lim)
    plt.show()
