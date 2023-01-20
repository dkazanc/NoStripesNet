import matplotlib.pyplot as plt


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
        # if min == max, then the data is a constant value, and so normalising will have no effect
        # this also avoids a Divide By Zero error
        if tmp_min == tmp_max:
            return data
        return self.a + ((data - tmp_min) * (self.b - self.a)) / (tmp_max - tmp_min)


def plot_images(*images, titles=None, subplot_size=None, axis='off', clim=(None, None)):
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


def plot_data(*data, titles=None, subplot_size=None, lim=(None, None)):
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
    plt.show()
