
# imports

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# functions

def use_cait_style(x_size=7.2, y_size=4.45, fontsize=18, autolayout=True, dpi=None):
    """
    Use the pyplot plot style that is used within the Cait plotting routines.

    :param x_size: The width of the plot in cm.
    :type x_size: float
    :param y_size: The height of the plot in cm.
    :type y_size: float
    :param fontsize: The font size of the plot.
    :type fontsize: int
    :param autolayout: Activate autolayout.
    :type autolayout: bool
    :param dpi: The dots per inch for the plot.
    :type dpi: int
    """
    plt.style.use('seaborn-paper')
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['font.size'] = fontsize
    if autolayout is not None:
        mpl.rcParams['figure.autolayout'] = autolayout
    mpl.rcParams['figure.figsize'] = (x_size, y_size)
    if dpi is not None:
        mpl.rcParams['figure.dpi'] = dpi
    mpl.rcParams['axes.titlesize'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 6
    mpl.rcParams['legend.fontsize'] = fontsize
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    if dpi is not None:
        mpl.rcParams['savefig.dpi'] = dpi

def make_grid(ax=None):
    """
    Produce the pyplot plot grid that is used within the Cait plotting routines.

    :param ax: A pyplot axis object, optional.
    :type ax: object
    """
    if ax is None:
        # major grid lines
        plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
        # minor grid lines
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
    else:
        # major grid lines
        ax.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
        # minor grid lines
        ax.minorticks_on()
        ax.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)


def scatter_img(x_data, y_data, height=2800, width=2800, alpha=0.3, xlims=None, ylims=None):
    """
    Produce a scatter plot to plot as an image with pyplot.imshow.

    :param x_data: The values for the x axis.
    :type x_data: 1D array
    :param y_data: The values for the x axis.
    :type y_data: 1D array
    :param height: The number of pixels on the x axis.
    :type height: int
    :param width: The number of pixels on the y axis.
    :type width: int
    :param alpha: The occupacity of one event inside a pixel, between 0 and 1.
    :type alpha: float
    :param xlims: The limits on the x axis.
    :type xlims: 2-tuple
    :param ylims: The limits on the y axis.
    :type ylims: 2-tuple
    :return: List of the x limits, the y limits and the density values for the image plot.
    :rtype: list of (2-tuple, 2-tuple, 2D matrix)
    """
    if xlims is None:
        xlims = (x_data.min(), x_data.max())
    if ylims is None:
        ylims = (y_data.min(), y_data.max())

    dxl = xlims[1] - xlims[0]
    dyl = ylims[1] - ylims[0]

    buffer = np.zeros((height + 1, width + 1))
    for i, (x, y) in enumerate(tqdm(zip(x_data, y_data))):
        x0 = int(round(((x - xlims[0]) / dxl) * width))
        y0 = int(round((1 - (y - ylims[0]) / dyl) * height))
        if x0 > 0 and x0 < width + 1 and y0 > 0 and y0 < height + 1:
            buffer[y0, x0] += alpha
            if buffer[y0, x0] > 1.0: buffer[y0, x0] = 1.0
    return xlims, ylims, buffer