
# imports

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# functions

def use_cait_style(x_size=7.2, y_size=4.45, fontsize=18, autolayout=True, dpi=None):
    """
    TODO

    :param x_size:
    :type x_size:
    :param y_size:
    :type y_size:
    :param fontsize:
    :type fontsize:
    :param autolayout:
    :type autolayout:
    :param dpi:
    :type dpi:
    :return:
    :rtype:
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
    TODO

    :param ax:
    :type ax:
    :return:
    :rtype:
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
    TODO

    TODO add stack link

    :param x_data:
    :type x_data:
    :param y_data:
    :type y_data:
    :param height:
    :type height:
    :param width:
    :type width:
    :param alpha:
    :type alpha:
    :param xlims:
    :type xlims:
    :param ylims:
    :type ylims:
    :return:
    :rtype:
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