
# imports

import matplotlib as mpl
import matplotlib.pyplot as plt

# functions

def use_cait_style(x_size=7.2, y_size=4.45, fontsize=18, autolayout=True, dpi=None):
    """
    TODO

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

def make_grid():
    # major grid lines
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)