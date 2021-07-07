# imports

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib.pyplot as plt
from ..styles import make_grid, use_cait_style

# function

def threshold_model(x, a0, a1, a2):
    """
    Fit model for the threshold

    :param x: The grid on which the model is evaluated.
    :type x: array
    :param a0: Estimated constant survival probability above threshold.
    :type a0: float
    :param a1: Estimated threshold value.
    :type a1: float
    :param a2: Estimator for the energy resolution.
    :type a2: float
    :return: The evaluated error function
    :rtype: array
    """
    return 0.5 * a0 * (1 + erf((x - a1) / (np.sqrt(2) * a2)))

def fit_trigger_efficiency(binned_energies, survived_fraction, a1_0, a0_0=1, a2_0=0.01,
                           plot=False, title=None, xlim=None):
    """
    Fit and plot the trigger efficiency.

    :param binned_energies: The bin edges, in keV.
    :type binned_energies: array of length nmbr_bins + 1
    :param survived_fraction: The number of survived events per bin, in keV.
    :type survived_fraction: array
    :param a0_0: Start Value for estimated constant survival probability above threshold.
    :type a0_0: float
    :param a1_0: Start Value for estimated threshold value, in keV.
    :type a1_0: float
    :param a2_0: Start Value for estimator for the energy resolution, in keV.
    :type a2_0: float
    :param plot: Plot the fitted function.
    :type plot: bool
    :param title: The title for the plot.
    :type title: bool
    :param xlim: The x limits for the plot.
    :type xlim: bool
    :return: The fitted values a0, a1, a2.
    :rtype: list
    """
    x_grid = binned_energies[:-1] + (binned_energies[1:] - binned_energies[:-1])/2

    pars, _ = curve_fit(xdata=x_grid, ydata=survived_fraction, p0=(a0_0, a1_0, a2_0))

    a0, a1, a2 = pars

    print('Estimated constant survival probability: ', a0)
    print('Estimated energy threshold (keV): ', a1)
    print('Estimated energy resolution (keV): ', a2)

    if plot:
        if xlim is None:
            fine_grid = np.linspace(0, x_grid[-1], 1000)
        else:
            fine_grid = np.linspace(xlim[0], xlim[1], 1000)

        plt.close()
        use_cait_style()
        plt.hist(x=x_grid, bins=binned_energies, weights=survived_fraction)
        plt.plot(fine_grid, threshold_model(fine_grid, *pars), color='red', linewidth=2)
        make_grid()
        plt.ylabel('Survival Probability')
        plt.xlabel('Energy (keV)')
        if title is not None:
            plt.title(title)
        if xlim is not None:
            plt.xlim(xlim)
        plt.show()

    return a0, a1, a2
