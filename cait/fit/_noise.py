# imports

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize, curve_fit
import numba as nb


# function

def noise_trigger_template(x_max, d, sigma):
    # TODO
    P = (d / np.sqrt(2 * np.pi) / sigma)
    P *= np.exp(-(x_max / np.sqrt(2) / sigma) ** 2)
    P *= (1 / 2 + erf(x_max / np.sqrt(2) / sigma) / 2) ** (d - 1)
    return P

def wrapper(x_max, d, sigma):
    # TODO

    sigma_lower_bound = 0.0001
    if sigma < sigma_lower_bound:
        P = 1000*np.abs(sigma_lower_bound - sigma) + 1000
    else:
        P = noise_trigger_template(x_max, d, sigma)

    return P

def get_noise_parameters_binned(counts,
                                bins,
                                ):
    # TODO

    x_data = bins[:-1] + (bins[1] - bins[0]) / 2
    ydata = counts

    pars, _ = curve_fit(f=wrapper,
                        xdata=x_data,
                        ydata=ydata,
                        check_finite=True,
                        )

    return pars
