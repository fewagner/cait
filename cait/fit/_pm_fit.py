# -------------------------------------------------------
# IMPORTS
# -------------------------------------------------------

import numpy as np
# import numba as nb
from scipy.optimize import curve_fit
from ..fit._templates import pulse_template
import warnings

# -------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------

# @nb.njit
def arrays_equal(a, b):
    """
    A utlis function to compare if two arrays are the same, not used anymore in the current build

    :param a: the first array
    :param b: the second array
    :return: bool, if the same arrays or not
    """
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True


def fit_pulse_shape(event, x0=None, sample_length=0.04):
    """
    Fits the pulse shape model to a given event and returns the fit parameters

    :param event: 1D array, the event to fit
    :param x0: None or 1D array (t0, An, At, tau_n, tau_in, tau_t); the start values for the fit
    :param sample_length: float, the length of one sample in milliseconds
    :return: 1D array length 6, the fitted parameters
    """
    record_length = event.shape[0]
    if x0 is None:
        height_event = np.max(event)
        x0 = np.array([0, 1/height_event, 0.2/height_event, 5, 2, 1])

    t_dummy = (np.arange(0, record_length, dtype=float) - record_length / 4) * sample_length
    try:
        par, _ = curve_fit(f=pulse_template,
                           xdata=t_dummy,
                           ydata=event,
                           p0=x0,
                           bounds=np.array([[-50, -np.inf, -np.inf, 1e-8, 1e-8, 1e-8],
                                            [50, np.inf, np.inf, 1e8, 1e8, 1e8]]))

    except RuntimeError:
        print("Error - curve_fit failed")
        return np.zeros(x0.shape)
    print("Fit Successful.")
    return par
