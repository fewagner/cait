# -------------------------------------------------------
# IMPORTS
# -------------------------------------------------------

import numpy as np
import numba as nb
from scipy.optimize import curve_fit
from cait.fit._templates import pulse_template

# -------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------

@nb.njit
def arrays_equal(a, b):
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True


def fit_pulse_shape(event):
    max_event = np.max(event)
    record_length = event.shape[0]
    par_0 = np.array([record_length/4, max_event/2, max_event/2,
                        2.52196961e+03,  9.04381862e+00, 4.63752654e+02])
    t_dummy = np.arange(0, record_length, dtype=float)  # - record_length/4
    try:
        par, cov = curve_fit(f=pulse_template,
                                xdata=t_dummy,
                                ydata=event,
                                p0=[par_0], bounds=(-np.inf, np.inf),
                                maxfev=2000)
    except RuntimeError:
        print("Error - curve_fit failed")
        return np.zeros(par_0.shape)
    return par