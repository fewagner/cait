# -------------------------------------------------------
# IMPORTS
# -------------------------------------------------------

import numpy as np
import numba as nb
from scipy.optimize import curve_fit
from ..fit._templates import pulse_template


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


def fit_pulse_shape(event, x0=None, sample_length=0.04):
    record_length = event.shape[0]
    if x0 is None:
        height_event = np.max(event)
        x0 = [0, 1/height_event, 0.2/height_event, 5, 2, 1]

    t_dummy = (np.arange(0, record_length, dtype=float) - record_length / 4) * sample_length
    try:
        par, _ = curve_fit(f=pulse_template,
                           xdata=t_dummy,
                           ydata=event,
                           p0=x0)
        # , bounds=(-np.inf, np.inf),
        # maxfev=2000)
    except RuntimeError:
        print("Error - curve_fit failed")
        return np.zeros(x0.shape)
    return par
