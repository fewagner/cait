# -------------------------------------------------------
# IMPORTS
# -------------------------------------------------------

import numpy as np
import numba as nb
from scipy.optimize import curve_fit, minimize, differential_evolution, Bounds
from ..fit._templates import pulse_template
import warnings


# -------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------

# @nb.njit
def arrays_equal(a: list, b: list):
    """
    A utlis function to compare if two arrays are the same, not used anymore in the current build.

    :param a: The first array.
    :type a: array
    :param b: The second array.
    :type b: array
    :return: If the arrays are the same or not.
    :rtype: bool
    """
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True

@nb.njit
def boundscheck(pars, lbounds,
                ubounds):
    """
    Checks if parameters are out of bounds and returns large positive value if so.

    :param pars: The parameters that we want to check
    :type pars: array
    :param lbounds: The lower bounds for all parameters.
    :type lbounds: array
    :param ubounds: The upper bounds for all parameters.
    :type ubounds: array
    :return: Zero if all parameters are within their bounds, large positive if parameters exceed bounds.
    :rtype: float
    """
    retval = 0.0
    for i in np.arange(len(pars)):
        if pars[i] < lbounds[i]:
            retval += 1.0e100 - 1.0e100 * (pars[i] - lbounds[i])
        if pars[i] > ubounds[i]:
            retval += 1.0e100 - 1.0e100 * (ubounds[i] - pars[i])
    return retval

@nb.njit
def ps_minimizer(par, t, event, t0_lb, t0_ub, tau_lb):
    """
    The minimizer function for the parametric pulse shape model fit.

    This function is implemented with Numba in no python mode and precompiled, for better performance.

    :param par: The parameters that are minimized: (t0, An, At, tau_n, tau_in, tau_t).
    :type par: numpy array
    :param t: The time grid in ms.
    :type t: numpy array
    :param event: The time series of the event that is to fit.
    :type event: numpy array
    :param t0_lb:
    :type t0_lb: float
    :param t0_ub:
    :type t0_ub: float
    :param tau_lb:
    :type tau_lb: float
    :return: The loss value that it to be minimized.
    :rtype: float
    """
    out = 0

    # t0 bounds
    if par[0] < t0_lb:
        out += 1.0e100 - 1.0e100 * (par[0] - t0_lb)
    elif par[0] > t0_ub:
        out += 1.0e100 - 1.0e100 * (t0_lb - par[0])

    # tau lower bounds
    for p in par[3:6]:
        if p < tau_lb:
            out += 1.0e100 - 1.0e100 * (p - tau_lb)

    if out == 0:
        fit = pulse_template(t, par[0], par[1], par[2], par[3], par[4], par[5])
        out += np.sum((fit - event) ** 2)

    return out


def fit_pulse_shape(event, x0=None, sample_length=0.04, down=1, t0_start=-3, t0_bounds=(-10, 5), opt_start=False):
    """
    Fits the pulse shape model to a given event and returns the fit parameters.

    :param event: The event to fit.
    :param event: 1D array
    :param x0: None or 1D array (t0, An, At, tau_n, tau_in, tau_t); the start values for the fit

    :param sample_length: float, the length of one sample in milliseconds

    :param down: int, should be power of 2, downsample rate during the fit

    :param t0_start: float, the start value for t0

    :param t0_bounds: tuple, lower and upper bound for the t0 value

    :param opt_start: bool, if activated the start values are searched with a differential evolution algorithm

    :return: 1D array length 6, the fitted parameters

    """
    record_length = event.shape[0]
    if x0 is None:
        height_event = np.max(event)
        x0 = np.array([t0_start, 1 * height_event, 0.2 * height_event, 5, 2, 1])

    if down > 1:
        event = np.mean(event.reshape(len(event) / down, down), axis=1)
        sample_length *= down
        record_length /= down

    t_dummy = (np.arange(0, record_length, dtype=float) - record_length / 4) * sample_length
    # try:

    if opt_start:

        bounds = Bounds(lb=[t0_bounds[0], 0, 0, 1e-1, 1e-2, 1e-1],
                        ub=[t0_bounds[1], 5 * height_event, 5 * height_event, 3e3, 1e2, 1e3],
                        )

        par = differential_evolution(ps_minimizer,
                                     args=(t_dummy, event, t0_bounds[0], t0_bounds[1], 1e-2),
                                     bounds=bounds,
                                     strategy='rand1bin',
                                     workers=-1,
                                     maxiter=10000,
                                     popsize=50,
                                     tol=0.0001,
                                     updating='deferred'
                                     )

        x0 = par.x

    par = minimize(ps_minimizer,
                   x0=x0,
                   method='nelder-mead',
                   args=(t_dummy, event, t0_bounds[0], t0_bounds[1], 1e-2),
                   options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True},
                   )

    res = par.x

    return res
