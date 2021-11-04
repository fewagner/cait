import numpy as np
from numpy.linalg import lstsq
import numba as nb
from scipy.optimize import curve_fit
from ..fit._templates import exponential_bl


def box_car_smoothing(event, length=50):
    """
    Calculates a moving average on an event array and returns the smoothed event

    :param event: 1D array, the event to calcualte the MA
    :param length: the length of the moving average
    :return: 1D array the smoothed array
    """
    event = np.pad(event, length, 'edge')
    event = 0.02 * np.convolve(event, np.array([1]).repeat(50), 'same')
    return event[length:-length]


@nb.jit
def linregfit(XX, yy):
    """"
    Fit a large set of points to a regression.

    TODO
    """
    assert XX.shape == yy.shape, "Inputs mismatched"
    n_pnts, n_samples = XX.shape

    scale = np.empty(n_pnts)
    offset = np.empty(n_pnts)

    for i in nb.prange(n_pnts):
        X, y = XX[i], yy[i]
        A = np.vstack((np.ones_like(X), X)).T
        offset[i], scale[i] = lstsq(A, y)[0]

    return offset, scale


def rem_off(ev: list, baseline_model: str, pretrigger_samples: int = 500):
    """TODO"""
    if baseline_model == 'constant':
        ev -= np.mean(ev[:, :pretrigger_samples], axis=1, keepdims=True)
    elif baseline_model == 'linear':
        t = np.arange(0, pretrigger_samples, dtype=ev.dtype) * np.ones((ev.shape[0], pretrigger_samples), dtype=ev.dtype)
        offset, scale = linregfit(XX=t, yy=ev[:, :pretrigger_samples])
        t = np.arange(0, ev.shape[1], dtype=float) * np.ones(ev.shape, dtype=float)
        ev -= scale.reshape(ev.shape[0], 1) * t + offset.reshape(ev.shape[0], 1)
    elif baseline_model == 'exponential':
        t = np.arange(0, ev.shape[1], dtype=ev.dtype)
        for i in range(ev.shape[0]):
            popt, _ = curve_fit(exponential_bl,
                                xdata=t[:pretrigger_samples], ydata=ev[i, :pretrigger_samples],
                                p0=[np.mean(ev[i, :pretrigger_samples]), 0])
            ev[i] -= exponential_bl(t, *popt)
    else:
        raise NotImplementedError('This baseline model is not implemented.')
