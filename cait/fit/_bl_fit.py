
import numpy as np
from scipy.optimize import curve_fit
from ._templates import baseline_template_quad
import numba as nb

def fit_quadratic_baseline(event):
    """
    Fits a quadratic baseline template the to given event

    :param event: 1D array, the event to fit the baseline template
    :return: list of 3 floats (offset, linear_drift, quadratic_drift)
    """
    length_event = len(event)
    idx = np.linspace(0, length_event - 1, length_event)
    popt, _ = curve_fit(baseline_template_quad, idx, event)

    offset = popt[0]
    linear_drift = popt[1]
    quadratic_drift = popt[2]

    return offset, linear_drift, quadratic_drift

def get_rms(x, y):
    """
    Utils function to geh the average RMS between two arrays

    :param x: 1D array
    :param y: 1D array
    :return: float, the RMS value
    """
    return np.mean((x - y)**2)
