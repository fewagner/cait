
import numpy as np
from scipy.optimize import curve_fit
from ._templates import baseline_template_quad

def fit_quadratic_baseline(event):
    length_event = len(event)
    idx = np.linspace(0, length_event - 1, length_event)
    popt, _ = curve_fit(baseline_template_quad, idx, event)

    offset = popt[0]
    linear_drift = popt[1]
    quadratic_drift = popt[2]

    return offset, linear_drift, quadratic_drift