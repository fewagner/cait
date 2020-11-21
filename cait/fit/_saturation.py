# imports

import numpy as np


# functions

def logistic_curve(x, max, slope):
    """
    Returns the evaluated logistics curve at x.

    :param x: the x value or grid
    :type x: scalar or 1D numpy array
    :param max: the upper limit of the log curve
    :type max: float
    :param slope: the factor is multiplied to x
    :type slope: float
    :return: the evaluated log curve at x
    :rtype: scalar or 1D numpy array
    """
    return max * (1 / (1 + np.exp(-slope * x)) - 0.5)


# derivative at 0 is k*l*np.exp(k*x0)/(1+np.exp(k*x0))**2

def scaled_logistic_curve(x, max, slope):
    """
    Logistics Curve scaled to max derivative 1, evaluated at x.

    :param x: the x value or grid
    :type x: scalar or 1D numpy array
    :param max: the upper limit of the log curve
    :type max: float
    :param slope: the factor is multiplied to x
    :type slope: float
    :return: the evaluated scaled log curve at x
    :rtype: scalar or 1D numpy array
    """
    return logistic_curve(x / scale_factor(max, slope), max, slope)


def scale_factor(max, slope):
    """
    "Returns the slope of the logistics curve at x=0."

    :param max: the upper limit of the log curve
    :type max: float
    :param slope: the factor is multiplied to x
    :type slope: float
    :return: the slope of the log curve at x=0
    :rtype: float
    """
    return 0.25 * slope * max
