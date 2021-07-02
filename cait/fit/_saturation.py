# imports

import numpy as np
import numba as nb


# functions

@nb.njit
def logistic_curve(x, A, K, C, Q, B, nu):
    """
    Returns the evaluated logistics curve at x.

    :param x: The x value or grid.
    :type x: scalar or 1D numpy array
    :param A: A parameter of the generalized logistics curve.
    :type A: float
    :param K: A parameter of the generalized logistics curve.
    :type K: float
    :param C: A parameter of the generalized logistics curve.
    :type C: float
    :param Q: A parameter of the generalized logistics curve.
    :type Q: float
    :param B: A parameter of the generalized logistics curve.
    :type B: float
    :param nu: A parameter of the generalized logistics curve.
    :type nu: float
    :return: The evaluated log curve at x.
    :rtype: scalar or 1D numpy array
    """
    return A + (K - A) / (C + Q * np.exp(-B * x)) ** (1 / nu)

@nb.njit
def scaled_logistic_curve(x, A, K, C, Q, B, nu):
    """
    Logistics Curve scaled to max derivative 1, evaluated at x.

    :param x: The x value or grid.
    :type x: scalar or 1D numpy array
    :param A: A parameter of the generalized logistics curve.
    :type A: float
    :param K: A parameter of the generalized logistics curve.
    :type K: float
    :param C: A parameter of the generalized logistics curve.
    :type C: float
    :param Q: A parameter of the generalized logistics curve.
    :type Q: float
    :param B: A parameter of the generalized logistics curve.
    :type B: float
    :param nu: A parameter of the generalized logistics curve.
    :type nu: float
    :return: The evaluated scaled log curve at x.
    :rtype: scalar or 1D numpy array
    """
    return logistic_curve(x / scale_factor(A, K, C, Q, B, nu), A, K, C, Q, B, nu)

@nb.njit
def scale_factor(A, K, C, Q, B, nu):
    """
    Returns the slope of the logistics curve at x=0.

    :param A: A parameter of the generalized logistics curve.
    :type A: float
    :param K: A parameter of the generalized logistics curve.
    :type K: float
    :param C: A parameter of the generalized logistics curve.
    :type C: float
    :param Q: A parameter of the generalized logistics curve.
    :type Q: float
    :param B: A parameter of the generalized logistics curve.
    :type B: float
    :param nu: A parameter of the generalized logistics curve.
    :type nu: float
    :return: The slope of the log curve at x=0.
    :rtype: float
    """
    # (B * Q * (K - A) * np.exp(-B * x) * (Q * np.exp(-B * x) + C)**(-1/nu - 1))/nu ... at x = 0
    return (B * Q * (K - A) * (Q + C)**(-1/nu - 1))/nu
