# -----------------------------------------------------
# IMPORTS
# -----------------------------------------------------

import numpy as np
from scipy.optimize import minimize

# -----------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------

def baseline_template_quad(t, c0, c1, c2):
    """
    Template for the baseline fit, with constant linear and
    quadratic component

    :param t: 1D array, the time grid
    :param c0: float, constant component
    :param c1: float, linear component
    :param c2: float, quadratic component
    :return: 1D array, the parabola evaluated on the time grid
    """
    return c0 + t * c1 + t ** 2 * c2


def baseline_template_cubic(t, c0, c1, c2, c3):
    """
    Template for the baseline fit, with constant linear,
    quadratic and cubic component

    :param t: 1D array, the time grid
    :param c0: float, constant component
    :param c1: float, linear component
    :param c2: float, quadratic component
    :param c3: float, cubic component
    :return: 1D array, the cubic polynomial evaluated on the time grid
    """
    return c0 + t * c1 + t ** 2 * c2 + t ** 3 * c3


def pulse_template(t, t0, An, At, tau_n, tau_in, tau_t):
    """
    Parametric model for the pulse shape, 6 parameters

    :param t: 1D array, the time grid
    :param t0: float, the pulse onset time
    :param An: float, Amplitude of the first nonthermal pulse component
    :param At: float, Amplitude of the thermal pulse component
    :param tau_n: float, parameter for decay 1. comp and rise 2. comp
    :param tau_in: float, parameter for rise 1. comp
    :param tau_t: float, parameter for decay 2. comp
    :return: 1D array, the pulse model evaluated on the time grid
    """

    pulse = (t >= t0).astype(float)
    t_red = t[t >= t0]
    pulse[t >= t0] *= (An * (np.exp(-(t_red - t0) / tau_n) - np.exp(-(t_red - t0) / tau_in)) + \
                       At * (np.exp(-(t_red - t0) / tau_t) - np.exp(-(t_red - t0) / tau_n)))
    return pulse

# Define gauss function
def gauss(x, *p):
    """
    Evaluate a gauss function on a given grid x

    :param x: 1D array, the grid
    :param p: 1D array length 3: (normalization, mean, stdeviation)
    :return: 1D array of same length than x, evaluated gauss function
    """
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

# -----------------------------------------------------
# CLASSES
# -----------------------------------------------------

class sev_fit_template:
    """
    Class to store pulse fit models for individual detectors

    :param par: 1D array with size 6, the fit parameter of the sev
        (t0, An, At, tau_n, tau_in, tau_t)
    :param t: 1D array, the time grid on which the pulse shape model is evaluated
    """

    def __init__(self, pm_par, t):
        self.pm_par = pm_par
        self.t = t

    def sef(self, h, t0, a0):
        """
        Standard Event Model with Flat Baseline

        :param h: float, hight of pulse shape
        :param t0: float, onset of pulse shape
        :param a0: float, offset of the baseline
        :return: 1D array, the pulse model evaluated on the time grid
        """
        x = self.pm_par
        x[0] += t0
        return h * pulse_template(self.t, *x) + a0

    def sel(self, h, t0, a0, a1):
        """
        Standard Event Model with Linear Baseline

        :param h: float, hight of pulse shape
        :param t0: float, onset of pulse shape
        :param a0: float, offset of the baseline
        :param a1: float, linear drift component of the baseline
        :return: 1D array, the pulse model evaluated on the time grid
        """
        x = self.pm_par
        x[0] += t0
        return h * pulse_template(self.t, *x) + a0 + \
               a1 * (self.t - t0)

    def seq(self, h, t0, a0, a1, a2):
        """
        Standard Event Model with Quadradtic Baseline

        :param h: float, hight of pulse shape
        :param t0: float, onset of pulse shape
        :param a0: float, offset of the baseline
        :param a1: float, linear drift component of the baseline
        :param a2: float, quadratic drift component of the baseline
        :return: 1D array, the pulse model evaluated on the time grid
        """
        x = self.pm_par
        x[0] += t0
        return h * pulse_template(self.t, *x) + a0 + \
               a1 * (self.t - t0) + a2 * (self.t - t0)

    def sec(self, h, t0, a0, a1, a2, a3):
        """
        Standard Event Model with Quadradtic Baseline

        :param h: float, hight of pulse shape
        :param t0: float, onset of pulse shape
        :param a0: float, offset of the baseline
        :param a1: float, linear drift component of the baseline
        :param a2: float, quadratic drift component of the baseline
        :param a3: float, cubic drift component of the baseline
        :return: 1D array, the pulse model evaluated on the time grid
        """
        x = self.pm_par
        x[0] += t0
        return h * pulse_template(self.t, *x) + a0 + \
               a1 * self.t + a2 * self.t + \
               a3 * self.t

    def fit(self, event, order_polynomial=1):
        """
        Calculates the SEV Fit Parameter

        :param event: 1D array, the event to fit
        :param order_polynomial: int, the order of the polynomial to fit
        :return: 1D array, the sev fit parameters
        """

        event = event - np.mean(event[:int(len(event)/8)])

        # handle for fit template
        if order_polynomial == 0:
            template = self.sef
        elif order_polynomial == 1:
            template = self.sel
        elif order_polynomial == 2:
            template = self.seq
        elif order_polynomial == 3:
            template = self.sec
        else:
            raise KeyError('Order Polynomial must be 0,1,2 or 3!')

        # find initial values
        iv = np.zeros(int(2 + order_polynomial))
        iv[1] = 0  # offset
        iv[0] = np.max(event) # height
        iv[2] = (event[-1] - template(iv[0], 0, *iv[1:])[-1] - event[0]) / (self.t[-1] - self.t[0]) # linear slope

        # fit t0
        t0_minimizer = lambda par: np.sum((template(iv[0], par, *iv[1:]) - event) ** 2)
        res = minimize(t0_minimizer,
                       np.array([0]))
        t0 = res.x
        # fit height and baseline
        par_minimizer = lambda par: np.sum((template(par[0], t0, *par[1:]) - event) ** 2)
        res = minimize(par_minimizer,
                       np.array([iv]))
        par = np.zeros(len(res.x)+1)
        par[0] = res.x[0]
        par[1] = t0
        par[2:] = res.x[1:]

        return par