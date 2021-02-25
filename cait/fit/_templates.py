# -----------------------------------------------------
# IMPORTS
# -----------------------------------------------------

import numpy as np
from scipy.optimize import minimize
import numba as nb


# -----------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------

@nb.njit
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


@nb.njit
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


@nb.njit
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
    n = t.shape[0]
    cond = t > t0
    # print(cond)
    pulse = np.zeros(n)
    # pulse = 1e-5*np.ones(n)
    pulse[cond] = (An * (np.exp(-(t[cond] - t0) / tau_n) - np.exp(-(t[cond] - t0) / tau_in)) + \
                   At * (np.exp(-(t[cond] - t0) / tau_t) - np.exp(-(t[cond] - t0) / tau_n)))
    return pulse


# Define gauss function
@nb.njit
def gauss(x, *p):
    """
    Evaluate a gauss function on a given grid x

    :param x: 1D array, the grid
    :param p: 1D array length 3: (normalization, mean, stdeviation)
    :return: 1D array of same length than x, evaluated gauss function
    """
    A, mu, sigma = p
    return A / sigma / np.sqrt(2 * np.pi) * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


@nb.njit
def sec(t, h, t0, a0, a1, a2, a3, t00, An, At, tau_n, tau_in, tau_t):
    """
    Standard Event Model with Cubic Baseline

    :param h: float, hight of pulse shape
    :param t0: float, onset of pulse shape
    :param a0: float, offset of the baseline
    :param a1: float, linear drift component of the baseline
    :param a2: float, quadratic drift component of the baseline
    :param a3: float, cubic drift component of the baseline
    :return: 1D array, the pulse model evaluated on the time grid
    """
    t0_comb = t0 + t00
    cond = t > t0_comb
    pulse = a0 + \
            a1 * t + a2 * t ** 2 + \
            a3 * t ** 3
    pulse[cond] += h * (An * (np.exp(-(t[cond] - t0_comb) / tau_n) - np.exp(-(t[cond] - t0_comb) / tau_in)) + \
                        At * (np.exp(-(t[cond] - t0_comb) / tau_t) - np.exp(-(t[cond] - t0_comb) / tau_n)))

    return pulse


# -----------------------------------------------------
# CLASSES
# -----------------------------------------------------

class sev_fit_template:
    """
    Class to store pulse fit models for individual detectors

    :param par: 1D array with size 6, the fit parameter of the sev
        (t0, An, At, tau_n, tau_in, tau_t)
    :param t: 1D array, the time grid on which the pulse shape model is evaluated
    :param down: int, power of 2, the downsample rate of the event for fitting
    """

    def __init__(self, pm_par, t, down=1, t0_bounds=(-20, 20), truncation_level=None, interval_restriction_factor=None):
        self.pm_par = np.array(pm_par)
        self.down = down
        if down > 1:
            t = np.mean(t.reshape(int(len(t) / down), down), axis=1)  # only first of the grouped time values
        self.t = np.array(t)
        self.t0_bounds = t0_bounds
        self.truncation_level = truncation_level
        if interval_restriction_factor is not None:
            if interval_restriction_factor > 0.8 or interval_restriction_factor < 0:
                raise KeyError("interval_restriction_factor must be float > 0 and < 0.8!")
            self.interval_restriction_factor = interval_restriction_factor
            self.low = int(self.interval_restriction_factor * (len(self.t) - 1) / 4)
            self.up = int((len(self.t) - 1) * (1 - (3 / 4) * self.interval_restriction_factor))
            print(self.low, self.up)
        else:
            self.low = 0
            self.up = len(self.t)

    def sef(self, h, t0, a0):
        """
        Standard Event Model with Flat Baseline

        :param h: float, hight of pulse shape
        :param t0: float, onset of pulse shape
        :param a0: float, offset of the baseline
        :return: 1D array, the pulse model evaluated on the time grid
        """
        x = self.pm_par
        x[0] -= t0
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
        x[0] -= t0
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
        x[0] -= t0
        return h * pulse_template(self.t, *x) + a0 + \
               a1 * (self.t - t0) + a2 * (self.t - t0) ** 2

    def wrap_sec(self, h, t0, a0, a1, a2, a3):
        """
        A wrapper function for the pulse shape model.
        """
        return sec(self.t, h, t0, a0, a1, a2, a3, *self.pm_par)

    @staticmethod
    @nb.njit
    def t0_minimizer(par, h0, a00, a10, a20, a30, event, t, t0_lb, t0_ub, low, up, pm_par, trunc_flag):
        # TODO

        out = 0

        # t0 bounds
        if par[0] < t0_lb:
            out += 1.0e100 - 1.0e100 * (par[0] - t0_ub)
        elif par[0] > t0_ub:
            out += 1.0e100 - 1.0e100 * (t0_ub - par[0])
        else:
            # truncate in interval
            fit = sec(t[low:up], h0, par[0], a00, a10, a20, a30,
                      pm_par[0], pm_par[1], pm_par[2], pm_par[3],
                      pm_par[4], pm_par[5])

            # truncate in height
            out += np.sum((event[trunc_flag] - fit[trunc_flag]) ** 2)

        return out

    @staticmethod
    @nb.njit
    def par_minimizer(par, t0, event, t, low, up, pm_par, trunc_flag):
        # TODO

        # truncate in interval
        fit = sec(t[low:up], par[0], t0, par[1], par[2], par[3], par[4], pm_par[0],
                  pm_par[1], pm_par[2], pm_par[3], pm_par[4], pm_par[5])

        # truncate in height
        out = np.sum((fit[trunc_flag] - event[trunc_flag]) ** 2)

        return out

    def fit_cubic(self, event):
        """
        Calculates the standard event fit parameters with a cubic baseline model.

        :param event: The event to fit.
        :type event: 1D array
        :return: The sev fit parameters.
        :rtype: array
        """

        if self.down > 1:
            event = np.mean(event.reshape(int(len(event) / self.down), self.down), axis=1)

        offset = np.mean(event[:int(len(event) / 8)])
        event -= offset

        if self.truncation_level is not None:
            truncation_flag = event < self.truncation_level
        else:
            truncation_flag = np.ones(len(event), dtype=bool)

        # find d, height and k approx
        a00 = 0  # np.mean(event[:1000])
        h0 = np.max(event)
        a10 = (event[-1] - sec(self.t, h0, 0, 0, 0, 0, 0, *self.pm_par)[-1] - event[0]) / (self.t[-1] - self.t[0])
        a20 = 0
        a30 = 0

        # fit t0
        res = minimize(fun=self.t0_minimizer,
                       x0=np.array([-3]),
                       method='nelder-mead',
                       args=(h0, a00, a10, a20, a30, event, self.t, self.t0_bounds[0], self.t0_bounds[1],
                             self.low, self.up, self.pm_par, truncation_flag,),
                       options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True},
                       )
        t0 = res.x

        # fit height, d and k with fixed t0
        res = minimize(self.par_minimizer,
                       x0=np.array([h0, a00, a10, a20, a30]),
                       method='nelder-mead',
                       args=(t0, event, self.t, self.low, self.up, self.pm_par, truncation_flag,),
                       options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                       )
        h, a0, a1, a2, a3 = res.x
        a0 += offset

        return h, t0, a0, a1, a2, a3
