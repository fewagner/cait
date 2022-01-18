# -----------------------------------------------------
# IMPORTS
# -----------------------------------------------------

import numpy as np
from scipy.optimize import minimize
import numba as nb
import torch
from ._saturation import scaled_logistic_curve


# -----------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------

@nb.njit
def baseline_template_quad(t, c0, c1, c2):
    """
    Template for the baseline fit, with constant linear and
    quadratic component.

    :param t: The time grid.
    :type t: 1D array
    :param c0: Constant component.
    :type c0: float
    :param c1: Linear component.
    :type c1: float
    :param c2: Quadratic component.
    :type c2: float
    :return: The parabola evaluated on the time grid.
    :rtype: 1D array
    """
    return c0 + t * c1 + t ** 2 * c2


@nb.njit
def baseline_template_cubic(t, c0, c1, c2, c3):
    """
    Template for the baseline fit, with constant linear,
    quadratic and cubic component.

    :param t: The time grid.
    :type t: 1D array
    :param c0: Constant component.
    :type c0: float
    :param c1: Linear component.
    :type c1: float
    :param c2: Quadratic component.
    :type c2: float
    :param c3: Cubic component.
    :type c3: float
    :return: The cubic polynomial evaluated on the time grid.
    :rtype: 1D array

    """
    return c0 + t * c1 + t ** 2 * c2 + t ** 3 * c3


@nb.njit
def exponential_bl(t, c0, c1):
    """
    Template for the baseline fit, with constant and exponential component.

    :param t: The time grid.
    :type t: 1D array
    :param c0: Constant component.
    :type c0: float
    :param c1: Exponential component.
    :type c1: float
    :return: The cubic polynomial evaluated on the time grid.
    :rtype: 1D array
    """
    return c0 + np.exp(t * c1)


@nb.njit
def pulse_template(t, t0, An, At, tau_n, tau_in, tau_t):
    """
    Parametric model for the pulse shape, 6 parameters.

    This method was described in "(1995) F. Pröbst et. al., Model for cryogenic particle detectors with superconducting phase
    transition thermometers."

    :param t: 1D array, the time grid; attention, this needs to be provided in compatible units with the fit parameters!

    :param t0: The pulse onset time.
    :type t0: float
    :param An: Amplitude of the first nonthermal pulse component.
    :type An: float
    :param At: Amplitude of the thermal pulse component.
    :type At: float
    :param tau_n: Parameter for decay 1. comp and rise 2. comp.
    :type tau_n: float
    :param tau_in: Parameter for rise 1. comp.
    :type tau_in: float
    :param tau_t: Parameter for decay 2. comp.
    :type tau_t: float
    :return: The pulse model evaluated on the time grid.
    :rtype: 1D array
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

    :param x: The grid.
    :type x: 1D array
    :param p: Length 3, (normalization, mean, stdeviation).
    :type p: 1D array
    :return: Evaluated gauss function.
    :return: 1D array
    """
    A, mu, sigma = p
    return A / sigma / np.sqrt(2 * np.pi) * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


@nb.njit
def sec(t, h, t0, a0, a1, a2, a3, t00, An, At, tau_n, tau_in, tau_t):
    """
    Standard Event Model with Cubic Baseline.

    This method was described in "F. Reindl, Exploring Light Dark Matter With CRESST-II Low-Threshold Detector",
    available via http://mediatum.ub.tum.de/?id=1294132 (accessed on the 9.7.2021).

    :param h: Height of pulse shape.
    :type h: float
    :param t0: Onset of pulse shape.
    :param t0: float
    :param a0: Offset of the baseline.
    :param a0: float
    :param a1: Linear drift component of the baseline.
    :param a1: float
    :param a2: Quadratic drift component of the baseline.
    :param a2: float
    :param a3: Cubic drift component of the baseline.
    :param a3: float
    :param t00: Onset of standard event.
    :param t00: float
    :param An: Amplitude of non-thermal component in standard event.
    :param An: float
    :param At: Amplitude of thermal component in standard event.
    :param At: float
    :param tau_n: Parametric pulse shape model parameter tau_n of standard event.
    :param tau_n: float
    :param tau_in: Parametric pulse shape model parameter tau_in of standard event.
    :param tau_in: float
    :param tau_t: Parametric pulse shape model parameter tau_t of standard event.
    :param tau_t: float
    :return: The pulse model evaluated on the time grid.
    :return: 1D array
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
    Class to store pulse fit models for individual detectors.

    This method was described in "F. Reindl, Exploring Light Dark Matter With CRESST-II Low-Threshold Detector",
    available via http://mediatum.ub.tum.de/?id=1294132 (accessed on the 9.7.2021).

    :param par: 1D array with size 6, the fit parameter of the sev
        (t0, An, At, tau_n, tau_in, tau_t).
    :type par: 1D array with size 6
    :param t: The time grid on which the pulse shape model is evaluated.
    :type t: 1D array
    :param down: Power of 2, the downsample rate of the event for fitting.
    :type down: int
    :param t0_bounds: The lower and upper bounds for the t0 value.
    :type t0_bounds: tuple
    :param truncation_level: All values above this are excluded from the fit.
    :type truncation_level: float
    :param interval_restriction_factor: Value between 0 and 1, the inverval of the event is restricted around
        1/4 by this factor.
    :type interval_restriction_factor: float
    :param saturation_pars: The fit parameter of the saturation curve (A, K, C, Q, B, nu).
    :rtype saturation_pars: 1D array with size 6
    """

    def __init__(self, pm_par, t, down=1, t0_bounds=(-20, 20), truncation_level=None,
                 interval_restriction_factor=None, saturation_pars=None):
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
        self.saturation_pars = saturation_pars

    def sef(self, h, t0, a0):
        """
        Standard Event Model with Flat Baseline.

        :param h: Height of pulse shape.
        :type h: float
        :param t0: Onset of pulse shape.
        :type t0: float
        :param a0: Offset of the baseline.
        :type a0: float
        :return: The pulse model evaluated on the time grid.
        :rtype: 1D array
        """
        x = self.pm_par
        x[0] -= t0
        return h * pulse_template(self.t, *x) + a0

    def sel(self, h, t0, a0, a1):
        """
        Standard Event Model with Linear Baseline.

        :param h: Height of pulse shape.
        :type h: float
        :param t0: Onset of pulse shape.
        :type t0: float
        :param a0: Offset of the baseline.
        :type a0: float
        :param a1: Linear drift component of the baseline.
        :type a1: float
        :return: The pulse model evaluated on the time grid.
        :rtype: 1D array
        """
        x = self.pm_par
        x[0] -= t0
        return h * pulse_template(self.t, *x) + a0 + \
               a1 * (self.t - t0)

    def seq(self, h, t0, a0, a1, a2):
        """
        Standard Event Model with Quadradtic Baseline.

        :param h: Height of pulse shape.
        :type h: float
        :param t0: Onset of pulse shape.
        :type t0: float
        :param a0: Offset of the baseline.
        :type a0: float
        :param a1: Linear drift component of the baseline.
        :type a1: float
        :param a2: Quadratic drift component of the baseline.
        :type a2: float
        :return: The pulse model evaluated on the time grid.
        :rtype: 1D array
        """
        x = self.pm_par
        x[0] -= t0
        return h * pulse_template(self.t, *x) + a0 + \
               a1 * (self.t - t0) + a2 * (self.t - t0) ** 2

    def _wrap_sec(self, h, t0, a0, a1, a2, a3):
        """
        A wrapper function for the pulse shape model.
        """
        return sec(self.t, h, t0, a0, a1, a2, a3, *self.pm_par)

    @staticmethod
    @nb.njit
    def _t0_minimizer(par, h0, a00, a10, a20, a30, event, t, t0_lb, t0_ub, low, up, pm_par, trunc_flag):

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
    def _par_minimizer(par, t0, event, t, low, up, pm_par, trunc_flag):

        # truncate in interval
        fit = sec(t[low:up], par[0], t0, par[1], par[2], par[3], par[4], pm_par[0],
                  pm_par[1], pm_par[2], pm_par[3], pm_par[4], pm_par[5])

        # truncate in height
        out = np.sum((fit[trunc_flag] - event[trunc_flag]) ** 2)

        return out

    @staticmethod
    @nb.njit
    def _t0_minimizer_sat(par, h0, a00, a10, a20, a30, event, t, t0_lb, t0_ub, low, up, pm_par, trunc_flag,
                          A, K, C, Q, B, nu):

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

            fit = scaled_logistic_curve(fit, A, K, C, Q, B, nu)

            out += np.sum((event - fit) ** 2)

        return out

    @staticmethod
    @nb.njit
    def _par_minimizer_sat(par, t0, event, t, low, up, pm_par, trunc_flag,
                           A, K, C, Q, B, nu):

        # truncate in interval
        fit = sec(t[low:up], par[0], t0, par[1], par[2], par[3], par[4], pm_par[0],
                  pm_par[1], pm_par[2], pm_par[3], pm_par[4], pm_par[5])

        fit = scaled_logistic_curve(fit, A, K, C, Q, B, nu)

        out = np.sum((fit - event) ** 2)

        return out

    def fit_cubic(self, pars):
        """
        Calculates the standard event fit parameters with a cubic baseline model.

        :param pars: The event to fit, the fixed onset value, a start value for the onset.
        :type pars: list of (1D array, float, float)
        :return: The sev fit parameters.
        :rtype: 1D array of length 6
        """

        event, t0, t0_start = pars

        if self.down > 1:
            event = np.mean(event.reshape(int(len(event) / self.down), self.down), axis=1)

        offset = np.mean(event[:int(len(event) / 8)])
        event -= offset

        if self.truncation_level is not None:
            truncation_flag = event < self.truncation_level
            truncated = np.any(truncation_flag is False)
            if truncated:
                last_saturated = np.where(truncation_flag is False)[-1]
                h0 = self.truncation_level / sec(self.t, 1, t0_start, 0, 0, 0, 0, *self.pm_par)[last_saturated]
            else:
                h0 = np.max(event)
        else:
            truncated = False
            truncation_flag = np.ones(len(event), dtype=bool)
            h0 = np.max(event)

        # find d, height and k approx
        a00 = 0  # np.mean(event[:1000])
        a10 = (event[-1] - sec(self.t, h0, t0_start, 0, 0, 0, 0, *self.pm_par)[-1] - event[0]) / (
                self.t[-1] - self.t[0])
        a20 = 0
        a30 = 0

        if t0 is None:
            # fit t0
            if self.saturation_pars is None and not truncated:  # no saturation and truncation
                res = minimize(fun=self._t0_minimizer,
                               x0=np.array([t0_start]),
                               method='nelder-mead',
                               args=(h0, a00, a10, a20, a30, event, self.t, self.t0_bounds[0], self.t0_bounds[1],
                                     self.low, self.up, self.pm_par, truncation_flag),
                               options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8,
                                        'adaptive': True},
                               )
                t0 = res.x
            elif self.saturation_pars is not None:  # in case we have saturation curve
                res = minimize(fun=self._t0_minimizer_sat,
                               x0=np.array([t0_start]),
                               method='nelder-mead',
                               args=(h0, a00, a10, a20, a30, event, self.t, self.t0_bounds[0], self.t0_bounds[1],
                                     self.low, self.up, self.pm_par, truncation_flag,
                                     self.saturation_pars[0], self.saturation_pars[1], self.saturation_pars[2],
                                     self.saturation_pars[3], self.saturation_pars[4], self.saturation_pars[5]),
                               options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8,
                                        'adaptive': True},
                               )
                t0 = res.x
            elif truncated:
                # in truncated case we fit the height first
                res = minimize(self._par_minimizer_sat,
                               x0=np.array([h0, a00, a10, a20, a30]),
                               method='nelder-mead',
                               args=(t0_start, event, self.t, self.low, self.up, self.pm_par, truncation_flag,
                                     self.saturation_pars[0], self.saturation_pars[1], self.saturation_pars[2],
                                     self.saturation_pars[3], self.saturation_pars[4], self.saturation_pars[5]),
                               options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                               )
                h, a0, a1, a2, a3 = res.x
            else:
                raise NotImplementedError('We should never reach this code.')

        # fit height, d and k with fixed t0
        if self.saturation_pars is not None:  # case with saturation curve
            res = minimize(self._par_minimizer_sat,
                           x0=np.array([h0, a00, a10, a20, a30]),
                           method='nelder-mead',
                           args=(t0, event, self.t, self.low, self.up, self.pm_par, truncation_flag,
                                 self.saturation_pars[0], self.saturation_pars[1], self.saturation_pars[2],
                                 self.saturation_pars[3], self.saturation_pars[4], self.saturation_pars[5]),
                           options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                           )
            h, a0, a1, a2, a3 = res.x
        elif not truncated:  # no saturation and no truncation
            res = minimize(self._par_minimizer,
                           x0=np.array([h0, a00, a10, a20, a30]),
                           method='nelder-mead',
                           args=(t0, event, self.t, self.low, self.up, self.pm_par, truncation_flag),
                           options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                           )
            h, a0, a1, a2, a3 = res.x
        elif truncated:  # truncated fit
            # in truncated case we fit only now the onset
            res = minimize(fun=self._t0_minimizer,
                           x0=np.array([t0_start]),
                           method='nelder-mead',
                           args=(h, a0, a1, a2, a3, event, self.t, self.t0_bounds[0], self.t0_bounds[1],
                                 self.low, self.up, self.pm_par, truncation_flag),
                           options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True},
                           )
            t0 = res.x
        else:
            raise NotImplementedError('We should never reach this code.')

        a0 += offset

        return h, t0, a0, a1, a2, a3
