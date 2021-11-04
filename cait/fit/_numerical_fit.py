import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
import numba as nb


@nb.njit
def shift(event, shift_ms=0, timebase_ms=0.04, max_shift=20):
    """TODO"""
    # convert to samples
    shift_sample = int(shift_ms / timebase_ms)
    bound_samples = int(max_shift / timebase_ms)
    return np.roll(event, shift_sample)[bound_samples:-bound_samples]


@nb.njit
def template(height, onset, offset, linear, quadratic, cubic, t, sev, timebase_ms=0.04, max_shift=20):
    """TODO"""
    return height * shift(event=sev, shift_ms=onset, timebase_ms=timebase_ms,
                          max_shift=max_shift) + cubic * t ** 3 + quadratic * t ** 2 + linear * t + offset


@nb.njit
def fitfunc(pars, event, t, sev, timebase_ms, max_shift):
    """TODO"""
    bound_samples = int(max_shift / timebase_ms)
    temp = template(height=pars[0], onset=pars[1], cubic=pars[2], quadratic=pars[3], linear=pars[4], offset=pars[5],
                    t=t, sev=sev, timebase_ms=timebase_ms, max_shift=max_shift)
    return np.mean(np.abs(temp - event[bound_samples:-bound_samples]) ** 2)


def array_fitter(args, t, sev, record_length=16384, timebase_ms=0.04, max_shift=20):
    """TODO"""
    ev, ph_start = args
    bound_samples = int(max_shift/timebase_ms)

    # set sv

    start_vals = np.array([ph_start, -3, 0, 0, 0, 0])
    start_vals[4], start_vals[5], _, _, _ = linregress(t[bound_samples:int(3 / 16 * record_length)],
                                                       ev[:int(3 / 16 * record_length) - bound_samples])

    # minimize
    res = minimize(
        fun=fitfunc,
        x0=start_vals,  # res.x,
        method='Nelder-Mead',
        args=(ev, t[2 * bound_samples:-2 * bound_samples], sev[bound_samples:-bound_samples], timebase_ms,
              max_shift),
        options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True},
    )

    return res.x
