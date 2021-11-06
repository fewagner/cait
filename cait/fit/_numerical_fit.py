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
    return np.roll(event, shift_sample)[bound_samples:-bound_samples]  # two times bounds samples shorter


@nb.njit
def template(height, onset, offset, linear, quadratic, cubic, t, sev, timebase_ms=0.04, max_shift=20):
    """TODO"""
    return height * shift(event=sev, shift_ms=onset, timebase_ms=timebase_ms,
                          max_shift=max_shift) + cubic * t ** 3 + quadratic * t ** 2 + linear * t + offset  # two times bounds samples shorter


@nb.njit
def fitfunc(pars, event, t, sev, timebase_ms, max_shift, truncation_flag, t0=None):
    """TODO"""
    # event and t same length
    if t0 is not None:
        pars[1] = t0
    bound_samples = int(max_shift / timebase_ms)
    temp = template(height=pars[0], onset=pars[1], offset=pars[2], linear=pars[3], quadratic=pars[4], cubic=pars[5],
                    t=t, sev=sev, timebase_ms=timebase_ms, max_shift=max_shift)
    return np.mean(np.abs(temp[truncation_flag[bound_samples:-bound_samples]] -
                          event[bound_samples:-bound_samples][truncation_flag[bound_samples:-bound_samples]]) ** 2)


def array_fitter(args, sev, t, record_length=16384, timebase_ms=0.04, max_shift=20, truncation_level=None):
    """TODO"""
    ev, ph_start, t0, t0_start = args
    bound_samples = int(max_shift/timebase_ms)

    # set sv
    start_vals = np.array([ph_start, t0_start, 0, 0, 0, 0])
    start_vals[3], start_vals[2], _, _, _ = linregress(t[bound_samples:int(3 / 16 * record_length)],
                                                       ev[:int(3 / 16 * record_length) - bound_samples])

    # truncation
    if truncation_level is not None:
        truncation_flag = ev < (truncation_level + start_vals[2])
    else:
        truncation_flag = np.ones(ev.shape[0], dtype=bool)

    # minimize
    res = minimize(
        fun=fitfunc,
        x0=start_vals,
        method='Nelder-Mead',
        args=(ev, t[bound_samples:-bound_samples], sev,  # [bound_samples:-bound_samples],
              timebase_ms, max_shift, truncation_flag, t0),
        options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True},
    )

    return res.x
