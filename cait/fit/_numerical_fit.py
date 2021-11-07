import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
import numba as nb
from ._saturation import scaled_logistic_curve


@nb.njit
def shift(event, shift_ms=0, timebase_ms=0.04, max_shift=20):
    """TODO"""
    # convert to samples
    shift_sample = int(shift_ms / timebase_ms)
    bound_samples = int(max_shift / timebase_ms)
    shift_sample = np.minimum(shift_sample, bound_samples)
    return np.roll(event, shift_sample)[bound_samples:-bound_samples]  # two times bounds samples shorter


@nb.njit
def template(height, onset, offset, linear, quadratic, cubic, t, sev, timebase_ms=0.04, max_shift=20):
    """TODO"""
    return height * shift(event=sev, shift_ms=onset, timebase_ms=timebase_ms,
                          max_shift=max_shift) + cubic * t ** 3 + quadratic * t ** 2 + linear * t + offset  # two times bounds samples shorter


@nb.njit
def fitfunc(pars, event, t, sev, timebase_ms, max_shift, truncation_flag, truncated=False, t0=None, offset=None,
            linear=None, quadratic=None, cubic=None, saturation_pars=None):
    """TODO"""
    # event and t same length
    if t0 is not None:
        pars[1] = t0
    if offset is not None:
        pars[2] = offset
    if linear is not None:
        pars[3] = linear
    if quadratic is not None:
        pars[4] = quadratic
    if cubic is not None:
        pars[5] = cubic
    if np.abs(pars[1]) > max_shift:
        return 1e8 + 1e5*(np.abs(pars[1] - max_shift))
    if np.abs(pars[0]) > 100:
        return 1e5 * (np.abs(pars[0]))
    height = np.max(event)
    if height > 3 * np.std(event[:500]) and pars[0] < 0:
        return 1e5*(np.abs(pars[0]))
    if truncated:
        if pars[0] < height:
            return 1e8 + 1e4 * (height - pars[0])/height
    bound_samples = int(max_shift / timebase_ms)
    if saturation_pars is not None:
        sev[:] = scaled_logistic_curve(sev, saturation_pars[0], saturation_pars[1], saturation_pars[2],
                                    saturation_pars[3], saturation_pars[4], saturation_pars[5], )
    temp = template(height=pars[0], onset=pars[1], offset=pars[2], linear=pars[3], quadratic=pars[4], cubic=pars[5],
                    t=t, sev=sev, timebase_ms=timebase_ms, max_shift=max_shift)
    retval = np.mean(np.abs(temp[truncation_flag[bound_samples:-bound_samples]] -
                          event[bound_samples:-bound_samples][truncation_flag[bound_samples:-bound_samples]]) ** 2)
    return retval


def array_fitter(args, sev, t, record_length=16384, timebase_ms=0.04, max_shift=20,
                 truncation_level=None, saturation_pars=None):
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
    if any(np.logical_not(truncation_flag)):
        truncated = True
    else:
        truncated = False
    if truncated:
        offset = start_vals[2]
        linear = 0
        quadratic = 0
        cubic = 0
    else:
        offset = None
        linear = None
        quadratic = None
        cubic = None

    # minimize
    res = minimize(
        fun=fitfunc,
        x0=start_vals,
        method='Nelder-Mead',
        args=(ev, t[bound_samples:-bound_samples], sev,  # [bound_samples:-bound_samples],
              timebase_ms, max_shift, truncation_flag, truncated, t0, offset, linear, quadratic, cubic, saturation_pars),
        options={'maxiter': None, 'maxfev': None, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True},
    )

    if truncated:
        res.x[2:6] = offset, linear, quadratic, cubic
    if t0 is not None:
        res.x[1] = t0

    return res.x
