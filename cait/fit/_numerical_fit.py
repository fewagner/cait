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
def doshift(arr, num, fill_value=np.nan):
    """TODO"""
    num_ = int(num)
    if num_ == 0:
        return arr
    elif num_ >= 0:
        return np.concatenate((np.full(num_, fill_value, dtype='f'), arr[:-num_]))
    else:
        return np.concatenate((arr[-num_:], np.full(-num_, fill_value, dtype='f')))


@nb.njit
def lstsqsol(X, y):
    """TODO"""
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)


@nb.njit
def fitfunc(pars, X, event_, sev, bs):
    """TODO"""
    shift, = pars
    if np.abs(np.floor(shift)) > bs:
        return 1e8 + 1e5 * np.abs(shift - bs)
    X[:, 0] = doshift(sev, shift)[bs:-bs]
    try:
        rms = np.mean((np.dot(X, lstsqsol(X, event_)) - event_) ** 2)
    except:
        print('Error in array fit, filling up with zeros!')
        rms = 0
    return rms


@nb.njit
def arr_fit_rms(pars, X, event_, sev, bs):
    """TODO"""
    a = np.empty(X.shape[1], dtype='f')
    a[0] = pars[0]
    shift = pars[1]
    a[1:] = pars[2:X.shape[1]+1]
    X[:, 0] = doshift(sev, shift)[bs:-bs]
    return np.mean((np.dot(X, a) - event_) ** 2)


def array_fit(args, sev, t, blcomp, trunclv, bs, no_bl_when_sat):
    """TODO"""
    assert blcomp in [1, 2, 3, 4], 'blcomp must be in [1,2,3,4]!'
    event, t0 = args
    record_length = event.shape[0]
    A = np.empty((record_length - 2 * bs, 1 + blcomp), dtype='f')
    pars = np.zeros(6, dtype=float)
    t_ = t[bs:-bs]
    for i in range(blcomp):
        A[:, i + 1] = t_ ** i

    event_ = event[bs:-bs]
    if trunclv is not None:
        truncflag = event - np.mean(event_[bs:bs + 500]) < trunclv
        truncflag[:bs] = True
        truncflag[-bs:] = True
    else:
        truncflag = np.ones(event.shape[0], dtype=bool)

    if all(truncflag):  # no truncation
        if t0 is None:
            res = minimize(fitfunc,
                           x0=np.array([0, ]),
                           args=(A, event_, sev, bs),
                           method="Powell").x[0]
        else:
            res = t0/(t[1] - t[0])
        A[:, 0] = doshift(sev, res)[bs:-bs]
        try:
            x = lstsqsol(A, event_)
        except:
            print('Error in array fit, filling up with zeros!')
            x = np.zeros(A.shape[1])
    else:
        if no_bl_when_sat:
            A = A[:, :2]
            blcomp = 1
        truncflag_ = truncflag[bs:-bs]
        A_ = A[truncflag_, :]
        if t0 is None:
            res = minimize(fitfunc,
                           x0=np.array([0, ]),
                           args=(A_, event_[truncflag_], sev[truncflag], bs),
                           method="Powell").x[0]
        else:
            res = t0/(t[1] - t[0])
        A_[:, 0] = doshift(sev, res)[bs:-bs][truncflag_]
        try:
            x = lstsqsol(A_, event_[truncflag_])
        except:
            print('Error in array fit, filling up with zeros!')
            x = np.zeros(A_.shape[1])

    pars[0] = x[0]
    pars[1] = res * (t[1] - t[0])
    for i in range(blcomp):
        pars[2 + i] = x[1 + i]

    return pars
