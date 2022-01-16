import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
import numba as nb
from ._saturation import scaled_logistic_curve


@nb.njit
def shift(event, shift_ms=0, timebase_ms=0.04, max_shift=20):
    """
    Shift an event by a number of samples.

    :param event: The event that is to be shifted.
    :type event: 1D numpy array
    :param shift_ms: The amount of milliseconds by which we want to shift it.
    :type shift_ms: float
    :param timebase_ms: The length of one sample in milliseconds.
    :type timebase_ms: float
    :param max_shift: The maximal shift that can be applied.
    :type max_shift: float
    :return: The shifted event. Attention, this array is shorter than the input array!
    :rtype: 1D numpy array
    """
    # convert to samples
    shift_sample = int(shift_ms / timebase_ms)
    bound_samples = int(max_shift / timebase_ms)
    shift_sample = np.minimum(shift_sample, bound_samples)
    return np.roll(event, shift_sample)[bound_samples:-bound_samples]  # two times bounds samples shorter


@nb.njit
def template(height, onset, offset, linear, quadratic, cubic, t, sev, timebase_ms=0.04, max_shift=20):
    """
    Shift the standard event by a given onset value and superpose a baseline model.

    :param height: The height to which we want to scale the standard event.
    :type height: float
    :param onset: The onset value by which we shift the sev, in ms.
    :type onset: float
    :param offset: The constant component of the baseline.
    :type offset: float
    :param linear: The linear component of the baseline.
    :type linear: float
    :param quadratic: The quadratic component of the baseline.
    :type quadratic: float
    :param cubic: The cubic component of the baseline.
    :type cubic: float
    :param t: The time grid on which the baseline model is evaluated, in ms.
    :type t: 1D numpy array
    :param sev: The standard event.
    :type sev: 1D numpy array
    :param timebase_ms: The length of one sample in milliseconds.
    :type timebase_ms: float
    :param max_shift: The maximal shift that can be applied.
    :type max_shift: float
    :return: The shifted standard event, superposed with the baseline model. Attention, this array is shorter than the sev array!
    :rtype: 1D numpy array
    """
    return height * shift(event=sev, shift_ms=onset, timebase_ms=timebase_ms,
                          max_shift=max_shift) + cubic * t ** 3 + quadratic * t ** 2 + linear * t + offset  # two times bounds samples shorter


@nb.njit
def doshift(arr, num, fill_value=np.nan):
    """
    Shift an array by a number of samples. Attention, this function is deprecated, use numpy.roll instead!

    :param arr: The array that we want to shift.
    :type arr: 1D numpy array
    :param num: The number of samples to shift. This is automatically rounded to the closest integer.
    :type num: float
    :param fill_value: The value with which we want to fill the boarders of the shifted sample.
    :type fill_value: float
    :return: The shifted array.
    :rtype: 1D numpy array
    """
    num_ = int(num)
    if num_ == 0:
        return arr
    elif num_ >= 0:
        return np.concatenate((np.full(num_, fill_value, dtype='f'), arr[:-num_]))
    else:
        return np.concatenate((arr[-num_:], np.full(-num_, fill_value, dtype='f')))


@nb.njit
def lstsqsol(X, y):
    """
    Solution to a least squares regression.

    :param X: The data matrix: (nmbr_samples, nmbr_features).
    :type X: 2-dim numpy array
    :param y: The regression objective.
    :type y: 1-dim numpy array
    :return: The solution, i.e. the fitted coefficients of the features.
    :rtype: 1D array
    """
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)


@nb.njit
def fitfunc(pars, X, event_, sev, bs):
    """
    The fit function for the array fit.

    :param pars: List of length one, the shift that is to be applied.
    :type pars: list
    :param X: The data matrix: (nmbr_samples, nmbr_features). The first column corresponds to the standard event, the
        others to baseline components.
    :type X: 2 dim numpy array
    :param event_: The event that we want to fit.
    :type event_: 1D numpy array
    :param sev: The standard event.
    :type sev: 1D numpy array
    :param bs: The maximum shift value, i.e. the bounds of the minimization problem.
    :type bs: int
    :return: The average rms value of the fit.
    :rtype: float
    """
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
    """
    Get the rms value of the array fit.

    :param pars: The fitted parameters: (height, shift, ... polynomial baseline coefficients ...)
    :type pars: list
    :param X: The data matrix: shape (nmbr_features, record_length).
    :type X: 2D numpy array
    :param event_: The event to fit.
    :type event_: 1D numpy array
    :param sev: The standard event
    :type sev: 1D numpy array
    :param bs: The maximum shift value, i.e. the bounds of the minimization problem.
    :type bs: int
    :return: The average rms value of the fit.
    :rtype: float
    """
    a = np.empty(X.shape[1], dtype='f')
    a[0] = pars[0]
    shift = pars[1]
    a[1:] = pars[2:X.shape[1]+1]
    X[:, 0] = doshift(sev, shift)[bs:-bs]
    return np.mean((np.dot(X, a) - event_) ** 2)


def array_fit(args, sev, t, blcomp, trunclv, bs, no_bl_when_sat):
    """
    The wrapper function which does the array fit for one event.

    :param args: The arguments which we provide to the wrapper: (event, t0). The event is the event to fit
        (1D numpy array), t0 (in milli seconds) is either a float or None. If it is a float,
        the t0 value is not searched but this one
        is used.
    :type args: list
    :param sev: The standard event
    :type sev: 1D float array
    :param t: The time grid on which the baseline model is evaluated, in ms.
    :type t: 1D float array
    :param blcomp: The number of baseline components, i.e. (order+1) of the polynomial which is used as baseline.
    :type blcomp: int
    :param trunclv: The truncation level. Values higher than this are excluded from the fit.
    :type trunclv: float
    :param bs: The bounds for the t0 search, in milli seconds.
    :type bs: float
    :param no_bl_when_sat:
    :type no_bl_when_sat: bool
    :return: The fitted parameters: (height, t0, ... polynomial baseline components ...)
    :rtype: list
    """
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
