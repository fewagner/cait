from typing import Union, List
import warnings

import numpy as np
from scipy.optimize import curve_fit

from ..functionbase import FitFncBaseClass

# We do not use scipy's parameter error estimation anyways so we can suppress this warning
warnings.filterwarnings("ignore", "Covariance of the parameters could not be estimated")

def exponential_decay(x, a, b, c):
    return a*np.exp(-b*x) + c

class FitBaseline(FitFncBaseClass):
    """
    Fit voltage traces with a polynomial or decaying exponential and return the fit parameters as well as the RMS.
    Also works for multiple channels simultaneously.

    :param model: Order of the polynomial, 'exponential'/'exp' (exponential baseline model; in the case of a failed fit, falls back to constant baseline, i.e. model=0), or 'voltage_minimum' (subtracting a constant value, which is close to the minimum of the voltage trace before the pulse -- with some fluctuation mitigation. This works better than the standard method when there is a pileup in the pre-trigger region), defaults to 0, i.e. a constant baseline.
    :type model: Union[int, str]
    :param where: Specifies a subset of data points to be used in the fit: Either a boolean flag of the same length of the voltage traces, a slice object (e.g. slice(0,50) for using the first 50 data points), or a float. If a float `where` is passed, the first `int(where*record_length)` samples are used (e.g. if `where=1/8`, the first 1/8th of the record window is used). Defaults to `slice(None, None, None)`.
    :type where: Union[List[bool], slice, float]
    :param xdata: x-data to use for the fit (has no effect for `order=0`). Specifying xdata is not necessary in general but if you want your fit parameters to have physical units (e.g. time constants) instead of just samples, you may use this option. Defaults to `None`, in which case `xdata=np.linspace(0,1,record_length)`.
    :type xdata: List[float]

    :return: Fit parameter(s) and RMS as a tuple.
    :rtype: Tuple[Union[float, numpy.ndarray], float]

    **Example:**

    .. code-block:: python

        import cait.versatile as vai

        # Construct mock data (which provides event iterator)
        md = vai.MockData()
        it = md.get_event_iterator()[0]

        # View effect of fitting baseline on events
        # We specify that for the fit, 1/8th of the record window should be used,
        # that we fit with a degree-0-polynomial (i.e. constant)
        # Specifying the xdata is not necessary, but it lets us plot in terms of time
        vai.Preview(it, vai.FitBaseline(where=1/8, model=0, xdata=it.t), xlabel="Time (ms)")

    .. image:: media/FitBaseline_preview.png
    """
    def __init__(self, model: Union[int, str] = 0, where: Union[List[bool], slice, float] = slice(None, None, None), xdata: List[float] = None):
        if not isinstance(model, (str, int)):
            raise NotImplementedError(f"Unsupported type '{type(model)}' for input 'order'.")
        elif isinstance(model, str) and model not in ['exponential',
                                                      'exp',
                                                      'voltage_minimum']:
            raise NotImplementedError(f"Unrecognized baseline model '{model}'.")
        elif isinstance(model, int) and model < 0:
            raise NotImplementedError(f"Polynomial order '{model}' is not supported, only non-zero integers are.")

        self._model = model
        self._where = where
        self._xdata = xdata
        self._A = None

        if isinstance(model, str) and model == 'voltage_minimum':
            from .mainparameters import MainParameters
            self._mp = MainParameters()

    def __call__(self, event):
        event = np.array(event)
        # ATTENTION: This is only set once, i.e. data has to have same length
        if self._xdata is None:
            self._xdata = np.linspace(0, 1, np.array(event).shape[-1])

        # Reshape array if ndim > 2
        orig_shape = None
        if event.ndim > 2:
            orig_shape = event.shape
            event = event.reshape(-1, event.shape[-1])

        # ATTENTION: this is set only once
        if isinstance(self._where, (float, int)):
            assert self._where <= 1, ValueError(f"If 'where' is a float, must be <= 1.  Got {self._where}")
            self._where = slice(0, int(np.array(event).shape[-1]*self._where))

        # Shortcut for constant baseline model
        if self._model == 0:
            self._fitpar =  np.mean(event[..., self._where], axis=-1)[..., None]
            self._rms = np.std(event[..., self._where], axis=-1)

        # Adapted model which is more stable in case of pre-trigger pile-up
        elif self._model == 'voltage_minimum':
            # Get onset
            t0s = np.astype(np.atleast_1d(self._mp(event)[1]), np.int32)
            # Easily handle multiple channels
            events = np.atleast_2d(event)
            # Model was developed for a fixed record length. Here, the indices
            # 600 and 100 were found to work well. Consequently, we scale the
            # indices now for an arbitrary record length.
            k1, k2 = int(600/2**14*event.shape[-1]), int(100/2**14*event.shape[-1])

            self._fitpar = np.zeros(events.shape[0])
            self._rms = np.zeros(events.shape[0])

            # If anyone finds a way to do this more elegantly, feel free to change!
            for i, (t0, event) in enumerate(zip(t0s, events)):
                t_min = k1
                if int(t0) > k1:
                    # Get baseline value as average of samples before the minimum. Minimum likely to sit at negative noise fluctuation, so do not include in average. Average before and not after to not average over part of the pulse.
                    t_min += np.argmin(event[k1:t0])

                self._fitpar[i] = np.mean(event[t_min-k1:t_min-k2])
                self._rms[i] = np.std(event[t_min-k1:t_min-k2])

            self._fitpar = np.squeeze(self._fitpar)[..., None]
            self._rms = np.squeeze(self._rms)[..., None]

        else:

            # Exponential fit
            if self._model in ['exponential', 'exp']:
                try:
                    if np.array(event).ndim > 1:
                        self._fitpar = np.array(
                            [
                              curve_fit(exponential_decay,
                                        self._xdata[self._where],
                                        event[k, self._where],
                                        bounds=([0, 0, -np.inf],[np.inf,np.inf,np.inf]))[0]
                              for k in range(np.array(event).shape[0])
                            ]
                        )

                        self._rms = np.array(
                            [
                                np.sqrt(np.mean((event[k, self._where] - exponential_decay(self._xdata[self._where], *self._fitpar[k]))**2))
                                for k in range(np.array(event).shape[0])
                            ]
                        )

                    else:
                        self._fitpar, *_ = curve_fit(exponential_decay,
                                                    self._xdata[self._where],
                                                    event[self._where],
                                                    bounds=([0, 0, -np.inf],[np.inf, np.inf, np.inf]))

                        self._rms = np.sqrt(np.mean((event[self._where] - exponential_decay(self._xdata[self._where], *self._fitpar))**2))

                except RuntimeError as e:
                    # Failed to fit an exponential; fall back to model=0
                    if event.ndim > 1:
                        self._fitpar = np.array([(0, 0, np.mean(event[k, self._where])) for k in range(event.shape[0])])
                        self._rms = np.array([np.std(event[k, self._where]) for k in range(event.shape[0])])
                    else:
                        self._fitpar = np.array([(0, 0, np.mean(event[self._where]))])
                        self._rms = np.array([np.std(event[self._where])])


            # Polynomial fit
            else:
                if self._A is None:
                    self._A = np.array([self._xdata[self._where]**k for k in range(self._model+1)]).T

                par, err, *_ = np.linalg.lstsq(self._A, event[..., self._where].T, rcond=None)
                self._fitpar = par.T
                self._rms = np.sqrt(err / self._xdata[self._where].shape[0])


        if orig_shape is not None:
            self._fitpar = self._fitpar.reshape(*orig_shape[:-1], -1)
            self._rms = self._rms.reshape(*orig_shape[:-1], -1)

        return self._fitpar, self._rms

    @property
    def batch_support(self):
        return 'full'

    def model(self, x: List, par: List):
        """

        """
        par = np.array(par)

        orig_shape = None
        if par.ndim > 2:
            orig_shape = par.shape
            par = par.reshape(-1, par.shape[-1])

        if self._model in ['exponential', 'exp']:
            if par.shape[-1] != 3:
                raise ValueError(f"3 parameters are required to fully describe this model, {len(par)} given.")

            if par.ndim > 1: # i.e. we have multiple channels
                if orig_shape is None:
                    return np.array([exponential_decay(x, *par[k]) for k in range(par.shape[0])])
                else:
                    return np.array([exponential_decay(x, *par[k]) for k in range(par.shape[0])]).reshape(
                            *orig_shape[:-1], -1)
            else:
                return exponential_decay(x, *par)
        elif self._model == "voltage_minimum":
            if par.ndim > 1: # i.e. we have multiple channels
                if orig_shape is None:
                    return np.array([np.ones_like(x)*par[k] for k in range(par.shape[0])])
                else:
                    return np.array([np.ones_like(x)*par[k] for k in range(par.shape[0])]).reshape(
                            *orig_shape[:-1], -1)
            else:
                return np.ones_like(x)*par
        else:
            if par.shape[-1] != self._model+1:
                raise ValueError(f"{self._model+1} parameter(s) are required to fully describe this model.")

            if par.ndim > 1: # i.e. we have multiple channels
                if orig_shape is None:
                    return np.array(
                        [np.sum(np.array([par[k][j]*x**j for j in range(self._model+1)]), axis=0) for k in range(par.shape[0])]
                    )
                else:
                    return np.array(
                        [np.sum(np.array([par[k][j]*x**j for j in range(self._model+1)]), axis=0) for k in range(par.shape[0])]
                    ).reshape(
                            *orig_shape[:-1], -1)

            else:
                return np.sum(np.array([par[k]*x**k for k in range(self._model+1)]), axis=0)

    def preview(self, event):
        # Call function (this will set all class attributes to be accessed for plotting)
        self(event)

        # This happens for constant baseline fit (self._xdata is never needed and therefore never
        # computed)
        if self._xdata is None:
            self._xdata = np.arange(np.array(event).shape[-1])

        # self._fitpar is not an array for self._model = 0 or 'voltage_minimum'
        if self._model in [0, 'voltage_minimum']:
            par = np.array([self._fitpar]).T
        else:
            par = self._fitpar

        # Reconstruct fit function
        fit = self.model(self._xdata, par)

        if np.ndim(event) > 1:
            d = dict()
            for i in range(np.ndim(event)):
                d[f'channel {i}'] = [self._xdata, event[i]]
                d[f'baseline fit channel {i}'] = [self._xdata, fit[i]]
        else:


            d = {'event': [self._xdata, event],
                 'baseline fit': [self._xdata, fit]}

        return dict(line = d)


    @property
    def where(self):
        return self._where


    @property
    def xdata(self):
        return self._xdata
