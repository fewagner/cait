from typing import List, Union
import warnings

import numpy as np
from scipy.optimize import curve_fit

from .abstract_functions import FitFncBaseClass

# We do not use scipy's parameter error estimation anyways so we can suppress this warning
warnings.filterwarnings("ignore", "Covariance of the parameters could not be estimated")

def exponential_decay(x, a, b, c):
    return a*np.exp(-b*x) + c

class FitBaseline(FitFncBaseClass):
    """
    Fit voltage traces with a polynomial or decaying exponential and return the fit parameters as well as the RMS.

    :param model: Order of the polynomial or 'exponential'/'exp', defaults to 0, i.e. a constant baseline.
    :type model: Union[int, str]
    :param where: Specifies a subset of data points to be used in the fit: Either a boolean flag of the same length of the voltage traces, a slice object (e.g. slice(0,50) for using the first 50 data points), or an integer. If an integer `where` is passed, the first `int(1/where)*record_length` samples are used (e.g. if `where=8`, the first 1/8 of the record window is used). Defaults to `slice(None, None, None).  
    :type where: Union[List[bool], slice, int]
    :param xdata: x-data to use for the fit (has no effect for `order=0`). Specifying x-data is not necessary in general but if you want your fit parameters to have physical units (e.g. time constants) instead of just samples, you may use this option. Defaults to `None`.
    :type xdata: List[float]

    :return: Fit parameter(s) and RMS as a tuple.
    :rtype: Tuple[Union[float, numpy.ndarray], float]
    """
    def __init__(self, model: Union[int, str] = 0, where: Union[List[bool], slice, int] = slice(None, None, None), xdata: List[float] = None):
        if type(model) not in [str, int]:
            raise NotImplementedError(f"Unsupported type '{type(model)}' for input 'order'.")
        elif type(model) is str and model not in ['exponential', 'exp']:
            raise NotImplementedError(f"Unrecognized baseline model '{model}'.")
        elif type(model) is int and model < 0:
            raise NotImplementedError(f"Polynomial order '{model}' is not supported, only non-zero integers are.")
        
        self._model = model
        self._where = where
        self._xdata = xdata
        self._A = None

    def __call__(self, event):
        # ATTENTION: this is set only once
        if type(self._where) is int:
            self._where = slice(0, int(len(event)/self._where))

        # Shortcut for constant baseline model
        if self._model == 0:
            self._fitpar =  np.mean(event[self._where])
            self._rms = np.std(event[self._where])
        else:
            # ATTENTION: This is only set once, i.e. data has to have same length 
            if self._xdata is None: 
                self._xdata = np.arange(len(np.array(event)))

            # Exponential fit
            if self._model in ['exponential', 'exp']:
                self._fitpar, *_ = curve_fit(exponential_decay, 
                                            self._xdata[self._where], 
                                            event[self._where],
                                            bounds=([0,0,-np.inf],[np.inf,np.inf,np.inf]))

                self._rms = np.sqrt(np.mean((event[self._where] - exponential_decay(self._xdata[self._where], *self._fitpar))**2))
            
            # Polynomial fit
            else:
                if self._A is None:
                    self._A = np.array([self._xdata[self._where]**k for k in range(self._model+1)]).T

                self._fitpar, err, *_ = np.linalg.lstsq(self._A, event[self._where], rcond=None)
                self._rms = np.sqrt(err)

        return self._fitpar, self._rms
    
    def model(self, x: List, par: List):
        """
        
        """
        if self._model in ['exponential', 'exp']:
            if len(par) != 3:
                raise ValueError(f"3 parameters are required to fully describe this model, {len(par)} given.")
            return exponential_decay(x, *par)
        else:
            if len(par) != self._model+1:
                raise ValueError(f"{self._model+1} parameters are required to fully describe this model.")
        
            return np.sum(np.array([par[k]*x**k for k in range(self._model+1)]), axis=0)
    
    def preview(self, event):
        # Call function (this will set all class attributes to be accessed for plotting)
        self(event)

        # This happens for constant baseline fit (self._xdata is never needed and therefore never
        # computed)
        if self._xdata is None: 
                self._xdata = np.arange(len(np.array(event)))

        # self._fitpar is not an array for self._model = 0
        par = [self._fitpar] if self._model == 0 else self._fitpar

        # Reconstruct fit function
        fit = self.model(self._xdata, par)

        return dict(line={'event': [self._xdata, event],
                          'fit': [self._xdata, fit]
                          })
    