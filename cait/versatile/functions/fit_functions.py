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
    Also works for multiple channels simultaneously.

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
            self._where = slice(0, int(np.array(event).shape[-1]/self._where))

        # Shortcut for constant baseline model
        if self._model == 0:
            self._fitpar =  np.mean(event[..., self._where], axis=-1)
            self._rms = np.std(event[..., self._where], axis=-1)
        else:
            # ATTENTION: This is only set once, i.e. data has to have same length 
            if self._xdata is None: 
                self._xdata = np.arange(np.array(event).shape[-1])

            # Exponential fit
            if self._model in ['exponential', 'exp']:
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
            
            # Polynomial fit
            else:
                if self._A is None:
                    self._A = np.array([self._xdata[self._where]**k for k in range(self._model+1)]).T

                par, err, *_ = np.linalg.lstsq(self._A, event[..., self._where].T, rcond=None)
                self._fitpar = par.T
                self._rms = np.sqrt(err)

        return self._fitpar, self._rms
    
    def model(self, x: List, par: List):
        """
        
        """
        par = np.array(par)
        
        if self._model in ['exponential', 'exp']:
            if par.shape[-1] != 3:
                raise ValueError(f"3 parameters are required to fully describe this model, {len(par)} given.")
                
            if par.ndim > 1: # i.e. we have multiple channels
                return np.array([exponential_decay(x, *par[k]) for k in range(par.shape[0])])
            else:
                return exponential_decay(x, *par)
        else:
            if par.shape[-1] != self._model+1:
                raise ValueError(f"{self._model+1} parameter(s) are required to fully describe this model.")
            
            if par.ndim > 1: # i.e. we have multiple channels
                return np.array(
                    [np.sum(np.array([par[k][j]*x**j for j in range(self._model+1)]), axis=0) for k in range(par.shape[0])]
                )
            else:
                return np.sum(np.array([par[k]*x**k for k in range(self._model+1)]), axis=0)
    
    def preview(self, event):
        # Call function (this will set all class attributes to be accessed for plotting)
        self(event)

        # This happens for constant baseline fit (self._xdata is never needed and therefore never
        # computed)
        if self._xdata is None: 
                self._xdata = np.arange(np.array(event).shape[-1])
                
        # self._fitpar is not an array for self._model = 0
        par = np.array([self._fitpar]).T if self._model == 0 else self._fitpar
            
        # Reconstruct fit function
        fit = self.model(self._xdata, par)
                
        if np.ndim(event) > 1:
            d = dict()
            for i in range(np.ndim(event)):
                d[f'channel {i}'] = [self._xdata, event[i]]
                d[f'fit channel {i}'] = [self._xdata, fit[i]]
        else:
            
        
            d = {'event': [self._xdata, event],
                 'fit': [self._xdata, fit]}
            
        return dict(line = d)
    
class ArrayFit(FitFncBaseClass):
    """
    
    """
    def __init__(self, sev: np.ndarray):
        raise NotImplementedError
        self._sev = np.array(sev)

    def __call__(self, event):
        self._fitted_sev = None
        self._height = None
        self._rms = None

        return self._height, self._rms
    
    def model(self, x: List, par: List):
        """
        
        """
        ...
    
    def preview(self, event):
        self(event)
        return dict(line={'event': [None, event],
                          'fitted sev': [None, self._fitted_sev]
                          })

class SEVFit(FitFncBaseClass):
    """
    
    """
    def __init__(self):
        raise NotImplementedError

    def __call__(self, event):
        ...
    
    def model(self, x: List, par: List):
        """
        
        """
        ...
    
    def preview(self, event):
        ...