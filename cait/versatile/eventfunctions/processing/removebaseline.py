import numpy as np

from ..functionbase import FncBaseClass
from ..scalarfunctions.fitbaseline import FitBaseline

class RemoveBaseline(FncBaseClass):
    """
    Remove baseline of given baseline model from an event voltage trace and return the new event.
    Also works for multiple channels simultaneously.

    :param fit_baseline: Dictionary of keyword arguments that are passed on to :class:`FitBaseline`. See below.
    :type fit_baseline: dict

    :return: Event with baseline removed
    :rtype: numpy.ndarray

    Parameters for :class:`FitBaseline`:
    :param model: Order of the polynomial or 'exponential', defaults to 0.
    :type model: Union[int, str]
    :param where: Specifies a subset of data points to be used in the fit: Either a boolean flag of the same length of the voltage traces, a slice object (e.g. slice(0,50) for using the first 50 data points), or a float. If a float `where` is passed, the first `int(where)*record_length` samples are used (e.g. if `where=1/8`, the first 1/8th of the record window is used). Defaults to `1/8`.  
    :type where: Union[List[bool], slice, int]
    :param xdata: x-data to use for the fit (has no effect for `order=0`). Specifying x-data is not necessary in general but if you want your fit parameters to have physical units (e.g. time constants) instead of just samples, you may use this option. Defaults to `None`.
    :type xdata: List[float]
    """
    def __init__(self, fit_baseline: dict = {'model': 0, 'where': 1/8, 'xdata': None}):
        self._fit_baseline = FitBaseline(**fit_baseline)

        if 'xdata' in fit_baseline.keys():
            self._xdata = fit_baseline['xdata']
        else:
            self._xdata = None

    def __call__(self, event):
        par, *_ = self._fit_baseline(event)
        if self._fit_baseline._model == 0:
            if np.ndim(event) > 1:
                self._shifted_event = event - np.array(par)[:, None]
            else:
                self._shifted_event = event - par
        else:
            # ATTENTION: This is set only once! (we have to set it here because 
            # previously we didn't know the length of 'event')
            if self._xdata is None: self._xdata = np.linspace(0, 1, np.array(event).shape[-1])
            self._shifted_event = event - self._fit_baseline.model(self._xdata, par)

        return self._shifted_event
        
    def preview(self, event) -> dict:
        self(event)
        
        if np.ndim(event) > 1:
            d = dict()
            for i in range(np.ndim(event)):
                d[f'channel {i}'] = [self._xdata, event[i]]
                d[f'baseline removed channel {i}'] = [self._xdata, self._shifted_event[i]]
        else:
            
        
            d = {'event': [self._xdata, event],
                 'baseline removed': [self._xdata, self._shifted_event]}
            
        return dict(line = d)