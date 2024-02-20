# FUNCTIONS THAT ARE MEANT TO BE APPLIED TO SINGLE EVENTS AND ALSO RETURN SINGLE EVENTS

import numpy as np
import scipy as sp
from scipy.signal.windows import tukey

from .abstract_functions import FncBaseClass
from .fit_functions import FitBaseline

########### HELPER FUNCTIONS ###########
class Unity(FncBaseClass):
    """
    Class that returns events unaltered. This is mostly used as a helper class to preview raw voltage traces.
    """
    def __call__(self, event):
        return event
    
    def preview(self, event):
        if event.ndim > 1:
            lines = {f'channel {k}': [None, ev] for k, ev in enumerate(event)}
        else:
            lines = {'channel 0': [None, event]}

        return dict(line=lines, 
                    axes={"xaxis": {"label": "data index"},
                          "yaxis": {"label": "data (V)"}
                         })
    
class Lags(FncBaseClass):
    def __init__(self, ref_event: np.ndarray):
        if np.array(ref_event).ndim > 1:
            raise Exception("Only one-dimensional events are supported.")
        self._ref_event = np.array(ref_event)

    def __call__(self, event):
        corr = sp.signal.correlate(self._ref_event, event, mode="full", method="fft")
        lags = sp.signal.correlation_lags(self._ref_event.size, event.size, mode="full")
        self._lag = lags[np.argmax(corr)]
        return self._lag

############ EVENT FUNCTIONS ############
class Downsample(FncBaseClass):
    """
    Downsample an event by a given factor (which has to be a factor of the event's length).
    Also works for multiple channels simultaneously.

    :param down: The factor by which to downsample the voltage trace.
    :type down: int
 
    :return: Downsampled event.
    :rtype: np.ndarray
    """
    def __init__(self, down: int = 2):
        self._down = down

    def __call__(self, event):
        if event.ndim > 1:
            shape = (event.shape[0], int(event.shape[-1]/self._down), self._down)
        else:
            shape = (int(event.shape[-1]/self._down), self._down)
            
        self._downsampled = np.mean(np.reshape(event, shape), axis=-1)
        return self._downsampled
    
    def preview(self, event):
        self(event)
        x = np.arange(event.shape[-1])
        x_down = np.mean(np.reshape(x,(int(len(x)/self._down), self._down)), axis=1)
        
        if np.ndim(event) > 1:
            d = dict()
            for i in range(np.ndim(event)):
                d[f'channel {i}'] = [x, event[i]]
                d[f'downsampled channel {i}'] = [x_down, self._downsampled[i]]
        else:
            d = {'event': [x, event],
                 'downsampled event': [x_down, self._downsampled]}
            
        return dict(line = d)
    
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

class BoxCarSmoothing(FncBaseClass):
    # TODO: implement for multiple channels
    """
    Apply box-car-smoothing (moving average) to a voltage trace.

    :param length: Length (in samples) of the moving average. Defaults to 50.
    :type length: int
 
    :return: Smoothed event.
    :rtype: np.ndarray
    """
    def __init__(self, length: int = 50):
        self._length = length

    def __call__(self, event):
        event = np.pad(event, self._length, 'edge')
        event = 1/self._length * np.convolve(event, np.array([1]).repeat(self._length), 'same')
        self._smooth_event =  event[self._length:-self._length]

        return self._smooth_event
    
    def preview(self, event):
        self(event)
        return dict(line = {'event': [None, event],
                            'smooth event': [None, self._smooth_event]})
   
class TukeyFiltering(FncBaseClass):
    """
    Apply the Tukey window function to a voltage trace. 
    Also works for multiple channels simultaneously.

    :param alpha: The parameter of the Tukey window function. Defaults to 0.25.
    :type alpha: float
 
    :return: Event with applied window function.
    :rtype: np.ndarray
    """
    def __init__(self, alpha: float = 0.25):
        self._alpha = alpha

    def __call__(self, event):
        self._new_event = event*tukey(event.shape[-1], alpha=self._alpha)
        return self._new_event
    
    def preview(self, event) -> dict:
        self(event)
        if np.ndim(event) > 1:
            d = dict()
            for i in range(np.ndim(event)):
                d[f'channel {i}'] = [None, event[i]]
                d[f'after window channel {i}'] = [None, self._new_event[i]]
        else:
            d = {'event': [None, event],
                 'after window': [None, self._new_event]}
        return dict(line = d)

class OptimumFiltering(FncBaseClass):
    """
    Apply an optimum filter to a voltage trace. 
    Works for multiple channels simultaneously if optimum filter is also given for multiple channels.

    :param of: The optimum filter to use.
    :type of: np.ndarray
 
    :return: Filtered event.
    :rtype: np.ndarray

    >>> ev_it = dh.get_event_iterator("events")
    >>> of = vai.OF().from_dh(dh)
    >>> f = vai.OptimumFiltering(of)
    >>> filtered_events = vai.apply(f, ev_it)
    """
    def __init__(self, of):
        self._of = of

    def __call__(self, event):
        if np.ndim(event) != np.ndim(self._of):
            raise ValueError(f"Number of dimensions of OF ({np.ndim(self._of)}) and event ({np.ndim(event)}) are incompatible.")
        
        # Note that this works for multiple channels simultaneously
        self._filtered_event = np.fft.irfft(np.fft.rfft(event)*self._of)
        return self._filtered_event
    
    def preview(self, event) -> dict:
        self(event)
        if np.ndim(event) > 1:
            d = dict()
            for i in range(np.ndim(event)):
                d[f'channel {i}'] = [None, event[i]]
                d[f'filtered channel {i}'] = [None, self._filtered_event[i]]
        else:
            d = {'event': [None, event],
                 'filtered event': [None, self._filtered_event]}
        return dict(line = d)
        
class Align(FncBaseClass):
    """
    Align a voltage trace relative to a reference trace by shifting its indices.
    This function calculates the lags (shift between two signals) using cross-correlations and shifts it afterwards. Note that the output has the same size as the input, i.e. samples are just periodically shifted.

    :param ref_event: The reference voltage trace.
    :type ref_event: np.ndarray
 
    :return: Shifted event.
    :rtype: np.ndarray

    >>> ev_it = dh.get_event_iterator("events")
    >>> sev = vai.SEV().from_dh(dh)
    >>> f = vai.Align(sev)
    >>> aligned_events = vai.apply(f, ev_it)
    """
    def __init__(self, ref_event: np.array):
        self._ref_event = ref_event
        self._lags = Lags(ref_event)

    def __call__(self, event):
        self._shifted_event = np.roll(event, self._lags(event))
        return self._shifted_event
    
    def preview(self, event):
        self(event)
        return dict(line = {'event': [None, event],
                            'reference event': [None, self._ref_event],
                            'shifted event': [None, self._shifted_event]})