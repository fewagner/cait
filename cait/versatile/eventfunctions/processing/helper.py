import numpy as np
import scipy as sp

from ..functionbase import FncBaseClass

class Unity(FncBaseClass):
    """
    Class that returns events unaltered. This is mostly used as a helper class to preview raw voltage traces.

    :param t: The time array (in milliseconds) corresponding to the voltage trace.
    :type t: np.ndarray
    """
    def __init__(self, t: np.ndarray = None):
        self._t = t

    def __call__(self, event):
        return event
    
    @property
    def batch_support(self):
        return 'trivial'
    
    def preview(self, event):
        if event.ndim > 1:
            lines = {f'channel {k}': [self._t, ev] for k, ev in enumerate(event)}
        else:
            lines = {'channel 0': [self._t, event]}

        return dict(line=lines, 
                    axes={"xaxis": {"label": "time (ms)" if self._t is not None else "data index"},
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
    
    @property
    def batch_support(self):
        return 'none'