import numpy as np
import scipy as sp

from ..functionbase import FncBaseClass

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