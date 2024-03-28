import numpy as np
import scipy as sp

from ..functionbase import FncBaseClass

class BoxCarSmoothing(FncBaseClass):
    """
    Apply box-car-smoothing (moving average) to a voltage trace.
    Also works for multiple channels simultaneously.

    :param length: Length (in samples) of the moving average. Defaults to 50.
    :type length: int
 
    :return: Smoothed event.
    :rtype: np.ndarray
    """
    def __init__(self, length: int = 50):
        self._length = length

    def __call__(self, event):
        event = np.array(event)
        n = event.ndim
        shape = (event.shape[0], self._length) if n > 1 else (self._length, )
        pad = ((0, 0), (self._length, self._length)) if n > 1 else self._length

        event = np.pad(event, pad, 'edge')
        event = 1/self._length * sp.signal.fftconvolve(event, 
                                    np.ones(shape), 
                                    mode="same", 
                                    axes=-1)
        self._smooth_event =  event[..., self._length:-self._length]

        return self._smooth_event
    
    @property
    def batch_support(self):
        return 'trivial'
    
    def preview(self, event):
        self(event)
        if np.ndim(event) > 1:
            d = dict()
            for i in range(np.ndim(event)):
                d[f'channel {i}'] = [None, event[i]]
                d[f'smoothed channel {i}'] = [None, self._smooth_event[i]]
        else:
            d = {'event': [None, event],
                 'after window': [None, self._smooth_event]}
        return dict(line = d)