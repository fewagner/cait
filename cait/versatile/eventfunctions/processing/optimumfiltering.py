import numpy as np

from ..functionbase import FncBaseClass

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