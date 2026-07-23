import numpy as np

from .helper import Lags
from ..functionbase import FncBaseClass

class Align(FncBaseClass):
    """
    Align a voltage trace relative to a reference trace by shifting its indices.
    This function calculates the lags (shift between two signals) using cross-correlations and shifts it afterwards. Note that the output has the same size as the input, i.e. samples are just periodically shifted.

    :param ref_event: The reference voltage trace.
    :type ref_event: np.ndarray
 
    :return: Shifted event.
    :rtype: np.ndarray

    .. code-block:: python

        ev_it = dh.get_event_iterator("events")
        sev = vai.SEV().from_dh(dh)
        f = vai.Align(sev)
        aligned_events = vai.apply(f, ev_it)
    """
    def __init__(self, ref_event: np.array):
        self._ref_event = ref_event
        self._lags = Lags(ref_event)

    def __call__(self, event):
        self._shifted_event = np.roll(event, self._lags(event))
        return self._shifted_event
    
    @property
    def batch_support(self):
        return 'none'
    
    def preview(self, event):
        self(event)
        return dict(line = {'event': [None, event],
                            'reference event': [None, self._ref_event],
                            'shifted event': [None, self._shifted_event]})