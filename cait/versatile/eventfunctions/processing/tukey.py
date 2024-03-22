import numpy as np
from scipy.signal.windows import tukey

from ..functionbase import FncBaseClass

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