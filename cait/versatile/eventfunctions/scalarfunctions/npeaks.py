from typing import Union
from functools import partial

import numpy as np
import cait.versatile as vai

from ..functionbase import FncBaseClass

class NPeaks(FncBaseClass):
    """
    Determine the number of peaks in an event by applying a moving z-score trigger to the trace.

    :param window_size: The size of the sliding window for the z-score trigger. If it's an integer, this will be the number of samples in the window, if it's a float, the number will be scaled to the record_length of the events (e.g. if 1/20, the window will be 1/20th of the record_length). The larger the window, the more robust the trigger is. However, you will miss potential triggers in the beginning of the event because the first sample that can reliably searched after applying the moving z-score is at ``window_size``. Defaults to 1/20
    :type window_size: Union[int, float], optional
    :param threshold: The threshold (in sigmas) for the trigger, defaults to 3.5
    :type threshold: float, optional

    :return: Number of peaks found.
    :rtype: int

    **Example:**

    .. code-block:: python

        import numpy as np
        import cait.versatile as vai

        md = vai.MockData()
        sev = md.sev[0]
        it = md.get_event_iterator()[0]

        def pileup(event):  
            # generate a random pulse height between 0 and 3
            height = 3*np.random.rand()
            # generate a random position for the pulse after the first pulse
            pos = np.random.randint(md.record_length//8, 3*md.record_length//4)
            
            # overlay new event on top of original
            new_event = event.copy()
            new_event[pos:] += sev[:(md.record_length-pos)]
            
            return new_event

        vai.Preview(it.with_processing(pileup), vai.NPeaks(window_size=1/20, threshold=5))

    .. image:: media/NPeaks_preview.png
    """
    def __init__(self, window_size: Union[int, float] = 1/20, threshold: float = 3.5):
        self._window_size = window_size
        self._threshold = threshold

        if isinstance(window_size, int):
            self._trigger = partial(vai.trigger_zscore, 
                                    record_length=window_size,
                                    threshold=threshold)
        else:
            self._trigger = None

        self._trigger_inds = list()

    def __call__(self, event):
        if np.array(event).ndim > 1:
            raise NotImplementedError(f"Multi-channel events are not supported by {self.__class__.__name__}")
        if self._trigger is None:
            record_length = np.array(event).shape[-1]
            window_size = int(record_length*self._window_size)
            self._trigger = partial(vai.trigger_zscore, 
                                    record_length=window_size,
                                    threshold=self._threshold)
            
        self._trigger_inds, _ = self._trigger(event)

        return len(self._trigger_inds)
    
    @property
    def batch_support(self):
        return 'none'
    
    def preview(self, event):
        n = self(event)
        x = np.arange(event.shape[-1])
        
        l = {'event': [x, event]}
        s = {'triggers': [x[self._trigger_inds] if n>0 else [],
                             event[self._trigger_inds] if n>0 else []]}
            
        return dict(line=l, scatter=s)