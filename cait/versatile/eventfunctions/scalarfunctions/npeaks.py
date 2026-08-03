from functools import partial
from typing import Union

import numpy as np

import cait.versatile as vai

from ..functionbase import ScalarFncBaseclass


class NPeaks(ScalarFncBaseclass):
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
    _outputs = [
        ("n_peaks", int),
    ]

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
        # Reshape array if ndim > 2
        orig_shape = None
        if event.ndim > 2:
            orig_shape = event.shape
            event = event.reshape(-1, event.shape[-1])

        if self._trigger is None:
            record_length = np.array(event).shape[-1]
            window_size = int(record_length*self._window_size)
            self._trigger = partial(vai.trigger_zscore, 
                                    record_length=window_size,
                                    threshold=self._threshold)
            
        if event.ndim == 1:
            # Single channel, no batches
            self._num_triggers = len(self._trigger(event)[0])
        else:
            # Multiple channels and/or batches
            self._num_triggers = np.array([len(self._trigger(event[ch])[0]) for ch in range(event.shape[0])])
        if not orig_shape is None: 
            # Done only if multiple channels and multiple batches
            self._num_triggers = self._num_triggers.reshape(*orig_shape[:-1], -1)

        return self._num_triggers
    
    @property
    def batch_support(self):
        return 'full'
    
    def preview(self, event):
        n = self(event)
        x = np.arange(event.shape[-1])
        
        l = {'event': [x, event]}
        s = {'triggers': [x[self._trigger_inds] if n>0 else [],
                             event[self._trigger_inds] if n>0 else []]}
            
        return dict(line=l, scatter=s)