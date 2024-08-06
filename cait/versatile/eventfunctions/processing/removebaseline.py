import numpy as np

from ..functionbase import FncBaseClass
from ..scalarfunctions.fitbaseline import FitBaseline

class RemoveBaseline(FncBaseClass):
    """
    Remove baseline of given baseline model from an event voltage trace and return the new event.
    Also works for multiple channels simultaneously.

    :param fit_baseline: Dictionary of keyword arguments that are passed on to :class:`FitBaseline`.
    :type fit_baseline: dict

    :return: Event with baseline removed
    :rtype: numpy.ndarray

    **Example:**
    ::
    
        import cait.versatile as vai

        # Construct mock data (which provides event iterator)
        md = vai.MockData()
        it = md.get_event_iterator()[0]

        # View effect of removing baseline on events
        vai.Preview(it, vai.RemoveBaseline())

    .. image:: media/RemoveBaseline_preview.png
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
    
    @property
    def batch_support(self):
        return 'trivial'
        
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