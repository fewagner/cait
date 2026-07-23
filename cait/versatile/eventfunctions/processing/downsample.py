import numpy as np

from ..functionbase import FncBaseClass

class Downsample(FncBaseClass):
    """
    Downsample an event by a given factor (which has to be a factor of the event's length).
    Also works for multiple channels simultaneously.

    :param down: The factor by which to downsample the voltage trace.
    :type down: int
 
    :return: Downsampled event.
    :rtype: np.ndarray

    **Example:**

    .. code-block:: python
    
        import cait.versatile as vai

        # Construct mock data (which provides event iterator)
        md = vai.MockData()
        it = md.get_event_iterator()[0].with_processing(vai.RemoveBaseline())

        # View effect of downsample on events
        vai.Preview(it, vai.Downsample(16))

    .. image:: media/Downsample_preview.png
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
    
    @property
    def batch_support(self):
        return 'trivial'
    
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