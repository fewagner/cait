from typing import Union, List

import numpy as np

from .iteratorbase import IteratorBaseClass

class MockIterator(IteratorBaseClass):
    """
    Iterator object that returns voltage traces for given channels and indices of an RDTChannel instance. 

    :param mock: A MockData instance.
    :type mock: MockData
    :param channels: The channels that we are interested in. Has to be a subset of ``mock``'s channels. If None, all channels are considered. Defaults to None.
    :type channels: Union[int, List[int]]
    :param inds: The indices of ``mock`` that we want to iterate over. If None, all indices are considered. Defaults to None
    :type inds: Union[int, List[int]]
    :param batch_size: The number of events to be returned at once (these are all read together). There will be a trade-off: large batch_sizes cause faster read speed but increase the memory usage.
    :type batch_size: int

    :return: Iterable object
    :rtype: MockIterator
    """
    def __init__(self, 
                 mock, 
                 channels: Union[int, List[int]] = None, 
                 inds: Union[int, List[int]] = None, 
                 batch_size: int = None):
        
        if inds is None: inds = np.arange(mock.n_events)
        inds = [inds] if isinstance(inds, int) else [int(i) for i in inds]

        # Does batch handling and creates properties self._inds, self.uses_batches, and self.n_batches
        super().__init__(inds=inds, batch_size=batch_size)

        self._mock = mock
        
        if channels is None: channels = list(range(mock.n_channels))
        self._channels = [channels] if isinstance(channels, int) else channels

        # Save values to reconstruct iterator:
        self._params = {'mock': mock, 
                        'channels': self._channels, 
                        'inds': inds,
                        'batch_size': batch_size}

    def __enter__(self):
        return self
    
    def __exit__(self, typ, val, tb):
        ...
    
    def __iter__(self):
        self._current_batch_ind = 0
        return self

    def _next_raw(self):
        if self._current_batch_ind < self.n_batches:
            event_inds_in_batch = self._inds[self._current_batch_ind]
            self._current_batch_ind += 1

            out = self._mock.get_event(event_inds_in_batch, self._channels)

            if not self.uses_batches or self.n_channels == 1:
                out = np.squeeze(out)
            if self.uses_batches and out.ndim == 1:
                out = out[None,:]

            return out
        else:
            raise StopIteration
        
    @property
    def record_length(self):
        return self._mock.record_length
    
    @property
    def dt_us(self):
        return self._mock.dt_us
    
    @property
    def timestamps(self):
        return self._mock.timestamps[self._params["inds"]]

    @property
    def n_channels(self):
        return len(self._channels)
    
    @property
    def _slice_info(self):
        return (self._params, ('channels', 'inds'))