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

    :return: Iterable object
    :rtype: MockIterator
    """
    def __init__(self, mock, channels: Union[int, List[int]] = None, inds: Union[int, List[int]] = None):
        super().__init__()

        self._mock = mock
        
        if channels is None: channels = list(range(mock.n_channels))
        self._channels = [channels] if isinstance(channels, int) else channels

        if inds is None: inds = np.arange(mock.n_events)
        self._inds = [inds] if isinstance(inds, int) else [int(i) for i in inds]

        # Save values to reconstruct iterator:
        self._params = {'mock': mock, 'channels': self._channels, 'inds': self._inds}
    
    def __len__(self):
        return len(self._inds)

    def __enter__(self):
        return self
    
    def __exit__(self, typ, val, tb):
        ...
    
    def __iter__(self):
        self._current_ind = 0
        return self

    def _next_raw(self):
        if self._current_ind < len(self._inds):
            out = self._mock.get_event(self._inds[self._current_ind], self._channels)
            self._current_ind += 1

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
    def uses_batches(self):
        return False # Just to be consistent with H5Iterator
    
    @property
    def n_batches(self):
        return len(self)
    
    @property
    def n_channels(self):
        return len(self._channels)
    
    @property
    def _slice_info(self):
        return (self._params, ('channels', 'inds'))