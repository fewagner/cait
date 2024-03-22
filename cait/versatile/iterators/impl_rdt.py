from typing import Union, List

import numpy as np

from .iteratorbase import IteratorBaseClass

class RDTIterator(IteratorBaseClass):
    """
    Iterator object that returns voltage traces for given channels and indices of an RDTChannel instance. 

    :param rdt_channel: An RDTChannel instance.
    :type rdt_channel: RDTChannel
    :param channels: The channels that we are interested in. Has to be a subset of `rdt_channel.key`. If None, all channels given by `rdt_channel.key` are considered. Defaults to None.
    :type channels: Union[int, List[int]]
    :param inds: The indices of `rdt_channel` that we want to iterate over. If None, all indices are considered. Defaults to None
    :type inds: Union[int, List[int]]

    :return: Iterable object
    :rtype: RDTIterator
    """
    def __init__(self, rdt_channel, channels: Union[int, List[int]] = None, inds: Union[int, List[int]] = None):
        super().__init__()

        self._rdt_channel = rdt_channel

        # Helper variable to make check consistent
        key = (self._rdt_channel.key,) if isinstance(self._rdt_channel.key, int) else self._rdt_channel.key
        
        if channels is None: channels = list(key)
        self._channels = [channels] if isinstance(channels, int) else channels

        if not set(self._channels).issubset(set(key)):
            raise ValueError(f"Not all values in channels ({channels}) are present in key {self._rdt_channel.key}.")

        if inds is None: inds = np.arange(len(self._rdt_channel))
        self._inds = [inds] if isinstance(inds, int) else [int(i) for i in inds]

        # Save index array for channel selection in __next__ (the values in self._channels correspond to 
        # actual channel numbers in the RDT file. Here, we are interested in the indices of the already selected
        # channels, i.e., e.g., 0 for the first channel of an RDTChannel instance)
        channel_select = np.array([key.index(x) for x in self._channels])
        self._channel_select = channel_select if channel_select.size > 1 else channel_select[0]

        # Save values to reconstruct iterator:
        self._params = {'rdt_channel': rdt_channel, 'channels': self._channels, 'inds': self._inds}
    
    def __len__(self):
        return len(self._inds)

    def __enter__(self):
        return self # Just to be consistent with H5Iterator
    
    def __exit__(self, typ, val, tb):
        ... # Just to be consistent with H5Iterator
    
    def __iter__(self):
        self._current_ind = 0
        return self

    def _next_raw(self):
        if self._current_ind < len(self._inds):
            out = self._rdt_channel[self._channel_select, self._inds[self._current_ind]]
            self._current_ind += 1

            return out
        else:
            raise StopIteration
        
    @property
    def record_length(self):
        return self._rdt_channel._rdt_file.record_length
    
    @property
    def dt_us(self):
        return self._rdt_channel._rdt_file.dt_us
    
    @property
    def timestamps(self):
        return self._rdt_channel.timestamps[self._params["inds"]]

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