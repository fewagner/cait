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
    :param batch_size: The number of events to be returned at once (these are all read together). There will be a trade-off: large batch_sizes cause faster read speed but increase the memory usage.
    :type batch_size: int

    :return: Iterable object
    :rtype: RDTIterator
    """
    def __init__(self, 
                 rdt_channel, 
                 channels: Union[int, List[int]] = None, 
                 inds: Union[int, List[int]] = None,
                 batch_size: int = None):

        self._rdt_channel = rdt_channel

        # Helper variable to make check consistent
        key = (self._rdt_channel.key,) if isinstance(self._rdt_channel.key, int) else self._rdt_channel.key
        
        if channels is None: channels = list(key)
        self._channels = [channels] if isinstance(channels, int) else channels

        if not set(self._channels).issubset(set(key)):
            raise ValueError(f"Not all values in channels ({channels}) are present in key {self._rdt_channel.key}.")

        if inds is None: inds = np.arange(len(self._rdt_channel))
        inds = [inds] if isinstance(inds, int) else [int(i) for i in inds]

        # Does batch handling and creates properties self._inds, self.uses_batches, and self.n_batches
        # Also sets up serializing
        super().__init__(rdt_channel=rdt_channel,
                         channels=channels,
                         inds=inds, 
                         batch_size=batch_size)

        # Save index array for channel selection in __next__ (the values in self._channels correspond to 
        # actual channel numbers in the RDT file. Here, we are interested in the indices of the already selected
        # channels, i.e., e.g., 0 for the first channel of an RDTChannel instance)
        channel_select = np.array([key.index(x) for x in self._channels])
        self._channel_select = channel_select if channel_select.size > 1 else channel_select[0]

        # Save values to reconstruct iterator:
        self._params = {'rdt_channel': rdt_channel, 
                        'channels': self._channels, 
                        'inds': inds,
                        'batch_size': batch_size}
    
    def __iter__(self):
        self._current_batch_ind = 0
        return self

    def _next_raw(self):
        if self._current_batch_ind < self.n_batches:
            event_inds_in_batch = self._inds[self._current_batch_ind]
            self._current_batch_ind += 1

            out = self._rdt_channel[self._channel_select, event_inds_in_batch]

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
    def ds_start_us(self):
        return self._rdt_channel.start_us
    
    @property
    def timestamps(self):
        return self._rdt_channel.timestamps[self._params["inds"]]
    
    @property
    def n_channels(self):
        return len(self._channels)
    
    @property
    def _slice_info(self):
        return (self._params, ('channels', 'inds'))