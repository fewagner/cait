from typing import List, Union

import numpy as np

from .iteratorbase import IteratorBaseClass

class PulseSimIterator(IteratorBaseClass):
    """
    Iterator object that returns voltage traces superimposed with a SEV. 

    :param iterator: An iterator (of baselines, stream_chunks, etc.) that you want to superimpose the SEV on.
    :type iterator: IteratorBaseClass
    :param sev: The SEV to superimpose. Has to match the number of channels of 'iterator' and its record length, i.e. requires shape ``(iterator.n_channels, iterator.record_length)``.
    :type sev: np.ndarray
    :param pulse_heights: The pulse heights to scale the SEV. One for each event in 'iterator' and each channel, i.e. with shape ``(iterator.n_channels, len(iterator))``.
    :type pulse_heights: np.ndarray
    :param channels: The channels that we are interested in. Has to be a subset of iterator's channels. If None, all channels are considered. Defaults to None.
    :type channels: Union[int, List[int]]
    :param inds: The indices of 'iterator' that we want to iterate over. If None, all indices are considered. Defaults to None
    :type inds: Union[int, List[int]]
    :param batch_size: The number of events to be returned at once (these are all read together). There will be a trade-off: large batch_sizes cause faster read speed but increase the memory usage.
    :type batch_size: int

    :return: Iterable object
    :rtype: PulseSimIterator
    """
    def __init__(self, 
                 iterator: IteratorBaseClass, 
                 sev: np.ndarray,
                 pulse_heights: List[float], 
                 channels: Union[int, List[int]] = None, 
                 inds: List[int] = None,
                 batch_size: int = None):
        
        # check if dimensions for sev, iterator and pulse_heights match
        sev, phs = np.array(sev), np.array(pulse_heights)
        N, nch = len(iterator), iterator.n_channels

        if nch>1 and not sev.ndim>1:
            raise ValueError(f"For multi-channel iterators, also 'sev' must be multi-channel.")
        if nch>1 and not phs.ndim>1:
            raise ValueError(f"For multi-channel iterators, also 'pulse_heights' must be multi-channel.")
        
        if (nch!=sev.shape[0]) or (nch!=phs.shape[0]):
            raise ValueError(f"Number of channels in 'iterator', 'sev', and 'pulse_heights' must be equal. Got {[nch, sev.shape[0], phs.shape[0]]}.")

        if N!=phs.shape[-1]:
            raise ValueError(f"Number of events in 'iterator', and 'pulse_heights' must be equal. Got {[N, phs.shape[-1]]}.")
        
        if sev.shape[-1] != iterator.record_length:
            raise ValueError(f"Length of 'sev' has to match the record length of 'iterator'. Got {[sev.shape[-1], iterator.record_length]}.")
            
        if channels is None: channels = list(range(iterator.n_channels)) 

        if isinstance(channels, int):
            self._channels = channels
            self._n_channels = 1
        elif isinstance(channels, list):
            self._channels = channels if len(channels)>1 else channels[0]
            self._n_channels = len(channels)
        else:
            raise TypeError(f"Unsupported type {type(channels)} for input argument 'channels'")

        if inds is None: inds = np.arange(len(iterator))
        inds = [inds] if isinstance(inds, int) else [int(i) for i in inds]
        
        # Does batch handling and creates properties self._inds, self.uses_batches, and self.n_batches
        super().__init__(inds=inds, batch_size=batch_size)

        # Save values to reconstruct iterator:
        self._params = {'iterator': iterator, 
                        'sev': sev, 
                        'pulse_heights': pulse_heights,
                        'channels': self._channels, 
                        'inds': inds, 
                        'batch_size': batch_size}

        # We use the tools implemented by iterator to already 
        # select the correct channels, indices, and batch_size.
        # If we do so, we can just iterate it to get the correct
        # (batched) events.
        if batch_size is None:
            self._it = iterator[self._channels, inds].flatten() 
        else:
            self._it = iterator[self._channels, inds].with_batchsize(batch_size)

        # For sev and phs, we have to manually select the channels. 
        # Furthermore, we make use of the self._inds created by
        # super().__init__() to slice the correct (batched) pulse_heights
        # in _next_raw().
        self._sev = sev[self._channels]
        self._phs = pulse_heights[self._channels]
        
    def __enter__(self):
        # enter underlying iterator
        self._it.__enter__()
        return self
    
    def __exit__(self, typ, val, tb):
        self._it.__exit__(typ, val, tb)
    
    def __iter__(self):
        self._current_batch_ind = 0
        # start iteration of underlying iterator
        # (this way we can retrieve elements faster)
        self._itit = self._it.__iter__()
        return self

    def _next_raw(self):
        if self._current_batch_ind < self.n_batches:
            event_inds_in_batch = self._inds[self._current_batch_ind]
            self._current_batch_ind += 1
            
            sim_phs = self._phs[..., event_inds_in_batch].T[...,None]
            events = next(self._itit)

            return sim_phs*self._sev + events
        
        else:
            raise StopIteration
        
    @property
    def record_length(self):
        return self._it.record_length
    
    @property
    def dt_us(self):
        return self._it.dt_us
    
    @property
    def ds_start_us(self):
        return self._it.ds_start_us

    @property
    def timestamps(self):
        return self._it.timestamps
    
    @property
    def n_channels(self):
        return self._n_channels
    
    @property
    def _slice_info(self):
        return (self._params, ('channels', 'inds'))