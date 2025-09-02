from typing import List, Union

import numpy as np

from ...fit import pulse_template
from .iteratorbase import IteratorBaseClass


class PulseSimIterator(IteratorBaseClass):
    """
    Iterator object that returns voltage traces superimposed with a SEV.
    The SEV can EITHER be specified by a template array OR by fit parameters [t0, An, At, tau_n, tau_in, tau_t] where the time constants are given in ms. The n-component pulse shape model with parameters [t0, A1, A2, ..., Ak, tau_in, tau_2, ..., tau_k, tau_n] is also supported.

    :param iterator: An iterator (of baselines, stream_chunks, etc.) that you want to superimpose the SEV on.
    :type iterator: IteratorBaseClass
    :param pulse_heights: The pulse heights to scale the SEV. One for each event in 'iterator' and each channel, i.e. with shape ``(iterator.n_channels, len(iterator))``.
    :type pulse_heights: np.ndarray
    :param sev: The SEV to superimpose. Has to match the number of channels of ``iterator`` and its record length, i.e. requires shape ``(iterator.n_channels, iterator.record_length)``. Cannot be specified together with ``sev_fitpars``.
    :type sev: np.ndarray
    :param sev_fitpars: The fit parameters for the SEV to superimpose. Has to match the number of channels of ``iterator``. Cannot be specified together with ``sev``.
    :type sev_fitpars: List[List[float]]
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
                 pulse_heights: List[float], 
                 sev: np.ndarray = None,
                 sev_fitpars: List[List[float]] = None,
                 channels: Union[int, List[int]] = None, 
                 inds: List[int] = None,
                 batch_size: int = None):
        
        if np.sum([x is None for x in [sev, sev_fitpars]]) != 1:
            raise ValueError(f"You have to specify EITHER a sev or its fit parameters. Not both.")
        
        # We will use a flag 'using_fit' to distinguish between the two cases
        # and just call the variable collectively 'sev_or_pars'.
        self._using_fit = sev_fitpars is not None
        sev_or_pars = sev_fitpars if self._using_fit else sev

        # check if dimensions for sev, iterator and pulse_heights match
        sev_or_pars, phs = np.atleast_2d(sev_or_pars), np.atleast_2d(pulse_heights)
        N, nch = len(iterator), iterator.n_channels

        if nch>1 and not sev_or_pars.ndim>1:
            raise ValueError(f"For multi-channel iterators, also 'sev'/'sev_fitpars' must be multi-channel.")
        if nch>1 and not phs.ndim>1:
            raise ValueError(f"For multi-channel iterators, also 'pulse_heights' must be multi-channel.")
        
        if (nch!=sev_or_pars.shape[0]) or (nch!=phs.shape[0]):
            raise ValueError(f"Number of channels in 'iterator', 'sev'/'sev_fitpars', and 'pulse_heights' must be equal. Got {[nch, sev_or_pars.shape[0], phs.shape[0]]}.")

        if N!=phs.shape[-1]:
            raise ValueError(f"Number of events in 'iterator', and 'pulse_heights' must be equal. Got {[N, phs.shape[-1]]}.")
        
        if (not self._using_fit) and (sev_or_pars.shape[-1] != iterator.record_length):
            raise ValueError(f"Length of 'sev' has to match the record length of 'iterator'. Got {[sev_or_pars.shape[-1], iterator.record_length]}.")
        
        if self._using_fit and ((sev_or_pars.shape[-1]-2)%2 != 0):
            raise ValueError(f"Number of parameters in 'sev_fitpars' has to be 2k+2 for k>1. Got {sev_or_pars.shape[-1]}.")
            
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
        super().__init__(
            inds=inds, 
            batch_size=batch_size,
            iterator=iterator,
            sev=np.array(sev).tolist() if sev is not None else None,
            sev_fitpars=np.array(sev_fitpars).tolist() if sev_fitpars is not None else None,
            pulse_heights=np.array(phs).tolist(),
            channels=channels
        )

        # Save values to reconstruct iterator:
        self._params = {'iterator': iterator, 
                        'sev': sev, 
                        'sev_fitpars': sev_fitpars,
                        'pulse_heights': phs,
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
        self._sev_or_pars = sev_or_pars[self._channels]
        self._phs = phs[self._channels]

        # If fit pars are used, we have to evaluate the pulse model at some
        # point for which the time array of the iterator is used:
        self._fit_t = iterator.t
        
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

            if self._using_fit:
                pulse = np.array([
                    pulse_template(self._fit_t, *pars) 
                    for pars in np.atleast_2d(self._sev_or_pars)
                ])
            else:
                pulse = self._sev_or_pars

            return sim_phs*pulse + events
        
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