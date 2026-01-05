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
    :type pulse_heights: List[List[float]]
    :param shift_samples: If specified for all elements in ``pulse_heights`` (i.e. also for all channels). The respective x-shift (in samples) is applied to the superimposed pulse. If you use a ``sev`` (see below), the edges of the shifted array are padded with the average of the first/last 10 samples. If you use ``sev_fitpars`` (see below), no padding is required because the fit can just be extrapolated.
    :type shift_samples: List[List[int]]
    :param shift_subsamples: If specified for all elements in ``pulse_heights`` (i.e. also for all channels). The respective x-shift (in fractional samples) is applied to the superimposed pulse. If you use a ``sev`` (see below), the array is linearly interpolated between samples. If you use ``sev_fitpars`` (see below), the model is evaluated at the intermediate points. Note that all values have to be in the interval [0, 1), corresponding to shifts between zero and one sample.
    :type shift_subsamples: List[List[float]]
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

    .. code-block:: python

        import numpy as np
        import scipy as sp

        import cait.versatile as vai
        from cait.versatile.iterators import PulseSimIterator

        # Use mock data. You will have a more meaningful iterator.
        md = vai.MockData()
        sev = md.sev[0]

        # This is just a cheeky way to get an iterator of random noise. 
        # You will have actual noise.
        noise_it = md.get_event_iterator()[0].with_processing(lambda x: sp.stats.norm.rvs(loc=0, scale=0.1, size=md.record_length))

        # Define random pulse heights
        sim_phs = sp.stats.uniform.rvs(size=len(noise_it))

        # Set up the iterator containing simulated pulses on top of
        # the noise traces.
        # Check out the docstring to learn about advanced ways to 
        # simulate events, e.g. by adding template shifts.
        pulse_sim_it = PulseSimIterator(
            iterator=noise_it, 
            pulse_heights=sim_phs, 
            sev=sev,
        )

        # Preview your pulses.
        vai.Preview(pulse_sim_it)

        # In a next step, you could for example filter the simulated
        # events, determine their pulse heights, and estimate the baseline
        # resolution (the code below is a minimal example and definitely not
        # perfect!). In this case, it makes more sense to simulate a fixed
        # pulse height:
        pulse_sim_it_fixed_ph = PulseSimIterator(
            iterator=noise_it, 
            pulse_heights=0.5*np.ones_like(sim_phs), 
            sev=sev,
        )

        reconstructed_phs = vai.apply(
            np.max,
            pulse_sim_it_fixed_ph.with_processing(vai.OptimumFiltering(md.of[0]))
        )

        # Plot a histogram of reconstructed pulse heights.
        vai.Histogram(reconstructed_phs)
    """
    def __init__(self, 
                 iterator: IteratorBaseClass, 
                 pulse_heights: List[List[float]],
                 shift_samples: List[List[int]] = None,
                 shift_subsamples: List[List[float]] = None,
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
        
        # Sanitize shifts
        if shift_samples is not None:
            shift_samples = np.atleast_2d(shift_samples).astype(np.int32)
            if shift_samples.shape != phs.shape:
                raise ValueError(f"The shapes of 'shift_samples' and 'pulse_heights' have to be identical. Got {shift_samples.shape} and {phs.shape}.")
        if shift_subsamples is not None:
            shift_subsamples = np.atleast_2d(shift_subsamples).astype(np.float32)
            if shift_subsamples.shape != phs.shape:
                raise ValueError(f"The shapes of 'shift_subsamples' and 'pulse_heights' have to be identical. Got {shift_subsamples.shape} and {phs.shape}.")
            if not np.all((shift_subsamples>=0)*(shift_subsamples<1)):
                raise ValueError("All values in 'shift_subsamples' have to be in the interval [0, 1).")
            
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
            shift_samples=np.array(shift_samples).tolist() if shift_samples is not None else None,
            shift_subsamples=np.array(shift_subsamples).tolist() if shift_subsamples is not None else None,
            channels=channels,
        )

        # Save values to reconstruct iterator:
        self._params = {'iterator': iterator, 
                        'sev': sev, 
                        'sev_fitpars': sev_fitpars,
                        'pulse_heights': phs,
                        'shift_samples': shift_samples,
                        'shift_subsamples': shift_subsamples,
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

        # We will always have shift samples (easier handling) and
        # just set them to zero if not specified
        if shift_samples is not None:
            self._shift_samples = shift_samples[self._channels]
        else:
            self._shift_samples = np.zeros(self._phs.shape, dtype=np.int32)

        if shift_subsamples is not None:
            self._shift_subsamples = shift_subsamples[self._channels]
        else:
            self._shift_subsamples = np.zeros(self._phs.shape, dtype=np.float32)

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
    
    def _shift_fit_pars_and_eval(self, pars: np.ndarray, ks: np.ndarray, zs: np.ndarray):
        # for all fitpar-tuple in pars, the corresponding shift in ks (sample)
        # and zs (subsample) is applied and the parameters are evaluated on self.t 
        temp_array = pars.reshape(-1, pars.shape[-1])
        temp_k = ks.flatten()
        temp_z = zs.flatten()

        new_array = np.zeros((temp_array.shape[0], self.record_length))

        for j in range(temp_array.shape[0]):
            p = temp_array[j]
            k = temp_k[j]
            z = temp_z[j]
            new_array[j, :] = pulse_template(
                self.t, 
                *(p[0] + (k + z)*self.dt_us/1000, *p[1:]),
            )

        return np.reshape(new_array, tuple(list(pars.shape)[:-1]+[self.record_length]))
    
    def _shift_array(self, pulses: np.ndarray, ks: np.ndarray, zs: np.ndarray):
        # return the input array with all pulses shifted by k (samples) 
        # and z (subsample) according to the values in ks and zs. 
        # If required, edges are padded.
        temp_array = pulses.reshape(-1, pulses.shape[-1])
        temp_k = ks.flatten()
        temp_z = zs.flatten()

        new_array = np.zeros(temp_array.shape)

        for j in range(temp_array.shape[0]):
            k = temp_k[j]
            z = temp_z[j]
            if k == 0:
                new_array[j, :] = temp_array[j, :]
            elif k > 0:
                new_array[j, k:] = temp_array[j, :-k]
                new_array[j, :k] = np.mean(temp_array[j, :10])
            else:
                # Note that k is negative here
                new_array[j, :k] = temp_array[j, -k:]
                new_array[j, k:] = np.mean(temp_array[j, -10:])
            
            # Interpolate template if necessary (subsample shift)
            if z > 0:
                new_array[j] = np.hstack([
                    new_array[j][0], 
                    new_array[j][:-1] + (1 - z)*np.diff(new_array[j])
                    ])

        return np.reshape(new_array, pulses.shape)

    def _next_raw(self):
        if self._current_batch_ind < self.n_batches:
            event_inds_in_batch = self._inds[self._current_batch_ind]
            self._current_batch_ind += 1
            
            sim_phs = self._phs[..., event_inds_in_batch].T[..., None]
            shifts = self._shift_samples[..., event_inds_in_batch].T[..., None]
            subshifts = self._shift_subsamples[..., event_inds_in_batch].T[..., None]
            events = next(self._itit)

            if self._using_fit:
                pulse = self._shift_fit_pars_and_eval(
                    # Extend the fitpars array so that we can easier treat different cases
                    # (batches, no batches, single channel, multi channel, ...)
                    np.broadcast_to(
                        self._sev_or_pars, 
                        tuple(list(events.shape)[:-1] + [self._sev_or_pars.shape[-1]]) 
                    ),
                    shifts,
                    subshifts,
                )
            else:
                pulse = self._shift_array(
                    # Extend the sev array so that we can easier treat different cases
                    # (batches, no batches, single channel, multi channel, ...)
                    np.broadcast_to(self._sev_or_pars, events.shape),
                    shifts,
                    subshifts,
                )

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