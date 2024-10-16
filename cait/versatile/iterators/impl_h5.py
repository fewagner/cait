from typing import Union, List
from contextlib import nullcontext

import numpy as np
import h5py

from .iteratorbase import IteratorBaseClass

class H5Iterator(IteratorBaseClass):
    """
    Iterator object for HDF5 datasets that iterates along the "event-dimension" (second dimension of 3-dimensional events data) of a dataset and returns the event voltage traces.
    If the Iterator is used as a context manager, the HDF5 file is not closed during iteration which improves file access speed.

    The datasets in the HDF5 file are assumed to have shape `(channels, events, data)` but the iterator *always* returns data event by event. If batches are used (see below), they are returned with the events dimension being the first dimension. To explain the returned shapes we start from a general dataset with shape `(n_channels, n_events, n_data)`. Note that `n_channels`, `n_events`, or `n_data` could be 1, but in total, a 3-dimensional dataset is needed. 
    For a batch size of 1, the iterator in this case returns shapes `(n_channels, n_data)`. 
    For a batch size > 1, the iterator in this case returns shapes `(batch_size, n_channels, n_data)`. Notice that the first dimension always has the events (batch_size).

    :param dh: DataHandler instance connected to the HDF5 file.
    :type dh: DataHandler
    :param group: Group in the HDF5 file to read events from.
    :type group: str
    :param dataset: Dataset in the HDF5 file to read events from.
    :type dataset: str
    :param inds: List of event indices to iterate. If left None, the H5Iterator will iterate over all events.
    :type inds: List[int]
    :param channels: Integer or list of integers specifying the channel(s) to iterate. If left None, the H5Iterator will iterate over all channels.
    :type channels: Union[int, List[int]]
    :param batch_size: The number of events to be returned at once (these are all read together). There will be a trade-off: large batch_sizes cause faster read speed but increase the memory usage.
    :type batch_size: int

    :return: Iterable object
    :rtype: EventIterator

    .. code-block:: python

        it = H5Iterator(dh, "events", batch_size=100, channels=1, inds=[0,2,19,232])
        for i in it:
            print(i.shape)
    
        with it as opened_it:
            for i in opened_it:
                print(i.shape)
    """

    def __init__(self, dh, group: str, channels: Union[int, List[int]] = None, inds: List[int] = None, batch_size: int = None):

        # Check if dataset has correct shape:
        with h5py.File(dh.get_filepath(), 'r') as f:
            ndim = f[group]['event'].ndim
            shape = f[group]['event'].shape
            if ndim != 3:
                raise ValueError(f"Only 3-dimensional datasets can be used to construct H5Iterator. Dataset 'event' in group '{group}' is {ndim}-dimensional.")

        self._dh = dh
        self._path = dh.get_filepath()
        self._group = group

        n_events_total = shape[1]

        if channels is None: channels = list(range(shape[0])) 

        if isinstance(channels, int):
            self._channels = channels
            self._n_channels = 1
        elif isinstance(channels, list):
            self._channels = channels if len(channels)>1 else channels[0]
            self._n_channels = len(channels)
        else:
            raise TypeError(f"Unsupported type {type(channels)} for input argument 'channels'")
            
        if inds is None: inds = np.arange(n_events_total)
        inds = [inds] if isinstance(inds, int) else [int(i) for i in inds]

        # Does batch handling and creates properties self._inds, self.uses_batches, and self.n_batches
        super().__init__(inds=inds, batch_size=batch_size)

        # Save values to reconstruct iterator:
        self._params = {'dh': dh, 
                        'group': group, 
                        'channels': self._channels, 
                        'inds': inds, 
                        'batch_size': batch_size}

        # If list specifies all channels, we replace it by a None-slice to bypass h5py's restriction on fancy indexing
        # Notice that this has to be done after saving the list in self._params
        if self._channels == list(range(shape[0])): self._channels = slice(None)

        # For multiple channels and batched data, we transpose the array as read from the HDF5 file to ensure
        # that the first dimension of the output is always the event dimension
        self._should_be_transposed = self.uses_batches and self._n_channels > 1

        self._file_open = False

    def __enter__(self):
        self._f = h5py.File(self._path, 'r')
        self._file_open = True
        return self
    
    def __exit__(self, typ, val, tb):
        self._f.close()
        self._file_open = False
    
    def __iter__(self):
        self._current_batch_ind = 0
        return self

    def _next_raw(self):
        if self._current_batch_ind < self.n_batches:
            event_inds_in_batch = self._inds[self._current_batch_ind]
            self._current_batch_ind += 1

            # Open HDF5 file if not yet open, else use already open file
            with h5py.File(self._path, 'r') if not self._file_open else nullcontext(self._f) as f:
                out = f[self._group]["event"][self._channels, event_inds_in_batch]

                # transpose data when using batches such that first dimension is ALWAYS the event dimension
                if self._should_be_transposed: out = np.transpose(out, axes=[1,0,2])
                    
                return out
        
        else:
            raise StopIteration
        
    @property
    def record_length(self):
        return self._dh.record_length
    
    @property
    def dt_us(self):
        return self._dh.dt_us
    
    @property
    def ds_start_us(self):
        # calculate start from existing hours relative to timestamps
        with h5py.File(self._path, 'r') as f:
            sec = np.array(f[self._group]["time_s"][0], dtype=np.int64)
            mus = np.array(f[self._group]["time_mus"][0], dtype=np.int64)
            hours = np.array(f[self._group]["hours"][0], dtype=np.float32)

            return sec*int(1e6) + mus - int(1e6*3600*hours)

    @property
    def timestamps(self):
        with h5py.File(self._path, 'r') as f:
            sec = np.array(f[self._group]["time_s"][self._params["inds"]], dtype=np.int64)
            mus = np.array(f[self._group]["time_mus"][self._params["inds"]], dtype=np.int64)

        return sec*int(1e6) + mus
    
    @property
    def n_channels(self):
        return self._n_channels
    
    @property
    def _slice_info(self):
        return (self._params, ('channels', 'inds'))