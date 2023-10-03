import os
from typing import List, Union, Callable
from abc import ABC, abstractmethod
from contextlib import nullcontext

import numpy as np
import h5py

from .file import get_dataset_properties

class BatchResolver:
    def __init__(self, f):
        self.f = f

    def __call__(self, batch):
        return [self.f(ev) for ev in batch]

class IteratorBaseClass(ABC):
    def __init__(self):
        self.fncs = list()

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __enter__(self):
        ...
    
    @abstractmethod
    def __exit__(self, typ, val, tb):
        ...
    
    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __next__(self):
        ...

    def add_processing(self, *args: Callable):
        """
        Add functions to be applied to each event before returning it. Batches are supported, i.e. if the iterator returns events in batches, the specified functions are applied to all events in a batch separately. However, the user is responsible for handling multiple channels correctly: Events are passed to the functions directly, even if it includes multiple channels.

        :param args: Function(s) to be applied. Function signature: f(event: np.ndarray) -> np.ndarray
        :type args: Callable

        >>> it = EventIterator("path_to_file.h5", "events", "event")
        >>> it.add_processing(f1, f2, f3)
        """
        if self.uses_batches:
            self.fncs += [BatchResolver(f) for f in args]
        else:
            self.fncs += args

        # return instance such that it is chainable and can be used in one-liners
        return self

    def _apply_processing(self, out):
        for fnc in self.fncs:
            out = fnc(out)
        
        return out
        
    @property
    @abstractmethod
    def uses_batches(self):
        ...
    
    @property
    @abstractmethod
    def n_batches(self):
        ...

    @property
    @abstractmethod
    def n_channels(self):
        ...

class EventIterator(IteratorBaseClass):
    """
    Iterator object for HDF5 datasets that iterates along the "event-dimension" (first dimension for 1-dimensional data, or second dimension for 2- and 3-dimensional data) of a dataset. Most important use-case is iterating over event voltage traces in an analysis routine.
    If the Iterator is used as a context manager, the HDF5 file is not closed during iteration which improves file access speed.

    The datasets in the HDF5 file are assumed to have shape `(events, data)` or `(channels, events, data)` but the iterator *always* returns data event by event. If batches are used (see below), they are returned with the events dimension being the first dimension. To explain the returned shapes we start from a general dataset with shape `(n_channels, n_events, n_data)`. Note that `n_channels` and/or `n_data` could be 1 such that the shapes reduce to `(n_events, n_data)` or `(n_channels, n_events)` respectively. 
    For a batch size of 1, the iterator in these cases returns shapes `(n_channels, n_data)`, `(n_data)` and `(n_channels)`. 
    For a batch size > 1, the iterator in these cases returns shapes `(batch_size, n_channels, n_data)`, `(batch_size, n_data)` and `(batch_size, n_channels)`. Notice that the first dimension always has the events (batch_size).

    :param path_h5: Path to the HDF5 file.
    :type path_h5: str
    :param group: Group in the HDF5 file.
    :type group: str
    :param dataset: Dataset in the HDF5 file.
    :type dataset: str
    :param inds: List of event indices to iterate. If left None, the EventIterator will iterate over all events.
    :type inds: List[int]
    :param channels: Integer or list of integers specifying the channel(s) to iterate. If left None, the EventIterator will iterate over all channels.
    :type channels: Union[int, List[int]]
    :param batch_size: The number of events to be returned at once (these are all read together). There will be a trade-off: large batch_sizes cause faster read speed but increase the memory usage.
    :type batch_size: int

    :return: Iterable object
    :rtype: EventIterator

    >>> it = EventIterator("path_to_file.h5", "events", "event", batch_size=100, channels=1, inds=[0,2,19,232])
    >>> for i in it:
    ...    print(i.shape)
    
    >>> with it as opened_it:
    ...     for i in opened_it:
    ...         print(i.shape)
    """

    def __init__(self, path_h5: str, group: str, dataset: str, inds: List[int] = None, channels: Union[int, List[int]] = None, batch_size: int = None):
        super().__init__()

        self.path = path_h5
        self.group = group
        self.dataset = dataset
        
        # total number of events, shape of dataset, and axis along which events extend
        # function returns a list of n_events (which in this case is of length 1)
        n_events_total, shape, _, self.events_dim = get_dataset_properties([os.path.splitext(path_h5)[0]], group, dataset)
        n_events_total = n_events_total[0]

        if channels is not None:
            assert len(shape)>0, "The dataset is one-dimensional. Specifying channels is not supported here."
            self.channels = channels
            self._n_channels = 1 if isinstance(channels, int) else len(channels)

        elif len(shape)>0:
            self.channels = slice(None) 
            self._n_channels = shape[0]
        else:
            self.channels = None
            self._n_channels = 1
            
        if inds is None: inds = np.arange(n_events_total)
        self.n_events = len(inds)

        # self.inds will be a list of batches. If we just take the inds list, we have batches of size 1, if we take [inds]
        # all inds are in one batch, otherwise it is a list of lists where each list is a batch
        if batch_size is None or batch_size == 1:
            self.inds = inds
            self._uses_batches = False
        elif batch_size == -1:
            self.inds = [inds]
            self._uses_batches = True
        else: 
            self.inds = [inds[i:i+batch_size] for i in range(0, len(inds), batch_size)]
            self._uses_batches = True

        self._n_batches = len(self.inds)

        self.file_open = False
    
    def __len__(self):
        return self.n_events

    def __enter__(self):
        self.f = h5py.File(self.path, 'r')
        self.file_open = True
        return self
    
    def __exit__(self, typ, val, tb):
        self.f.close()
        self.file_open = False
    
    def __iter__(self):
        self.current_batch_ind = 0
        return self

    def __next__(self):
        if self.current_batch_ind < self._n_batches:
            event_inds_in_batch = self.inds[self.current_batch_ind]
            self.current_batch_ind += 1

            # For single channel data, the first dimension is always the event dimension (possibly 0-dimensional if 
            # batch size is 1). For multi-channel data and batch size = 1, the first dimension is (implicitly) the
            # event dimension (because it is 0-dimensional). Only for multi-channel data and batch size > 1 we have 
            # the event dimension in second place (and therefore have to transpose)
            should_be_transposed = self._uses_batches and self._n_channels > 1 and self.events_dim > 0 

            # Open HDF5 file if not yet open, else use already open file
            with h5py.File(self.path, 'r') if not self.file_open else nullcontext(self.f) as f:
                if self.events_dim > 0:
                    if should_be_transposed:
                        # transpose data such that first dimension is ALWAYS the event dimension
                        return np.transpose(f[self.group][self.dataset][self.channels,  
                        event_inds_in_batch], axes=[1,0,2])
                    else:
                        return self._apply_processing(
                                f[self.group][self.dataset][self.channels, event_inds_in_batch]
                        )
                else:
                    return self._apply_processing(
                            f[self.group][self.dataset][event_inds_in_batch]
                    )
        else:
            raise StopIteration
        
    @property
    def uses_batches(self):
        return self._uses_batches
    
    @property
    def n_batches(self):
        return self._n_batches
    
    @property
    def n_channels(self):
        return self._n_channels
        
# TODO: add test case
class StreamIterator(IteratorBaseClass):
    """
    Iterator object that returns voltage traces for given trigger indices of a stream file. 

    :param stream: The stream object to read the voltage traces from.
    :type stream: StreamBaseClass
    :param keys: The keys (channel names) of the stream object to be iterated over. 
    :type keys: Union[str, List[str]]
    :param inds: The stream indices for which we want to read the voltage traces. How this index is aligned in the returned record window is dictated by the `alignment` argument.
    :type inds: Union[int, List[int]]
    :param record_length: The number of samples to be returned for each index. Usually, those are powers of 2, e.g. 16384
    :type record_length: int
    :param alignment: A number in the interval [0,1] which determines the alignment of the record window (of length `record_length`) relative to the specified index. E.g. if `alignment=1/2`, the record window is centered around the index. Defaults to 1/4.
    :type alignment: float

    :return: Iterable object
    :rtype: StreamIterator
    """

    def __init__(self, stream, keys: Union[str, List[str]], inds: Union[int, List[int]], record_length: int, alignment: float = 1/4):
        super().__init__()

        if 0 > alignment or 1 < alignment:
            raise ValueError("'alignment' has to be in the interval [0,1]")
        
        self._stream = stream
        self._keys = [keys] if type(keys) is str else keys
        self._inds = [inds] if type(inds) is int else [int(i) for i in inds]
        self._record_length = record_length

        self._interval = (int(alignment*record_length), record_length - int(alignment*record_length))
    
    def __len__(self):
        return len(self._inds)

    def __enter__(self):
        # Just to be consistent with EventIterator
        return self
    
    def __exit__(self, typ, val, tb):
        # Just to be consistent with EventIterator
        ...
    
    def __iter__(self):
        self._current_ind = 0
        return self

    def __next__(self):
        if self._current_ind < len(self._inds):
            stream_ind = self._inds[self._current_ind]
            s = slice(stream_ind - self._interval[0], stream_ind + self._interval[1])

            self._current_ind += 1

            return self._apply_processing(
                np.array([self._stream[k, s, 'as_voltage'] for k in self._keys])
            )
            
        else:
            raise StopIteration
        
    @property
    def uses_batches(self):
        # Just to be consistent with EventIterator
        return False
    
    @property
    def n_batches(self):
        return len(self)
    
    @property
    def n_channels(self):
        return len(self._keys)
            