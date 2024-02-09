from typing import List, Union, Callable, Type
from abc import ABC, abstractmethod
from contextlib import nullcontext
from multiprocessing import Pool
from inspect import signature
import itertools

import numpy as np
from tqdm.auto import tqdm
import h5py

#### HELPER FUNCTIONS AND CLASSES ####
def _ensure_array(x):
    if isinstance(x, str): x = [x]
    elif isinstance(x, int): x = [x]
    elif isinstance(x, np.integer): x = [int(x)]
    return np.array(x)
    
def _ensure_not_array(x):
    if isinstance(x, np.ndarray): x = x.tolist()
    if isinstance(x, np.integer): x = int(x)
    if isinstance(x, str): x = str(x)
    return x

class BatchResolver:
    """
    Helper Class to resolve batched iterators.
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, batch):
        return [self.f(ev) for ev in batch]

#### ABSTRACT BASE CLASS ####
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

    def __next__(self):
        return self._apply_processing(self._next_raw())
    
    @abstractmethod
    def _next_raw(self):
        # This is to be used as the __next__ method by child classes. 
        # The purpose of having it separately is that the IteratorBaseClass 
        # then automatically applies the preprocessing
        ...

    def _apply_processing(self, out):
        for fnc in self.fncs:
            out = fnc(out) if not self.uses_batches else BatchResolver(fnc)(out)

        return out

    def __repr__(self):
        out = f"{self.__class__.__name__}(n_events={len(self)}, n_channels={self.n_channels}, uses_batches={self.uses_batches}"
        if len(self.fncs) > 0:
            out += f", preprocessing: {self.fncs})"
        else:
            out += ")"
        return out
    
    def __getitem__(self, val):
        # Slice Iterator as if it was layed out as a numpy.ndarray. 
        # The first argument slices the channel/key/... and the second slices the remaining list of events in the iterator.

        if not isinstance(val, tuple): val = (val,)

        if len(val) > 2:
            raise IndexError(f"Too many indices for iterator: 2 values can be indexed (channel list and event list) but {len(val)} were indexed.")

        # First index will numpy-slice channel (or equivalent)
        slice_channel = val[0]
        # Second index will numpy-slice indices (or equivalent)
        slice_inds    = val[1] if len(val)>1 else slice(None)

        # Get parameters to reconstruct iterator and make a copy
        params, keys = self._slice_info
        new_params = dict(**params)
        # Slice relevant arguments
        new_params[keys[0]] = _ensure_not_array(_ensure_array(params[keys[0]])[slice_channel])
        new_params[keys[1]] = _ensure_not_array(_ensure_array(params[keys[1]])[slice_inds])

        # Create new instance of iterator and add processing
        new_iterator = self.__class__(**new_params)
        new_iterator.add_processing(self.fncs.copy())

        # Return new iterator
        return new_iterator
    
    def __add__(self, other):
        if isinstance(self, IteratorCollection):
            l = [i.with_processing(self.fncs) for i in self.iterators]
        else:
            l = [self]
        
        if isinstance(other, IteratorBaseClass):
            if isinstance(other, IteratorCollection):
                l += [i.with_processing(other.fncs) for i in other.iterators]
            else:
                l += [other]
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

        return IteratorCollection(l)

    def add_processing(self, f: Union[Callable, List[Callable]]):
        """
        Add functions to be applied to each event before returning it. Batches are supported, i.e. if the iterator returns events in batches, the specified functions are applied to all events in a batch separately. However, the user is responsible for handling multiple channels correctly: Events are passed to the functions directly, even if it includes multiple channels.

        :param f: Function(s) to be applied. Function signature: f(event: np.ndarray) -> np.ndarray
        :type f: Union[Callable, List[Callable]]

        >>> it = EventIterator("path_to_file.h5", "events", "event")
        >>> it.add_processing([f1, f2, f3])
        """
        if not isinstance(f, list): f = [f]

        self.fncs += f

        # return instance such that it is chainable and can be used in one-liners
        return self
    
    def with_processing(self, f: Union[Callable, List[Callable]]):
        """
        Same as `add_processing` but it returns a new iterator instead of modifying the original one.

        :param f: Function(s) to be applied. Function signature: f(event: np.ndarray) -> np.ndarray
        :type f: Union[Callable, List[Callable]]

        >>> it = EventIterator("path_to_file.h5", "events", "event")
        >>> new_it = it.with_processing([f1, f2, f3])
        """

        return self[:,:].add_processing(f)
        
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

    @property
    @abstractmethod
    def _slice_info(self):
        # Returns a tuple containing a dictionary of input arguments used to construct
        # the iterator and another tuple which specifies the dictionary keys corresponding
        # to the channel- and index-equivalents. Note that those cannot have None-values
        # in the dictionary! E.g. 
        # return ( {'path_h5': 'path/to/file', 
        #           'dataset': 'events', 
        #           'channels': 0, 
        #           'inds': [1,2,3,4], 
        #           'batch_size': 1
        #           }, 
        #          ('channel', 'inds')
        #        )
        ...

class H5Iterator(IteratorBaseClass):
    """
    Iterator object for HDF5 datasets that iterates along the "event-dimension" (second dimension of 3-dimensional events data) of a dataset and returns the event voltage traces.
    If the Iterator is used as a context manager, the HDF5 file is not closed during iteration which improves file access speed.

    The datasets in the HDF5 file are assumed to have shape `(channels, events, data)` but the iterator *always* returns data event by event. If batches are used (see below), they are returned with the events dimension being the first dimension. To explain the returned shapes we start from a general dataset with shape `(n_channels, n_events, n_data)`. Note that `n_channels`, `n_events`, or `n_data` could be 1, but in total, a 3-dimensional dataset is needed. 
    For a batch size of 1, the iterator in this case returns shapes `(n_channels, n_data)`. 
    For a batch size > 1, the iterator in this case returns shapes `(batch_size, n_channels, n_data)`. Notice that the first dimension always has the events (batch_size).

    :param path_h5: Path to the HDF5 file.
    :type path_h5: str
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

    >>> it = H5Iterator("path_to_file.h5", "events", "event", batch_size=100, channels=1, inds=[0,2,19,232])
    >>> for i in it:
    ...    print(i.shape)
    
    >>> with it as opened_it:
    ...     for i in opened_it:
    ...         print(i.shape)
    """

    def __init__(self, path_h5: str, group: str, dataset: str, channels: Union[int, List[int]] = None, inds: List[int] = None, batch_size: int = None):
        super().__init__()

        # Check if dataset has correct shape:
        with h5py.File(path_h5, 'r') as f:
            ndim = f[group][dataset].ndim
            shape = f[group][dataset].shape
            if ndim != 3:
                raise ValueError(f"Only 3-dimensional datasets can be used to construct H5Iterator. Dataset '{dataset}' in group '{group}' is {ndim}-dimensional.")

        self._path = path_h5
        self._group = group
        self._dataset = dataset

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
        self._n_events = len(inds)

        # self._inds will be a list of batches. If we just take the inds list, we have batches of size 1, if we take [inds]
        # all inds are in one batch, otherwise it is a list of lists where each list is a batch
        if batch_size is None or batch_size == 1:
            self._inds = inds
            self._uses_batches = False
        elif batch_size == -1:
            self._inds = [inds]
            self._uses_batches = True
        else: 
            self._inds = [inds[i:i+batch_size] for i in range(0, len(inds), batch_size)]
            self._uses_batches = True

        self._n_batches = len(self._inds)

        # Save values to reconstruct iterator:
        self._params = {'path_h5': path_h5, 'group': group, 
                        'dataset': dataset, 'channels': self._channels, 
                        'inds': inds, 'batch_size': batch_size}

        # If list specifies all channels, we replace it by a None-slice to bypass h5py's restriction on fancy indexing
        # Notice that this has to be done after saving the list in self._params
        if self._channels == list(range(shape[0])): self._channels = slice(None)

        # For multiple channels and batched data, we transpose the array as read from the HDF5 file to ensure
        # that the first dimension of the output is always the event dimension
        self._should_be_transposed = self._uses_batches and self._n_channels > 1

        self._file_open = False
    
    def __len__(self):
        return self._n_events

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
        if self._current_batch_ind < self._n_batches:
            event_inds_in_batch = self._inds[self._current_batch_ind]
            self._current_batch_ind += 1

            # Open HDF5 file if not yet open, else use already open file
            with h5py.File(self._path, 'r') if not self._file_open else nullcontext(self._f) as f:
                out = f[self._group][self._dataset][self._channels, event_inds_in_batch]

                # transpose data when using batches such that first dimension is ALWAYS the event dimension
                if self._should_be_transposed: out = np.transpose(out, axes=[1,0,2])
                    
                return out
        
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
    
    @property
    def _slice_info(self):
        return (self._params, ('channels', 'inds'))
        
class StreamIterator(IteratorBaseClass):
    """
    Iterator object that returns voltage traces for given trigger indices of a stream file. 

    :param stream: The stream object to read the voltage traces from.
    :type stream: StreamBaseClass
    :param keys: The keys (channel names) of the stream object to be iterated over. 
    :type keys: Union[str, List[str]]
    :param inds: The stream indices for which we want to read the voltage traces. This index is aligned according to 'alignment' (default: at 1/4th of the record window).
    :type inds: Union[int, List[int]]
    :param record_length: The number of samples to be returned for each index. Usually, those are powers of 2, e.g. 16384
    :type record_length: int
    :param alignment: A number in the interval [0,1] which determines the alignment of the record window (of length `record_length`) relative to the specified index. E.g. if `alignment=1/2`, the record window is centered around the index. Defaults to 1/4.
    :type alignment: float


    :return: Iterable object
    :rtype: StreamIterator
    """

    def __init__(self, stream, keys: Union[str, List[str]], inds: Union[int, List[int]], record_length: int, alignment: float = 1/4):
        if 0 > alignment or 1 < alignment:
            raise ValueError("'alignment' has to be in the interval [0,1]")
        
        super().__init__()
        
        self._stream = stream
        self._keys = [keys] if isinstance(keys, str) else keys
        self._inds = [inds] if isinstance(inds, int) else [int(i) for i in inds]
        self._record_length = record_length

        # Save values to reconstruct iterator:
        self._params = {'stream': stream, 'keys': self._keys, 
                        'inds': self._inds, 'record_length': record_length,
                        'alignment': alignment}

        self._interval = (int(alignment*record_length), record_length - int(alignment*record_length))
    
    def __len__(self):
        return len(self._inds)

    def __enter__(self):
        return self # Just to be consistent with EventIterator
    
    def __exit__(self, typ, val, tb):
        ... # Just to be consistent with EventIterator
    
    def __iter__(self):
        self._current_ind = 0
        return self

    def _next_raw(self):
        if self._current_ind < len(self._inds):
            stream_ind = self._inds[self._current_ind]
            s = slice(stream_ind - self._interval[0], stream_ind + self._interval[1])

            self._current_ind += 1

            if len(self._keys) == 1:
                out = self._stream[self._keys[0], s, 'as_voltage']
            else:
                out = [self._stream[k, s, 'as_voltage'] for k in self._keys]
            
            return np.array(out)
            
        else:
            raise StopIteration
        
    @property
    def uses_batches(self):
        return False # Just to be consistent with EventIterator
    
    @property
    def n_batches(self):
        return len(self)
    
    @property
    def n_channels(self):
        return len(self._keys)
    
    @property
    def _slice_info(self):
        return (self._params, ('keys', 'inds'))

class IteratorCollection(IteratorBaseClass):
    """
    Iterator object that chains multiple iterators.

    :param iterators: Iterator or List of Iterators to chain.
    :type iterators: Union[IteratorBaseClass, List[IteratorBaseClass]]

    :return: Iterable object
    :rtype: IteratorCollection

    >>> it = H5Iterator("path_to_file.h5", "events", "event")
    >>> it_collection = IteratorCollection([it, it])
    >>> # Or simply (output of iterator addition is IteratorCollection)
    >>> it_collection = it + it
    """
    def __init__(self, iterators: Union[IteratorBaseClass, List[IteratorBaseClass]]):
        super().__init__()
        # Check if all elements are IteratorBaseClass instances
        if isinstance(iterators, list):
            for it in iterators:
                if not isinstance(it, IteratorBaseClass):
                    raise TypeError(f"All iterators must be child classes of 'IteratorBaseClass'. Not '{type(it)}'.")
        else:
            if isinstance(iterators, IteratorBaseClass):
                iterators = [iterators]
            else:
                raise TypeError(f"Unsupported type '{type(iterators)}' for input argument 'iterators'.")
            
        # Check if batch usage and number of channels are consistent
        batch_usage = [it.uses_batches for it in iterators]
        channel_usage = [it.n_channels for it in iterators]
        if len(set(batch_usage)) != 1:
            raise ValueError(f"Either all iterators must use batches or none of them. Got {batch_usage}")
        if len(set(channel_usage)) != 1:
            raise ValueError(f"All iterators must contain the same number of channels. Got {channel_usage}")
        
        self._iterators = iterators
        self._uses_batches = batch_usage[0] # made sure that batch usage is consistent above
        self._n_channels = channel_usage[0] # made sure that number of channels is consistent above

    def __len__(self):
        return sum([len(it) for it in self._iterators])
    
    def __repr__(self):
        out = f"{self.__class__.__name__}(n_events={len(self)}, n_channels={self.n_channels}, uses_batches={self.uses_batches})["
        
        for it in self._iterators:
            out += f"\n\t- {it}"
        out += "\n\t]"

        if len(self.fncs) > 0:
            out += f"(preprocessing: {self.fncs})"

        return out

    def __enter__(self):
        for it in self._iterators: it.__enter__()
        return self
    
    def __exit__(self, typ, val, tb):
        for it in self._iterators: it.__exit__(typ, val, tb)
    
    def __iter__(self):
        self._chain = itertools.chain.from_iterable(self._iterators)
        return self

    def _next_raw(self):
        return next(self._chain)
    
    def __getitem__(self, val):
        # overriding IteratorBaseClass behavior
        if not isinstance(val, tuple): val = (val,)

        if len(val) > 2:
            raise IndexError(f"Too many indices for iterator: 2 values can be indexed (channel list and event list) but {len(val)} were indexed.")

        slice_channel = val[0]
        slice_inds    = val[1] if len(val)>1 else slice(None)

        channels = np.arange(self._n_channels)                  # Make array of all indices
        channels_sliced = channels[slice_channel]               # Indices that survive the slice
        channels_bool = np.zeros(self._n_channels, dtype=bool)  # Turn surviving indices into
        channels_bool[channels_sliced] = True                   # boolean array

        inds = np.arange(len(self))                  # Make array of all channels
        inds_sliced = inds[slice_inds]               # Channels that survive the slice
        inds_bool = np.zeros(len(self), dtype=bool)  # Turn surviving channels into
        inds_bool[inds_sliced] = True                # boolean array

        # Boolean array for channels is identical for all iterators in collection
        # But we have to split the boolean array of surviving events and forward the pieces
        # to iterators in collection
        lens = np.array([len(it) for it in self._iterators])
        sub_arrays = np.split(inds_bool, np.cumsum(lens)[:-1])

        # Slice iterators in collection
        new_iterators = [it[channels_bool, a] for it, a in zip(self._iterators, sub_arrays)]

        # Create new collection and add processing
        new_collection = IteratorCollection(new_iterators)
        new_collection.add_processing(self.fncs.copy())

        return new_collection

    @property
    def uses_batches(self):
        return self._uses_batches
    
    @property
    def n_batches(self):
        return sum([it.n_batches for it in self._iterators])

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def _slice_info(self):
        ... # Not needed here because __getitem__ was overridden
    
    @property
    def iterators(self):
        return self._iterators

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

# TODO: implement
class PulseSimIterator(IteratorBaseClass):
    ...

def apply(f: Callable, ev_iter: Type[IteratorBaseClass], n_processes: int = 1):
    """
    Apply a function to events provided by an EventIterator. 

    Multiprocessing and resolving batches as returned by the iterator is done automatically. The function returns a numpy array where the first dimension corresponds to the events returned by the iterator. Higher dimensions are as returned by the function that is applied. Batches are resolved, i.e. calls with an `EventIterator(..., batch_size=1)` and `EventIterator(..., batch_size=100)` yield identical results. 

    *Important*: Since `apply` uses multiprocessing, it is best not to use functions that are defined locally within jupyter lab, but rather to define them in a separate `.py` file and load them from the notebook. This is only relevant if you are trying to define your own function and not if you are just using already existing `cait` functions.

    :param f: Function to be applied to events. Note the restriction above.
    :type f: Callable
    :param ev_iter: Events for which the function should be applied.
    :type ev_iter: `~class:cait.versatile.file.EventIterator`
    :param n_processes: Number of processes to use for multiprocessing.
    :type n_processes: int

    :return: Results of `f` for all events in `ev_iter`. Has same structure as output of `f` (just with an additional event dimension).
    :rtype: Any

    >>> # Example when func has one output
    >>> it = dh.get_event_iterator("events", batch_size=42)
    >>> out = apply(func, it)
    >>> # Example when func has two outputs
    >>> it = dh.get_event_iterator("events", batch_size=42)
    >>> out1, out2 = apply(func, it)
    """
    # Check if 'ev_iter' is a cait.versatile iterator object
    if not isinstance(ev_iter, IteratorBaseClass):
        raise TypeError(f"Input argument 'ev_iter' must be an instance of {IteratorBaseClass} not '{type(ev_iter)}'.")
    
    # Check if 'f' is indeed a function
    if not callable(f):
        raise TypeError(f"Input argument 'f' must be callable.")
    
    # Check if 'f' takes exactly one argument (the event)
    if len(signature(f).parameters) != 1:
        raise TypeError(f"Input function {f} has too many arguments ({len(signature(f).parameters)}). Only functions which take one argument (the event) are supported.")
    
    if ev_iter.uses_batches: f = BatchResolver(f)

    with ev_iter as ev_it:
        if n_processes > 1:
            with Pool(n_processes) as pool:
                out = list(tqdm(pool.imap(f, ev_it), total=ev_iter.n_batches))
        else:
            out = [f(ev) for ev in tqdm(ev_it, total=ev_iter.n_batches)]
    
    # Chain batches such that the list is indistinguishable from a list using no batches
    # (If uses_batches, 'out' is a list of lists)
    if ev_iter.uses_batches: out = list(itertools.chain.from_iterable(out))

    # If elements in 'out' are tuples, this means that the function had multiple outputs.
    # In this case, we transpose the list so that we have a tuple of outputs where each element in the tuple is also converted to a numpy.array of len(ev_iter)
    if isinstance(out[0], tuple): 
        out = tuple(np.array(x) for x in zip(*out))
    else:
        out = np.array(out)

    return out