from abc import ABC, abstractmethod
from typing import Union, List, Callable
import itertools

import numpy as np

from .batchresolver import BatchResolver

#### HELPER FUNCTIONS ####
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

class IteratorBaseClass(ABC):
    def __init__(self, inds: List[int], batch_size: int = None):
        self.fncs = list()

        self.__n_events = len(inds)

        # self._inds will be a list of batches. If we just take the inds list, we have batches of size 1, if we take [inds]
        # all inds are in one batch, otherwise it is a list of lists where each list is a batch
        if batch_size is None or batch_size == 1:
            self.__inds = inds
            self._uses_batches = False
        elif batch_size == -1:
            self.__inds = [inds]
            self._uses_batches = True
        else: 
            self.__inds = [inds[i:i+batch_size] for i in range(0, len(inds), batch_size)]
            self._uses_batches = True

        self._n_batches = len(self._inds)

    def __len__(self):
        return self.__n_events

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
            out = fnc(out) if not self.uses_batches else BatchResolver(fnc, self.n_channels)(out)

        return out

    def __repr__(self):
        out = f"{self.__class__.__name__}(n_events={len(self)}, n_channels={self.n_channels}, uses_batches={self.uses_batches}, record_length={self.record_length}, dt_us={self.dt_us}"
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
    
    def grab(self, which: Union[int, list]):
        """
        Grab specified event(s) and return it/them as numpy array.

        :param which: Events of interest.
        :type which: Union[int, list]

        **Example:**
        ::
            import cait.versatile as vai

            # Get events from mock data
            it = vai.MockData().get_event_iterator()

            # Get the last event in the iterator
            selected_event = it.grab(-1)

            # Get events with indices 1, 7, 9
            selected_events = it.grab([1,7,9])
        """

        return np.squeeze(np.array(list(self[:, which])))[()]

    @property
    def t(self):
        """
        Return the time axis (record window) of the events in the iterator. It is a millisecond array with 0 being at 1/4th of the window.
        """
        return (np.arange(self.record_length) - self.record_length/4)*self.dt_us/1000

    @property
    def _inds(self):
        return self.__inds
    
    @property
    def uses_batches(self):
        """
        Returns True if the iterator returns batches.
        """
        return self._uses_batches
    
    @property
    def n_batches(self):
        """
        Returns the number of batches in the iterator.
        """
        return self._n_batches
    
    @property
    @abstractmethod
    def record_length(self):
        """
        Returns the record length (in samples) of the events in the iterator.
        """
        ...
    
    @property
    @abstractmethod
    def dt_us(self):
        """
        Returns the time base (in microseconds) of the events in the iterator.
        """
        ...

    @property
    @abstractmethod
    def ds_start_us(self):
        """
        The microsecond timestamp of the start of the recording for the datasource underlying this iterator object.
        """
        ...

    @property
    @abstractmethod
    def timestamps(self):
        """
        Returns microsecond timestamps corresponding to the trigger times of the events in the iterator.
        """
        ...

    @property
    @abstractmethod
    def n_channels(self):
        """
        Returns the number of channels in the iterator.
        """
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

class IteratorCollection(IteratorBaseClass):
    """
    Iterator object that chains multiple iterators.

    :param iterators: Iterator or List of Iterators to chain.
    :type iterators: Union[IteratorBaseClass, List[IteratorBaseClass]]

    :return: Iterable object
    :rtype: IteratorCollection

    >>> it = H5Iterator(dh, "events", "event")
    >>> it_collection = IteratorCollection([it, it])
    >>> # Or simply (output of iterator addition is IteratorCollection)
    >>> it_collection = it + it
    """
    def __init__(self, iterators: Union[IteratorBaseClass, List[IteratorBaseClass]]):
        # We do not construct the superclass because batching is handled differently
        self.fncs = list()
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
            
        # Check if batch usage, number of channels, record_length and dt_us are consistent
        batch_usage = [it.uses_batches for it in iterators]
        channel_usage = [it.n_channels for it in iterators]
        rec_usage = [it.record_length for it in iterators]
        dt_usage = [it.dt_us for it in iterators]
        if len(set(batch_usage)) != 1:
            raise ValueError(f"Either all iterators must use batches or none of them. Got {batch_usage}")
        if len(set(channel_usage)) != 1:
            raise ValueError(f"All iterators must contain the same number of channels. Got {channel_usage}")
        if len(set(rec_usage)) != 1:
            raise ValueError(f"All iterators must have the same record length. Got {rec_usage}")
        if len(set(dt_usage)) != 1:
            raise ValueError(f"All iterators must have the same time base. Got {dt_usage}")
        
        self._iterators = iterators
        self._uses_batches = batch_usage[0] # made sure that batch usage is consistent above
        self._n_channels = channel_usage[0] # made sure that number of channels is consistent above
        self._dt_us = dt_usage[0] # made sure that time base is consistent above
        self._record_length = rec_usage[0] # made sure that record length is consistent above

    # Overrides superclass
    def __len__(self):
        return sum([len(it) for it in self._iterators])
    
    def __repr__(self):
        out = f"{self.__class__.__name__}(n_events={len(self)})["
        
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
    def record_length(self):
        return self._record_length
    
    @property
    def dt_us(self):
        return self._dt_us
    
    @property
    def ds_start_us(self):
        return np.min([it.ds_start_us for it in self._iterators])
    
    @property
    def timestamps(self):
        return np.concatenate([it.timestamps for it in self._iterators])

    # Overrides superclass
    @property
    def uses_batches(self):
        return self._uses_batches
    
    # Overrides superclass
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