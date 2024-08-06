from abc import ABC, abstractmethod
from typing import Union, List, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ..datasourcebase import DataSourceBaseClass
from ...iterators.impl_stream import StreamIterator

class StreamBaseClass(DataSourceBaseClass):
    def __enter__(self):
        return self
    
    def __exit__(self, typ, val, tb):
        ...
        
    @abstractmethod
    def __len__(self):
        ...
        
    @abstractmethod
    def get_voltage_trace(self, key: str, where: slice):
        """
        Get the voltage trace for a given channel 'key' and slice 'where'.
        
        :return: Voltage trace.
        :rtype: np.ndarray
        """
        ...

    @property
    @abstractmethod
    def keys(self):
        """
        Available keys (channel names) in the stream.
        
        :return: List of keys.
        :rtype: list
        """
        ...

    @property
    @abstractmethod
    def start_us(self):
        """
        The microsecond timestamp at which the stream starts.
        
        :return: Microsecond timestamp
        :rtype: int
        """
        ...

    @property
    @abstractmethod
    def dt_us(self):
        """
        The length of a sample in the stream in microseconds.
        
        :return: Microsecond time-delta
        :rtype: int
        """
        ...

    @property
    @abstractmethod
    def tpas(self):
        """
        Dictionary of testpulse amplitudes in the stream. For hardware 'csmpl' this is read from a '.test_stamps' file. For hardware 'vdaq2' this is obtained from triggering the DAC channels first.
        
        :return: Testpulse amplitudes
        :rtype: dict of `np.ndarray`
        """
        ...

    @property
    @abstractmethod
    def tp_timestamps(self):
        """
        Dictionary of testpulse timestamps (microseconds) in the stream. For hardware 'csmpl' this is read from a '.test_stamps' file. For hardware 'vdaq2' this is obtained from triggering the DAC channels first.
        
        :return: Testpulse microsecond timestamps.
        :rtype: dict of `np.ndarray`
        """
        ...

    def __repr__(self):
        return f'{self.__class__.__name__}(start_us={self.start_us}, dt_us={self.dt_us}, length={self.__len__()}, keys={self.keys}, measuring_time_h={self.__len__()*self.dt_us/1e6/3600:.2f})'

    def __getitem__(self, val: Union[str, Tuple[str, Union[int, slice, list, np.ndarray]], Tuple[str, Union[int, slice, list, np.ndarray], str]]):
        # Only names and tuples are supported for slicing (no int)
        if type(val) not in [str, tuple]:
            raise TypeError(f'Unsupported type {type(val)} for slicing.')
            
        # If only a name is given, the channel with the respective name is returned
        if type(val) is str:
            return self.get_channel(val)
        
        # For tuples of length 2 or 3 there are specific behaviors
        else:
            if len(val) in [0, 1]:
                raise TypeError('No slicing support for tuples of length 0 or 1.')
                
            # Return the integer values for the stream 'name' if everything else is fine
            elif len(val) == 2:
                if type(val[0]) is str and type(val[1]) in [int, slice, list, np.ndarray]:
                    return self.get_channel(val[0])[val[1]]
                else:
                    raise TypeError('When slicing with two arguments, the first and second one have to be of type string and int/slice, respectively.')
            # Return the voltage values for the stream 'name' if everything else is fine
            elif len(val) == 3:
                if type(val[0]) is str and type(val[1]) in [int, slice, list, np.ndarray] and type(val[2]) is str:
                    if val[0] not in self.keys:
                        raise KeyError(f'{val[0]} not in stream. Available names: {self.keys}')
                    if val[2] != 'as_voltage':
                        raise ValueError(f'Unrecognized string "{val[2]}". Did you mean "as_voltage"?')
                    
                    if type(val[1]) is int:
                        # special case of indexing just with [-1]
                        if val[1] == -1: where = slice(val[1], None)
                        else: where = slice(val[1], val[1]+1)
                    else:
                        where = val[1]
                    return self.get_voltage_trace(key=val[0], where=where)
                else:
                    raise TypeError('When slicing with three arguments, the first, second and third one have to be of type string, int/slice and string, respectively.')
            else:  
                raise NotImplementedError(f'Tuples of length {len(val)} are not supported for slicing')
    
    @property
    def time(self):
        """
        Instance of `StreamTime`, which can be sliced to convert stream indices into microsecond timestamps and implements utility functions for the conversion to datetime for example.
        
        :return: StreamTime instance
        :rtype: `StreamTime`
        """
        if not hasattr(self, "_t"):
            self._t = StreamTime(self.start_us, self.dt_us, len(self))
        return self._t
    
    def get_channel(self, key: str):
        return StreamChannel(stream=self, key=key)
        
    def get_event_iterator(self, 
                           keys: Union[str, List[str]], 
                           record_length: int, 
                           inds: Union[int, List[int]] = None, 
                           timestamps: Union[int, List[int]] = None, 
                           alignment: float = 1/4,
                           batch_size: int = None):
        """
        Returns an iterator object over voltage traces for given trigger indices or timestamps of a stream file. 

        :param keys: The keys (channel names) of the stream object to be iterated over. 
        :type keys: Union[str, List[str]]
        :param record_length: The number of samples to be returned for each index. Usually, those are powers of 2, e.g. 16384
        :type record_length: int
        :param inds: The stream indices for which we want to read the voltage traces. This index is aligned at 1/4th of the record window. Either `inds` or `timestamps` has to be set.
        :type inds: Union[int, List[int]]
        :param timestamps: The stream timestamps for which we want to read the voltage traces. This timestamp is aligned at 1/4th of the record window. Either `inds` or `timestamps` has to be set.
        :type timestamps: Union[int, List[int]]
        :param alignment: A number in the interval [0,1] which determines the alignment of the record window (of length `record_length`) relative to the specified index. E.g. if `alignment=1/2`, the record window is centered around the index. Defaults to 1/4.
        :type alignment: float
        :param batch_size: The number of events to be returned at once (these are all read together). There will be a trade-off: large batch_sizes cause faster read speed but increase the memory usage.
        :type batch_size: int

        :return: Iterable object
        :rtype: StreamIterator
        """
        if ~np.logical_xor(inds is None, timestamps is None):
            raise ValueError("You have to specify EITHER 'inds' OR 'timestamps'.")
        
        if inds is None: inds = self.time.timestamp_to_ind(timestamps)

        return StreamIterator(self, 
                              keys=keys, 
                              inds=inds, 
                              record_length=record_length, 
                              alignment=alignment, 
                              batch_size=batch_size)
    
class StreamChannel:
    """
    An array-like object representing a single channel of a stream. For all intents and purposes, this can be treated like a numpy.ndarray.

    :param stream: The parent stream instance.
    :type stream: StreamBaseClass
    :param key: The key of the channel of 'stream' to be selected.
    :type key: str
    """
    def __init__(self, stream: StreamBaseClass, key: str):
        if key not in stream.keys:
            raise KeyError(f"'{key}' not in stream. Available keys: {stream.keys}")

        self._stream = stream
        self._key = key

    def __repr__(self):
        if len(self)<10:
            preview = ', '.join(str(x) for x in list(self[:]))
        else:
            preview = f"{self[0][0]}, {self[1][0]}, ..., {str(self[-1][0])}"

        return f"{self.__class__.__name__}([{preview}], shape={self.shape})"

    def __len__(self):
        return len(self._stream)
    
    def __getitem__(self, val) -> ArrayLike:
        return self._stream[self._key, val, 'as_voltage']
    
    @property
    def shape(self):
        return (len(self._stream),)
    
    @property
    def ndim(self):
        return 1

class StreamTime:
    """
    An object that encapsulates time data for a given Stream object. Not intended to be created by user.

    :param start_us: The first timestamp of the stream data in microseconds.
    :type start_us: int
    :param dt_us: The length of one sample in the stream data in microseconds.
    :type dt_us: int
    :param n: The number of datapoints in the stream data.
    :type n: int
    """
    def __init__(self, start_us: int, dt_us: int, n: int):
        self._start = start_us
        self._dt = dt_us
        self._n = n

    def __repr__(self):
        return f"{self.__class__.__name__}(timestamps=[{self._start}-{self._start+self._n*self._dt}], interval={self._dt}us)"
        
    def __getitem__(self, val):
        if type(val) is int:
            val = slice(val, val+1)
        
        if type(val) is slice:
            # Allows slicing [:n]
            if val.start is None:
                start = 0
            else:
                start = val.start
            
            # Allows slicing [n:]
            if val.stop is None:
                stop = self._n
            else:
                stop = val.stop
                
            # Allows slicing [:-n]
            if start < 0: start = self._n + start
            if stop <= 0: stop = self._n + stop
            
            if start > self._n or start < 0:
                raise IndexError(f'time index {start} out of range [0,{self._n})')
            if stop > self._n or stop < 0:
                raise IndexError(f'time index {stop} out of range [0,{self._n})')
                
            indices = np.arange(start=start, stop=stop, step=val.step)

        elif type(val) in [list, np.ndarray]:
            indices = np.array(val)

        else:
            raise NotImplementedError("val has to be either integer, slice, or list/np.ndarray.")

        return self._start + indices*self._dt
    
    def timestamp_to_datetime(self, timestamps: Union[int, List[int]]):
        """Function to convert timestamps to numpy.datetime objects."""
        # Call to check for out of range
        self.timestamp_to_ind(timestamps)

        return np.array(timestamps, dtype="datetime64[us]")
    
    def datetime_to_timestamp(self, datetimes: Union[np.datetime64, List[np.datetime64]]):
        """Function to convert numpy.datetime objects to timestamps."""
        out = np.array(datetimes, dtype=int)

        # Call to check for out of range
        self.timestamp_to_ind(out)

        return out
    
    def timestamp_to_ind(self, timestamps: Union[int, List[int]]):
        """Function to convert timestamps to indices."""
        out = (np.array(timestamps)-self._start)//self._dt
        if np.min(out) < 0 or np.max(out) >= self._n:
            raise IndexError("Requested timestamp is out of range.")
        
        return out