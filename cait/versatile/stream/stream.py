from abc import ABC, abstractmethod
from typing import Union, List, Tuple

import cait as ai
import numpy as np

from ..iterators import StreamIterator

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

class StreamBaseClass(ABC):
    @abstractmethod
    def __len__(self):
        ...
        
    @abstractmethod
    def get_channel(self, name: str):
        ...
        
    @abstractmethod
    def get_voltage_trace(self, name: str, where: slice):
        ...

    @property
    @abstractmethod
    def keys(self):
        ...

    @property
    @abstractmethod
    def start_us(self):
        ...

    @property
    @abstractmethod
    def dt_us(self):
        ...

    def __repr__(self):
        return f'{self.__class__.__name__}(start_us={self.start_us}, dt_us={self.dt_us}, length={self.__len__()}, keys={self.keys})'

    def __getitem__(self, val: Union[str, Tuple[str, Union[int, slice, list, np.ndarray]], Tuple[str, Union[int, slice, list, np.ndarray], str]]):
        # Only names and tuples are supported for slicing (no int)
        if type(val) not in [str, tuple]:
            raise TypeError(f'Unsupported type {type(val)} for slicing.')
            
        # If only a name is given, the channel with the respective name is returned
        if type(val) is str:
            if val in self.keys:
                return self.get_channel(val)
            else:
                raise KeyError(f'{val} not in stream. Available names: {self.keys}')
        
        # For tuples of length 2 or 3 there are specific behaviors
        else:
            if len(val) in [0, 1]:
                raise TypeError('No slicing support for tuples of length 0 or 1.')
                
            # Return the integer values for the stream 'name' if everything else is fine
            elif len(val) == 2:
                if type(val[0]) is str and type(val[1]) in [int, slice, list, np.ndarray]:
                    if val[0] in self.keys:
                        return self.get_channel(val[0])[val[1]]
                    else:
                        raise KeyError(f'{val[0]} not in stream. Available names: {self.keys}')
                else:
                    raise TypeError('When slicing with two arguments, the first and second one have to be of type string and int/slice, respectively.')
            # Return the voltage values for the stream 'name' if everything else is fine
            elif len(val) == 3:
                if type(val[0]) is str and type(val[1]) in [int, slice, list, np.ndarray] and type(val[2]) is str:
                    if val[0] not in self.keys:
                        raise KeyError(f'{val[0]} not in stream. Available names: {self.keys}')
                    if val[2] != 'as_voltage':
                        raise ValueError(f'Unrecognized string "{val[2]}". Did you mean "as_voltage"?')
                    
                    where = slice(val[1], val[1]+1) if type(val[1]) is int else val[1]
                    return self.get_voltage_trace(name=val[0], where=where)
                else:
                    raise TypeError('When slicing with three arguments, the first, second and third one have to be of type string, int/slice and string, respectively.')
            else:  
                raise NotImplementedError(f'Tuples of length {len(val)} are not supported for slicing')
    
    @property
    def time(self):
        if not hasattr(self, "_t"):
            self._t = StreamTime(self.start_us, self.dt_us, len(self))
        return self._t
    
    def get_event_iterator(self, keys: Union[str, List[str]], record_length: int, inds: Union[int, List[int]] = None, timestamps: Union[int, List[int]] = None, alignment: float = 1/4):
        """
        Returns an iterator object over voltage traces for given trigger indices or timestamps of a stream file. 

        :param keys: The keys (channel names) of the stream object to be iterated over. 
        :type keys: Union[str, List[str]]
        :param record_length: The number of samples to be returned for each index. Usually, those are powers of 2, e.g. 16384
        :type record_length: int
        :param inds: The stream indices for which we want to read the voltage traces. How this index is aligned in the returned record window is dictated by the `alignment` argument. Either `inds` or `timestamps` has to be set.
        :type inds: Union[int, List[int]]
        :param timestamps: The stream timestamps for which we want to read the voltage traces. How this timestamp is aligned in the returned record window is dictated by the `alignment` argument. Either `inds` or `timestamps` has to be set.
        :type timestamps: Union[int, List[int]]
        :param alignment: A number in the interval [0,1] which determines the alignment of the record window (of length `record_length`) relative to the specified index. E.g. if `alignment=1/2`, the record window is centered around the index. Defaults to 1/4.
        :type alignment: float

        :return: Iterable object
        :rtype: StreamIterator
        """
        if ~np.logical_xor(inds is None, timestamps is None):
            raise ValueError("You have to specify EITHER 'inds' OR 'timestamps'.")
        
        if inds is None: inds = self.time.timestamp_to_ind(timestamps)

        return StreamIterator(self, keys, inds, record_length, alignment)

class Stream_VDAQ2(StreamBaseClass):
    """
    Implementation of StreamBaseClass for hardware 'vdaq2'.
    VDAQ2 data is stored in .bin files. Its header contains instructions on how to read the data and all recorded channels are stored in the same file.
    """
    def __init__(self, file: str):
        # Get relevant info about file from its header
        header, keys, self._adc_bits, self._dac_bits, dt_tcp = ai.trigger.read_header(file)
        # Start timestamp of the file in us (header['timestamp'] is in ns)
        self._start = int(header['timestamp']/1000)
        # Temporal step size in us (= inverse sampling frequency)
        self._dt = header['downsamplingFactor']

        # VDAQ2 format could contain keys 'Settings' and 'Time' which we do not want to have as available data channels
        self._keys = list(set(keys) - set(['Time', 'Settings']))

        # Create memory map to binary file
        self._data = np.memmap(file, dtype=dt_tcp, mode='r', offset=header.nbytes)
        
    def __len__(self):
        return len(self._data)
    
    def get_channel(self, name: str):
        return self._data[name]
    
    def get_voltage_trace(self, name: str, where: slice):
        if name.lower().startswith('adc'): 
            bits = self._adc_bits
        elif name.lower().startswith('dac'):
            bits = self._dac_bits
        else:
            raise ValueError(f'Unable to assign the correct itemsize to name "{name}" as it does not start with "ADC" or "DAC".')
            
        return ai.data.convert_to_V(self._data[name][where], bits=bits, min=-20, max=20)
    
    @property
    def start_us(self):
        return self._start
    
    @property
    def dt_us(self):
        return self._dt
    
    @property
    def keys(self):
        return self._keys
    
class Stream_VDAQ3(StreamBaseClass):
    """
    Implementation of StreamBaseClass for hardware 'vdaq3'.
    VDAQ3 data is stored in .bin files. Its header contains instructions on how to read the data and all recorded channels are stored in the separate file.
    """
    def __init__(self, files: Union[str, List[str]]):
        if type(files) is str: files = [files]

        self._data, starts, dTs, lengths = dict(), [], [], []

        for file in files:
            dt_header = np.dtype([('startOfMessageID', 'i4'), 
                                  ('ChannelID', 'i4'),
                                  ('messageSizeInBytes', 'i4'),
                                  ('messageTypeID', 'i4'),
                                  ('nsTslUTC', 'i4'),
                                  ('nsTshUTC', 'i4'),
                                  ('nsTimeStamp64', 'i8'),
                                  ('nChannels', 'i4'),
                                  ('nSamples', 'i4'),
                                  ('nsTimeStep', 'i4'),
                                  ('idk', 'i4')
                                  ])

            header = np.fromfile(file, dtype=dt_header, count=1)[0]

            channel_name = header["ChannelID"] # maybe also filename, idk yet
            starts.append(header["nsTimeStamp64"])
            dTs.append(header["nsTimeStep"])

            # Data is 24 bits, i.e. 3 bytes long. Read 3 bytes at a time
            # MIGHT CHANGE
            dt_tcp = np.dtype([('byte1', '<u1'), 
                               ('byte2', '<u1'), 
                               ('byte3', '<u1')
                               ])

            self._data[str(channel_name)] = np.memmap(file, dtype=dt_tcp, mode='r', offset=header.nbytes)

            # HERE WE COULD PROBABLY USE THE INFO IN THE HEADER AT SOME POINT
            lengths.append(len(self._data[str(channel_name)]))
        
        if len(np.unique(starts)) > 1:
            raise Exception('Files have to start at the same time to be treated together.')
        if len(np.unique(dTs)) > 1:
            raise Exception('Files have to have the same time-delta to be treated together.')
        if len(np.unique(lengths)) > 1:
            raise Exception('Files have to have the same length to be treated together.')
        
        # Number of data points in stream
        self._len = lengths[0]
        # Start timestamp of the file in us (header['nsTimeStamp64'] is in ns)
        self._start = int(starts[0]/1000)
        # Temporal step size in us (= inverse sampling frequency)
        self._dt = dTs[0]/1000
        
    def __len__(self):
        return self._len
    
    def get_channel(self, name: str):
        return self._data[name]
    
    def get_voltage_trace(self, name: str, where: slice):
        # VDAQ3 writes 24bit values, here, we convert them to 32 bits such that numpy can handle them
        adc_32bit = np.vstack([self._data[name]["byte1"][where], 
                               self._data[name]["byte2"][where], 
                               self._data[name]["f3"][where],
                               np.zeros_like(self._data[name]["byte1"][where]),
                               ]).flatten("F").view("<u4")
 
        return ai.data.convert_to_V(adc_32bit, bits=32, min=-20, max=20)
    
    @property
    def start_us(self):
        return self._start
    
    @property
    def dt_us(self):
        return self._dt
    
    @property
    def keys(self):
        return list(self._data.keys())

class Stream(StreamBaseClass):
    """
    Factory class for providing a common access point to stream data.
    Currently, only vdaq2 and vdaq3 files are supported but an extension can be straight forwardly implemented by sub-classing :class:`StreamBaseClass` and adding it in the constructor.

    The data is accessed by means of slicing (see below). The `time` property is an object of :class:`StreamTime` and offers a convenient time interface as well (see below).

    :param src: The source for the stream. Depending on how the data is taken, this can either be the path to one file or a list of paths to multiple files. This input is handled by the specific implementation of the Stream Object. See below for examples.
    :type src: Union[str, List[str]]
    :param hardware: The hardware which was used to record the stream file. Valid options are ['vdaq2']
    :type hardware: str
    
    :Usage for different hardware:
    VDAQ2:
    Files are .bin files which contain all information necessary to construct the Stream object. It can be input as a single argument.
    >>> s = Stream(src=file.bin, hardware='vdaq2')
    
    VDAQ3:
    This hardware records different channels in separate files but each such file contains all relevant informations to identify them. All files can be input as a list.
    >>> s = Stream(src=[channel0.bin, channel1.bin], hardware='vdaq3')

    :Usage slicing:
    Valid options for slicing streams are the following:

    >>> # Get data for one channel
    >>> s['ADC1']

    >>> # Get data for one channel and slice it (two equivalent ways)
    >>> s['ADC1', 10:20]
    >>> s['ADC1'][10:20]

    >>> # Get data for one channel, slice it, and return the voltage 
    >>> # values instead of the ADC values
    >>> s['ADC1', 10:20, 'as_voltage']
    """
    def __init__(self, src: Union[str, List[str]], hardware: str):
        if hardware.lower() == "vdaq2":
            self._stream = Stream_VDAQ2(src)
        elif hardware.lower() == "vdaq3":
            self._stream = Stream_VDAQ3(src)
        else:
            raise NotImplementedError('Only vdaq2 and vdaq3 files are supported at the moment.')

    def __repr__(self):
        return repr(self._stream)
    
    def __len__(self):
        return len(self._stream)
        
    def get_channel(self, name: str):
        return self._stream.get_channel(name)
        
    def get_voltage_trace(self, name: str, where: slice):
        return self._stream.get_voltage_trace(name, where)

    @property
    def keys(self):
        return self._stream.keys

    @property
    def start_us(self):
        return self._stream.start_us

    @property
    def dt_us(self):
        return self._stream.dt_us