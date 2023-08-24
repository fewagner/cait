import cait as ai
import numpy as np

from typing import Union, List, Tuple
from abc import ABC, abstractmethod 

class StreamTime:
    def __init__(self, start_us: int, dt_us: int, n: int):
        self._start = start_us
        self._dt = dt_us
        self._n = n
        
    def __getitem__(self, val):
        if type(val) is int:
            val = slice(val, val+1)
        elif type(val) is not slice:
            raise NotImplementedError("val has to be either integer or slice.")
            
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
        if stop < 0: stop = self._n + stop + 1
        
        if start > self._n or start < 0:
            raise IndexError(f'time index {start} out of range [0,{self._n})')
        if stop > self._n or stop < 0:
            raise IndexError(f'time index {stop} out of range [0,{self._n})')
            
        indices = np.arange(start=start, stop=stop, step=val.step)
        return self._start + indices*self._dt
    
    def timestamp_to_datetime(self, timestamps):
        return np.array(timestamps, dtype="datetime64[us]")
    
    def datetime_to_timestamp(self, datetimes):
        return np.array(datetimes, dtype=int)
    
    def timestamp_to_ind(self, timestamps: Union[int, List[int]]) -> Union[int, List[int]]:
        return (np.array(timestamps)-self._start)//self._dt
    
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

    @abstractmethod
    def get_keys(self):
        ...

    @abstractmethod
    def get_start_us(self):
        ...

    @abstractmethod
    def get_dt_us(self):
        ...
    
class Stream:
    """
    Factory class for providing a common access point to stream data.
    Currently, only vdaq2 files are supported but an extension can be straight forwardly implemented by sub-classing :class:`StreamBaseClass` and adding it in the constructor.

    The data is accessed by means of slicing (see below). The `time` property is an object of :class:`StreamTime` and offers a convenient time interface as well (see below).

    :param src: The source for the stream. Depending on how the data is taken, this can either be the path to one file or a list of paths to multiple files. This input is handled by the specific implementation of the Stream Object. See below for examples.
    :type src: Union[str, List[str]]
    :param hardware: The hardware which was used to record the stream file. Valid options are ['vdaq2']
    :type hardware: str
    
    :Usage for different hardware:
    VDAQ2:
    Files are .bin files which contain all information necessary to construct the Stream object. It can be input as a single argument.
    >>> s = Stream(src=file.bin, hardware='vdaq2')
    
    VDAQ3 (not yet released):
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
        else:
            raise NotImplementedError('Only vdaq2 files are supported at the moment.')
            
        self._t = StreamTime(self._stream.get_start_us(), self._stream.get_dt_us(), len(self._stream))
        
    def __len__(self):
        return len(self._stream)
    
    def __getitem__(self, val: Union[str, Tuple[str, Union[int, slice]], Tuple[str, Union[int, slice], str]]):
        # Only names and tuples are supported for slicing (no int)
        if type(val) not in [str, tuple]:
            raise TypeError(f'Unsupported type {type(val)} for slicing.')
            
        # If only a name is given, the channel with the respective name is returned
        if type(val) is str:
            if val in self.keys:
                return self._stream.get_channel(val)
            else:
                raise KeyError(f'{val} not in stream. Available names: {self.keys}')
        
        # For tuples of length 2 or 3 there are specific behaviors
        else:
            if len(val) in [0, 1]:
                raise TypeError('No slicing support for tuples of length 0 or 1.')
                
            # Return the integer values for the stream 'name' if everything else is fine
            elif len(val) == 2:
                if type(val[0]) is str and type(val[1]) in [int, slice]:
                    if val[0] in self.keys:
                        return self._stream.get_channel(val[0])[val[1]]
                    else:
                        raise KeyError(f'{val[0]} not in stream. Available names: {self.keys}')
                else:
                    raise TypeError('When slicing with two arguments, the first and second one have to be of type string and int/slice, respectively.')
            # Return the voltage values for the stream 'name' if everything else is fine
            elif len(val) == 3:
                if type(val[0]) is str and type(val[1]) in [int, slice] and type(val[2]) is str:
                    if val[0] not in self.keys:
                        raise KeyError(f'{val[0]} not in stream. Available names: {self.keys}')
                    if val[2] != 'as_voltage':
                        raise ValueError(f'Unrecognized string "{val[2]}". Did you mean "as_voltage"?')
                    
                    where = slice(val[1], val[1]+1) if type(val[1]) is int else val[1]
                    return self._stream.get_voltage_trace(name=val[0], where=where)
                else:
                    raise TypeError('When slicing with three arguments, the first, second and third one have to be of type string, int/slice and string, respectively.')
            else:  
                raise NotImplementedError(f'Tuples of length {len(val)} are not supported for slicing')
    
    @property
    def keys(self):
        return self._stream.get_keys()
        
    @property
    def time(self):
        return self._t
    
    @property
    def start_us(self):
        return self._stream.get_start_us()
    
    @property
    def dt_us(self):
        return self._stream.get_dt_us()

class Stream_VDAQ2(StreamBaseClass):
    """
    Implementation of StreamBaseClass for hardware 'vdaq2'.
    VDAQ2 data is stored in .bin files. Its header contains instructions on how to read the data and all recorded channels are stored in the same file.
    """
    def __init__(self, file: str):
        # Get relevant info about file from its header
        header, self._keys, self._adc_bits, self._dac_bits, dt_tcp = ai.trigger.read_header(file)
        # Start timestamp of the file in us (header['timestamp'] is in ns)
        self._start = int(header['timestamp']/1000)
        # Temporal step size in us (= inverse sampling frequency)
        self._dt = header['downsamplingFactor']

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
    
    def get_start_us(self):
        return self._start
    
    def get_dt_us(self):
        return self._dt
    
    def get_keys(self):
        return self._keys