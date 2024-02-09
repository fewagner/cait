import os

from abc import ABC, abstractmethod
from typing import Union, List, Tuple

import cait as ai
import cait.versatile as vai
import numpy as np

from ..iterators import StreamIterator
from ..rdt import PARFile
from ..functions import RemoveBaseline

# Helper Function to get testpulse information from VDAQ2 files
def vdaq2_dac_channel_trigger(stream, threshold, record_length):
    channels = [x for x in stream.keys if x.lower().startswith('dac')]

    if not channels:
        raise KeyError("No DAC channels present in this stream file.")

    out_timestamps = dict()
    out_tpas = dict()

    for c in channels:
        inds, vals = vai.trigger(stream, 
                             key=c, 
                             threshold=threshold, 
                             record_length=record_length,
                             preprocessing=[lambda x: x**2, RemoveBaseline()])
        out_timestamps[c] = stream.time[inds]
        out_tpas[c] = np.sqrt(vals) 

    return out_timestamps, out_tpas

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
    def get_channel(self, key: str):
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
        Dictionary of testpulse amplitudes in the stream. For hardware 'cresst' this is read from a '.test_stamps' file. For hardware 'vdaq2' this is obtained from triggering the DAC channels first.
        
        :return: Testpulse amplitudes
        :rtype: dict of `np.ndarray`
        """
        ...

    @property
    @abstractmethod
    def tp_timestamps(self):
        """
        Dictionary of testpulse timestamps (microseconds) in the stream. For hardware 'cresst' this is read from a '.test_stamps' file. For hardware 'vdaq2' this is obtained from triggering the DAC channels first.
        
        :return: Testpulse microsecond timestamps.
        :rtype: dict of `np.ndarray`
        """
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
    
    def get_event_iterator(self, keys: Union[str, List[str]], record_length: int, inds: Union[int, List[int]] = None, timestamps: Union[int, List[int]] = None, alignment: float = 1/4):
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


        :return: Iterable object
        :rtype: StreamIterator
        """
        if ~np.logical_xor(inds is None, timestamps is None):
            raise ValueError("You have to specify EITHER 'inds' OR 'timestamps'.")
        
        if inds is None: inds = self.time.timestamp_to_ind(timestamps)

        return StreamIterator(self, keys, inds, record_length, alignment)

class Stream_CRESST(StreamBaseClass):
    """
    Implementation of StreamBaseClass for hardware 'cresst'.
    CRESST data is stored in `*.csmpl` files (for each channel separately). Additionally, we need a `*.par` file to read the start timestamp of the stream data from.
    """
    def __init__(self, files: List[str]):
        if not any([x.endswith('.par') for x in files]):
            raise ValueError("You have to provide a '.par' file to construct this class.")
        if not any([x.endswith('.csmpl') for x in files]):
            raise ValueError("You have to provide at least one '.csmpl' file to construct this class.")
        if any([os.path.splitext(x)[-1] not in [".csmpl", ".par", ".test_stamps", ".dig_stamps"] for x in files]):
            raise ValueError("Only file extensions ['.csmpl', '.par'] are supported.")
        
        par_path = [x for x in files if x.endswith('.par')][0]
        csmpl_paths = [x for x in files if x.endswith('.csmpl')]
        test_path = [x for x in files if x.endswith('.test_stamps')]
        dig_path = [x for x in files if x.endswith('.dig_stamps')]

        # Offset from the dig_stamps file (assuming a 10 MHz clock)
        offset = 0 if not dig_path else int(ai.trigger._csmpl.get_offset(dig_path[0])/10)

        self._par_file = PARFile(par_path)
        self._start = int(1e6*self._par_file.start_s + self._par_file.start_us - offset)
        self._dt = self._par_file.time_base_us

        self._data = dict()

        for f in csmpl_paths:
            name = os.path.splitext(os.path.basename(f))[0]
            self._data[name] = ai.trigger._csmpl.readcs(f)

        if test_path:
            if not dig_path:
                raise Exception("When including testpulse information using a '.test_stamps' file, you also have to provide the corresponding '.dig_stamps' file.")
            test_path = test_path[0]
            dtype = np.dtype([('stamp', np.uint64),
                                ('tpa', np.float32),
                                ('tpch', np.uint32)])
            stamps = np.fromfile(test_path, dtype=dtype)

            self._tpas = dict()
            self._tp_timestamps = dict()

            for k in list(set(stamps['tpch'])):
                mask = stamps['tpch'] == k
                self._tpas[str(k)] = stamps['tpa'][mask]
                # assuming 10 MHz clock
                self._tp_timestamps[str(k)] = self.start_us + stamps['stamp'][mask]/10 + offset

        self._keys = list(self._data.keys())

    def __len__(self):
        return len(self._data[self.keys[0]])
    
    def get_channel(self, key: str):
        return self._data[key]
    
    def get_voltage_trace(self, key: str, where: slice):
       return ai.data.convert_to_V(self._data[key][where], bits=16, min=-10, max=10)
    
    @property
    def start_us(self):
        return self._start
    
    @property
    def dt_us(self):
        return self._dt
    
    @property
    def keys(self):
        return self._keys
    
    @property
    def tpas(self):
        if not hasattr(self, '_tpas'):
            raise KeyError("Testpulse amplitudes not available. Include a '.test_stamps' and a '.dig_stamps' file when constructing this class to use this feature.")
        return self._tpas

    @property
    def tp_timestamps(self):
        if not hasattr(self, '_tp_timestamps'):
            raise KeyError("Testpulse timestamps not available. Include a '.test_stamps' and a '.dig_stamps' file when constructing this class to use this feature.")
        return self._tp_timestamps

# TODO: test cases
class Stream_VDAQ2(StreamBaseClass):
    """
    Implementation of StreamBaseClass for hardware 'vdaq2'.
    VDAQ2 data is stored in `*.bin` files. Its header contains instructions on how to read the data and all recorded channels are stored in the same file.
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

        # Create placeholders for testpulses
        self._tp_timestamps = None
        self._tpas = None
        
    def __len__(self):
        return len(self._data)
    
    def get_channel(self, key: str):
        return self._data[key]
    
    def get_voltage_trace(self, key: str, where: slice):
        if key.lower().startswith('adc'): 
            bits = self._adc_bits
        elif key.lower().startswith('dac'):
            bits = self._dac_bits
        else:
            raise ValueError(f'Unable to assign the correct itemsize to name "{key}" as it does not start with "ADC" or "DAC".')
            
        return ai.data.convert_to_V(self._data[key][where], bits=bits, min=-20, max=20)
    
    @property
    def start_us(self):
        return self._start
    
    @property
    def dt_us(self):
        return self._dt
    
    @property
    def keys(self):
        return self._keys
    
    @property
    def tpas(self):
        if self._tpas is None:
            # Trigger with generic threshold 0.001 V and record length 1 sec
            timestamps, tpas = vdaq2_dac_channel_trigger(self, 0.001, int(1e6/self.dt_us))

            self._tpas = tpas
            self._tp_timestamps = timestamps

        return self._tpas

    @property
    def tp_timestamps(self):
        if self._tp_timestamps is None:
            # Trigger with generic threshold 0.001 V and record length 1 sec
            timestamps, tpas = vdaq2_dac_channel_trigger(self, 0.001, int(1e6/self.dt_us))

            self._tpas = tpas
            self._tp_timestamps = timestamps

        return self._tp_timestamps

# TODO: finally implement and test cases    
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
    
    def get_channel(self, key: str):
        return self._data[key]
    
    def get_voltage_trace(self, key: str, where: slice):
        # VDAQ3 writes 24bit values, here, we convert them to 32 bits such that numpy can handle them
        adc_32bit = np.vstack([self._data[key]["byte1"][where], 
                               self._data[key]["byte2"][where], 
                               self._data[key]["f3"][where],
                               np.zeros_like(self._data[key]["byte1"][where]),
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
    
    @property
    def tpas(self):
        raise NotImplementedError("Not yet implemented")

    @property
    def tp_timestamps(self):
        raise NotImplementedError("Not yet implemented")

class Stream(StreamBaseClass):
    """
    Factory class for providing a common access point to stream data.
    Currently, only vdaq2 and vdaq3 files are supported but an extension can be straight forwardly implemented by sub-classing :class:`StreamBaseClass` and adding it in the constructor.

    The data is accessed by means of slicing (see below). The `time` property is an object of :class:`StreamTime` and offers a convenient time interface as well (see below).

    :param hardware: The hardware which was used to record the stream file. Valid options are ['cresst', 'vdaq2']
    :type hardware: str
    :param src: The source for the stream. Depending on how the data is taken, this can either be the path to one file or a list of paths to multiple files. This input is handled by the specific implementation of the Stream Object. See below for examples.
    :type src: Union[str, List[str]]
    
    :Usage for different hardware:
    CRESST:
    Files are .csmpl files which contain one channel each. Additionally, we need a .par file to read the start timestamp of the stream data from.
    >>> s = Stream(hardware='cresst', src=['par_file.par', 'stream_Ch0.csmpl', 'stream_Ch1.csmpl'])

    VDAQ2:
    Files are .bin files which contain all information necessary to construct the Stream object. It can be input as a single argument.
    >>> s = Stream(hardware='vdaq2', src='file.bin')

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
    def __init__(self, hardware: str, src: Union[str, List[str]]):
        if hardware.lower() == "cresst":
            self._stream = Stream_CRESST(src)
        elif hardware.lower() == "vdaq2":
            self._stream = Stream_VDAQ2(src)
        #elif hardware.lower() == "vdaq3":
        #    self._stream = Stream_VDAQ3(src)
        else:
            raise NotImplementedError('Only cresst and vdaq2 files are supported at the moment.')

    def __repr__(self):
        return repr(self._stream)
    
    def __len__(self):
        return len(self._stream)
        
    def get_channel(self, key: str):
        return self._stream.get_channel(key)
        
    def get_voltage_trace(self, key: str, where: slice):
        return self._stream.get_voltage_trace(key, where)

    @property
    def keys(self):
        return self._stream.keys

    @property
    def start_us(self):
        return self._stream.start_us

    @property
    def dt_us(self):
        return self._stream.dt_us
    
    @property
    def tpas(self):
        return self._stream.tpas

    @property
    def tp_timestamps(self):
        return self._stream.tp_timestamps