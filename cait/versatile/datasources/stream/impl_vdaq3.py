from typing import Union, List

import numpy as np
import cait as ai

from .streambase import StreamBaseClass

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