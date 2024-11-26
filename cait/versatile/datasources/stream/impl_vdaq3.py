from typing import Union, List
import json

import numpy as np
import cait as ai

from .streambase import StreamBaseClass
from ....readers import BinaryFile

# TODO: finally implement and test cases    
class Stream_VDAQ3(StreamBaseClass):
    """
    Implementation of StreamBaseClass for hardware 'vdaq3'.
    VDAQ3 data is stored in .bin files. Its header contains instructions on how to read the data and all recorded channels are stored in the separate file.
    """
    def __init__(self, files: Union[str, List[str]]):
        if type(files) is str: files = [files]

        self._data, starts, dTs, lengths, checks, precs = dict(), [], [], [], [], []
        
        # Data is 24 bits, i.e. 3 bytes long. Read 3 bytes at a time
        # Possibly also 32 bit
        # This is decided by 'sample_size_bytes' in the header which is either 3 or 4
        data_dtype = {
            3: np.dtype([('byte1', '<u1'), 
                         ('byte2', '<u1'), 
                         ('byte3', '<u1')]),
            4: np.dtype('<u4')
        }

        for f in files:
            # The first two bytes give the header size
            header_size = BinaryFile(path=f, dtype=np.dtype(np.int16), count=1)[0]

            # The header contains a json string with all information
            raw_header = BinaryFile(path=f,
                                    dtype=np.dtype(f"V{header_size-2}"),
                                    offset=2,
                                    count=1)[0]
            
            check, header_str, *_ = raw_header.tobytes().decode("ascii").split("\x00")
            header_json = json.loads(header_str)
            
            # Before the json string starts, there is a value to check whether
            # we are dealing with single channel files that belong together
            checks.append(check)
            
            dTs.append(header_json["timestep_ns"])
            
            # There is also a 'ts_ns' which gives the clock of the DAQ
            starts.append(header_json["ts_utc_ns"]//1000)

            channel_name = header_json["channel_id"]
            prec = header_json["sample_size_bytes"]
            
            if prec not in [3, 4]:
                raise NotImplementedError(f"Only 3 and 4 byte data precision are supported. Got {prec}.")
                
            precs.append(prec)

            self._data[f"Ch{channel_name}"] = BinaryFile(path=f, 
                                                         dtype=data_dtype[prec], 
                                                         offset=header_size)

            # HERE WE COULD PROBABLY USE THE INFO IN THE HEADER AT SOME POINT
            lengths.append(len(self._data[f"Ch{channel_name}"]))
        
        if len(np.unique(starts)) > 1:
            raise ValueError(f'Files have to start at the same time to be treated together. Got {starts}.')
        if len(np.unique(dTs)) > 1:
            raise ValueError(f'Files have to have the same time-delta to be treated together. Got {dTs}.')
        if len(np.unique(lengths)) > 1:
            raise ValueError(f'Files have to have the same length to be treated together. Got {lengths}.')
        if len(np.unique(checks)) > 1:
            raise ValueError(f'Files have to have the same check string to be treated together. Got {checks}.')
        if len(np.unique(precs)) > 1:
            raise ValueError(f'Files have to be recorded with identical byte precision to be treated together. Got {precs}.')
        
        # Number of data points in stream
        self._len = lengths[0]
        # Start timestamp of the file in us (header['nsTimeStamp64'] is in ns)
        self._start = int(starts[0]/1000)
        # Temporal step size in us (= inverse sampling frequency)
        self._dt = dTs[0]//1000
        # The byte precision
        self._prec = precs[0]
        
    def __len__(self):
        return self._len
    
    def __enter__(self):
        for bin_file in self._data.values(): bin_file.__enter__()
        return self
    
    def __exit__(self, typ, val, tb):
        for bin_file in self._data.values(): bin_file.__exit__(typ, val, tb)
    
    def get_trace(self, key: str, where: slice, voltage: bool = True):
        data = self._data[key][where]
        
        if self._prec == 3:
            # If written as 24bit values, here, we convert them to 32 bits such that numpy can handle them
            adc_32bit = np.vstack([
                    data["byte1"], 
                    data["byte2"], 
                    data["byte3"], 
                    np.zeros_like(data["byte1"])
                ]).flatten("F").view("<u4")

            return ai.data.convert_to_V(adc_32bit, bits=32, min=-20, max=20) if voltage else adc_32bit
        
        elif self._prec == 4:
            # If already written with 32bit, we don't have to do anything further
            return ai.data.convert_to_V(adc_32bit, bits=32, min=-20, max=20) if voltage else data
    
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
    def tp_keys(self):
        # Not yet implemented
        return []
    
    @property
    def tpas(self):
        raise NotImplementedError("Not yet implemented")

    @property
    def tp_timestamps(self):
        raise NotImplementedError("Not yet implemented")