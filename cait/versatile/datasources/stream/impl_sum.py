from typing import List

import numpy as np

from .streambase import StreamBaseClass

class StreamSum(StreamBaseClass):
    """
    Implementation of StreamBaseClass that represents the sum of stream channels (used primarily for double TES analysis).
    This class is equivalent to the stream class that is used to construct it but it adds an additional key 'sum' to `stream.keys` which can be accessed and returns the sum of the voltage traces of the specified channels.

    :param stream: The stream which includes the channels we want to sum.
    :type stream: StreamBaseClass
    :param keys: List of channel names that are used to calculate the sum.
    :type keys: List[str]

    **Example:**
        ::
            import cait as ai
            import cait.versatile as vai

            # Create mock data (skip if you already have data)
            test_data = ai.data.TestData(filepath='mockdata/mock_001', duration=1000)
            test_data.generate()

            # Create stream object
            stream = vai.Stream(hardware="csmpl", src=["mockdata/mock_001_Ch0.csmpl",
                                                        "mockdata/mock_001_Ch1.csmpl",
                                                        "mockdata/mock_001.par"])

            # Check available keys (only those can be used for the sum)
            print(stream.keys)

            # Create the stream sum
            ss = vai.StreamSum(stream, ["mock_001_Ch0", "mock_001_Ch1"])

            # View the stream
            vai.StreamViewer(ss)

    """
    def __init__(self, stream: StreamBaseClass, keys: List[str]):
        if not all([k in stream.keys for k in keys]):
            raise KeyError(f"All given keys have to be present in the stream's keys. Available: {stream.keys}, got: {keys}")
        
        self._stream = stream

        self._keys = ["sum"] + stream.keys
        self._sum_keys = keys

    def __len__(self):
        return len(self._stream)
    
    def __enter__(self):
        self._stream.__enter__()
        return self
    
    def __exit__(self, typ, val, tb):
        self._stream.__exit__(typ, val, tb)
    
    def get_voltage_trace(self, key: str, where: slice):
        if key == "sum":
            return np.sum([self._stream.get_voltage_trace(k, where) for k in self._sum_keys], axis=0)
        else:
            return self._stream.get_voltage_trace(key, where)
    
    @property
    def start_us(self):
        return self._stream.start_us
    
    @property
    def dt_us(self):
        return self._stream.dt_us
    
    @property
    def keys(self):
        return self._keys
    
    @property
    def tpas(self):
        return self._stream.tpas

    @property
    def tp_timestamps(self):
        return self._stream.tp_timestamps