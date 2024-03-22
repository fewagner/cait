from typing import Union, List

import numpy as np

from .iteratorbase import IteratorBaseClass

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
    def record_length(self):
        return self._record_length
    
    @property
    def dt_us(self):
        return self._stream.dt_us
    
    @property
    def timestamps(self):
        return self._stream.time[self._inds]

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