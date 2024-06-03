import os
from typing import List

import numpy as np
import cait as ai

from .streambase import StreamBaseClass
from ..hardwaretriggered.par_file import PARFile

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