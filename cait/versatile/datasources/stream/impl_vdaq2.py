import os
from functools import partial

import numpy as np
import cait as ai

from .streambase import StreamBaseClass
from ...functions.apply import apply
from ...functions.trigger.triggerbase import trigger_base
from ...functions.trigger.trigger_zscore import zscore_chunk
from ...eventfunctions.processing.removebaseline import RemoveBaseline
from ....readers import BinaryFile

# Helper Function to get testpulse information from VDAQ2 files
def vdaq2_dac_channel_trigger(stream, threshold, record_length):
    channels = [x for x in stream.keys if x.lower().startswith('dac')]

    if not channels:
        raise KeyError("No DAC channels present in this stream file.")

    out_timestamps = dict()
    out_tpas = dict()

    for c in channels:
        inds, _ =  trigger_base(stream=stream[c],
                            threshold=threshold,
                            filter_fnc=partial(zscore_chunk, record_length=record_length),
                            record_length=record_length)
        
        out_timestamps[c] = stream.time[inds]
        it = stream.get_event_iterator(keys=c, 
                                       record_length=record_length//5, 
                                       timestamps=out_timestamps[c],
                                       alignment=1/2)
        out_tpas[c] = apply(np.max, 
                            it.with_processing([partial(np.power, 2), RemoveBaseline()]),
                            n_processes=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())

    return out_timestamps, out_tpas

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
        self._dt = int(header['downsamplingFactor'])

        # VDAQ2 format could contain keys 'Settings' and 'Time' which we do not want to have as available data channels
        self._keys = list(set(keys) - set(['Time', 'Settings', 'SampleNr']))

        # Create memory map to binary file
        self._data = BinaryFile(path=file, dtype=dt_tcp, offset=header.nbytes)
        #self._data = np.memmap(file, dtype=dt_tcp, mode='r', offset=header.nbytes)

        # Create placeholders for testpulses
        self._tp_timestamps = None
        self._tpas = None
        
    def __len__(self):
        return len(self._data)
    
    def __enter__(self):
        self._data.__enter__()
        return self
    
    def __exit__(self, typ, val, tb):
        self._data.__exit__(typ, val, tb)
    
    def get_trace(self, key: str, where: slice, voltage: bool = True):
        if key.lower().startswith('adc'): 
            bits = self._adc_bits
        elif key.lower().startswith('dac'):
            bits = self._dac_bits
        else:
            raise ValueError(f'Unable to assign the correct itemsize to name "{key}" as it does not start with "ADC" or "DAC".')
        
        data = self._data[where][key]
        return ai.data.convert_to_V(data, bits=bits, min=-20, max=20) if voltage else data
    
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
            # Trigger with generic threshold 5 z-scores and record length 1 sec
            timestamps, tpas = vdaq2_dac_channel_trigger(self, 5, int(1e6/self.dt_us))

            self._tpas = tpas
            self._tp_timestamps = timestamps

        return self._tpas

    @property
    def tp_timestamps(self):
        if self._tp_timestamps is None:
            # Trigger with generic threshold 5 z-scores and record length 1 sec
            timestamps, tpas = vdaq2_dac_channel_trigger(self, 5, int(1e6/self.dt_us))

            self._tpas = tpas
            self._tp_timestamps = timestamps

        return self._tp_timestamps