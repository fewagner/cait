import numpy as np
import cait as ai

from .streambase import StreamBaseClass
from ...functions.trigger.trigger import trigger
from ...eventfunctions.processing.removebaseline import RemoveBaseline

# Helper Function to get testpulse information from VDAQ2 files
def vdaq2_dac_channel_trigger(stream, threshold, record_length):
    channels = [x for x in stream.keys if x.lower().startswith('dac')]

    if not channels:
        raise KeyError("No DAC channels present in this stream file.")

    out_timestamps = dict()
    out_tpas = dict()

    for c in channels:
        inds, vals = trigger(stream, 
                             key=c, 
                             threshold=threshold, 
                             record_length=record_length,
                             preprocessing=[lambda x: x**2, RemoveBaseline()])
        out_timestamps[c] = stream.time[inds]
        out_tpas[c] = np.sqrt(vals) 

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