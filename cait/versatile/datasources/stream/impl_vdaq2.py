from functools import partial

import numpy as np

import cait as ai

from ....readers import BinaryFile
from ...eventfunctions.processing.removebaseline import RemoveBaseline
from ...functions.apply import apply
from ...functions.trigger.trigger_zscore import zscore_chunk
from ...functions.trigger.triggerbase import trigger_base
from .streambase import StreamBaseClass


# Helper Function to get testpulse information from VDAQ2 files
def _square(x): 
    return x**2
    
def _max_and_argmax(x):
    ind = np.argmax(x)
    return ind, x[ind]
    
def vdaq2_dac_channel_trigger(stream, key, threshold, record_length):
    with stream:
        inds, _ =  trigger_base(stream=stream[key],
                                threshold=threshold,
                                filter_fnc=partial(zscore_chunk, record_length=record_length),
                                record_length=record_length)
    
    if not inds:
        out_timestamps, out_tpas = [], []

    else:
        out_timestamps = stream.time[inds]
        it = stream.get_event_iterator(keys=key,
                                       record_length=record_length,
                                       timestamps=out_timestamps,
                                       alignment=1/4)

        # we also want argmax to correct timestamps such that they mark the maximum
        argmaxs, out_tpas = apply(_max_and_argmax,
                             it.with_processing([_square, RemoveBaseline()]),
                             n_processes=ai._available_workers)

        out_timestamps = stream.time[inds + argmaxs - record_length//4] 

    return out_timestamps, out_tpas

def to_bin_string(s, nmbr_bits=None):
    """
    Returns a string of 0/1 values for any datatype.

    :param s: Any variable or object.
    :type s: any
    :return: The 0/1's of s' bits.
    :rtype: string
    """
    bit_list = str(s) if s <= 1 else bin(s >> 1) + str(s & 1)
    if nmbr_bits is not None:
        while len(bit_list) < nmbr_bits:
            bit_list = '0' + bit_list
    return bit_list

def read_header(path_bin):
    """
    Function that reads the header of a VDAQ2 `*.bin` file.

    :param path_bin: The path to the `*.bin` file.
    :type path_bin: string
    :return: list (dictionary with infos from header,
                    list of keys that are written in each sample,
                    bool True if adc is 16 bit,
                    bool True if dac is 16 bit)
    :rtype: list
    """

    keys = []

    dt_header = np.dtype([('ID', 'i4'),
                          ('numOfBytes', 'i4'),
                          ('downsamplingFactor', 'i4'),
                          ('channelsAndFormat', 'i4'),
                          ('timestamp', 'uint64'),
                          ])

    #header = np.fromfile(path_bin, dtype=dt_header, count=1)[0]
    header = BinaryFile(path=path_bin, dtype=dt_header, count=1)[0]

    channelsAndFormat = to_bin_string(header['channelsAndFormat'], 32)

    # print('channelsAndFormat: ', channelsAndFormat)

    # bit 0: Timestamp uint64
    if channelsAndFormat[-1] == '1': keys.append('Time')
    #bit 9 sample number, currently only used by NUCLEUS experiment
    if channelsAndFormat[-10] == '1': keys.append('SampleNr')
    # bit 1: settings uint32
    if channelsAndFormat[-2] == '1': keys.append('Settings')
    # bit 2-5: dac 1-4
    for c, b in enumerate([2, 3, 4, 5]):
        if channelsAndFormat[-int(b + 1)] == '1': keys.append('DAC' + str(c + 1))
    # bit 6-8: adc 1-3 
    for c, b in enumerate([6, 7, 8]):
        if channelsAndFormat[-int(b + 1)] == '1': keys.append('ADC' + str(c + 1))
    # bit 10-16: -
    # bit 16: 0...DAC 16 bit, 1...DAC 32 bit
    dac_short = not (channelsAndFormat[-17] == '1')
    # bit 17: 0...ADC 16 bit, 1...ADC 32 bit
    adc_short = not (channelsAndFormat[-18] == '1')
    # bit 18-31: -

    # construct data type

    dt_tcp = []

    for k in keys:
        if k.startswith('Time'): dt_tcp.append((k, 'uint64'))
        elif k.startswith('SampleNr'): dt_tcp.append((k, 'uint32'))
        elif k.startswith('Settings'): dt_tcp.append((k, 'i4'))
        elif k.startswith('DAC'): dt_tcp.append((k, 'i2' if dac_short else 'i4'))
        elif k.startswith('ADC'): dt_tcp.append((k, 'i2' if adc_short else 'i4'))

    dt_tcp = np.dtype(dt_tcp)

    adc_bits = 16 if adc_short else 24
    dac_bits = 16 if dac_short else 24

    return header, keys, adc_bits, dac_bits, dt_tcp

# TODO: test cases
class Stream_VDAQ2(StreamBaseClass):
    """
    Implementation of StreamBaseClass for hardware 'vdaq2'.
    VDAQ2 data is stored in `*.bin` files. Its header contains instructions on how to read the data and all recorded channels are stored in the same file.

    :param file: The binary file with the stream data (including the file extension).
    :type file: str
    :param dac_trig_thr: Trigger threshold (in sigmas) to find testpulses from the DAC channels. Defaults to 5 sigmas
    :type dac_trig_thr: float, optional
    :param dac_trig_win_len_ms: Trigger window length (in ms) to find testpulses from the DAC channels. Defaults to 500 ms
    :type dac_trig_win_len_ms: int, optional
    """
    def __init__(self, 
                 file: str, 
                 dac_trig_thr: float = 5.,      # sigmas
                 dac_trig_win_len_ms: int = 500 # ms
                 ):
        super().__init__(file=file, dac_trig_thr=dac_trig_thr, dac_trig_win_len_ms=dac_trig_win_len_ms)

        # Get relevant info about file from its header
        header, keys, self._adc_bits, self._dac_bits, dt_tcp = read_header(file)
        # Start timestamp of the file in us (header['timestamp'] is in ns)
        self._start = int(header['timestamp']/1000)
        # Temporal step size in us (= inverse sampling frequency)
        self._dt = int(header['downsamplingFactor'])

        # VDAQ2 format could contain keys 'Settings' and 'Time' which we do not want to have as available data channels
        self._keys = list(set(keys) - set(['Time', 'Settings', 'SampleNr']))

        # Create memory map to binary file
        self._data = BinaryFile(path=file, dtype=dt_tcp, offset=header.nbytes)
        #self._data = np.memmap(file, dtype=dt_tcp, mode='r', offset=header.nbytes)

        self._dac_trig_thr = dac_trig_thr
        self._dac_trig_win_len_ms = dac_trig_win_len_ms
        self._dac_trig_win_len = int(1000*dac_trig_win_len_ms/self._dt)

        # Create placeholders for testpulses
        self._tp_timestamps = dict()
        self._tpas = dict()
        
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
    def tp_keys(self):
        return [x for x in self.keys if x.startswith("DAC")]
    
    @property
    def tpas(self):
        return VDAQ2_TPAS(self)

    @property
    def tp_timestamps(self):
        return VDAQ2_TP_TS(self)
    
class VDAQ2_TPAS:
    """A helper class for accessing testpulse amplitudes of the VDAQ2 hardware (which requires triggering a DAC channel)."""
    def __init__(self, stream: Stream_VDAQ2):
        self._stream = stream

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys()})'

    def __getitem__(self, key: str):
        if not key in self.keys():
            raise KeyError(f"Invalid testpulse key '{key}'. Valid keys: {self.keys()}")
    
        if key not in self._stream._tpas.keys():
            print(f"Triggering {key} to obtain testpulse timestamps and testpulse amplitudes ...")
            timestamps, tpas = vdaq2_dac_channel_trigger(self._stream, key, 
                                                         self._stream._dac_trig_thr,
                                                         self._stream._dac_trig_win_len)

            self._stream._tpas[key] = tpas
            self._stream._tp_timestamps[key] = timestamps

        return self._stream._tpas[key]
    
    def keys(self):
        return self._stream.tp_keys
    
class VDAQ2_TP_TS:
    """A helper class for accessing testpulse timestamps of the VDAQ2 hardware (which requires triggering a DAC channel)."""
    def __init__(self, stream: Stream_VDAQ2):
        self._stream = stream

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys()})'

    def __getitem__(self, key: str):
        if not key in self.keys():
            raise KeyError(f"Invalid testpulse key '{key}'. Valid keys: {self.keys()}")
        
        if key not in self._stream._tp_timestamps.keys():
            print(f"Triggering {key} to obtain testpulse timestamps and testpulse amplitudes ...")
            timestamps, tpas = vdaq2_dac_channel_trigger(self._stream, key, 
                                                         self._stream._dac_trig_thr,
                                                         self._stream._dac_trig_win_len)

            self._stream._tpas[key] = tpas
            self._stream._tp_timestamps[key] = timestamps

        return self._stream._tp_timestamps[key]
    
    def keys(self):
        return self._stream.tp_keys