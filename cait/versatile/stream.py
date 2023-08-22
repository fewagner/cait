import cait as ai
import numpy as np
import warnings

from typing import Union

# Has no test case (yet)
class StreamFile:
    """
    Factory class for providing a common access point to stream data. Currently only used when calling class:`StreamViewer`.
    Currently, only vdaq2 files are supported but an extension can be straight forwardly implemented by defining a 'read_stream_xxx' function which takes as inputs: a file `file` (of whatever file extension) and an integer or slice `val` (which specifies the data range that is to be read). The function should return a tuple where the first entry is a time array (preferably of numpy datetimes) and the second entry is a dictionary of the form `{channel_name: voltage_values}`.

    :param file: The stream file (full path including extension).
    :type file: str
    :param hardware: The hardware which was used to record the stream file. Valid options are ['vdaq2']
    :type hardware: str
    """
    def __init__(self, file: str, hardware: str):
        
        # possibility to add more hardware options here
        if hardware == 'vdaq2':
            self.stream = Stream_VDAQ2(file)
        else:
            raise NotImplementedError("Only hardware 'vdaq2' is supported.")
        
        # get info about contents
        out = self.stream[0]
        self.start_timestamp_us = out[0] 
        self.available_channel_names = out[1].keys()

    def __getitem__(self, val):
        return self.stream[val] 

# Has no test case (yet)
class Stream_VDAQ2:
    """
    Helper class that provides a tuple (time_array, {channel_name: voltage_trace}) for a vdaq2 (.bin) file containing data points sliced according to `val` when sliced.

    :param file: The stream file (full path including extension; has to be a .bin-file).
    :type file: str
    """
    def __init__(self, file: str):
        # Get relevant info about file from its header
        header, self.keys, self.adc_bits, self.dac_bits, dt_tcp = ai.trigger.read_header(file)
        # Start timestamp of the file in us (header['timestamp'] is in ns)
        self.start_timestamp_us = int(header['timestamp']/1000)
        # Temporal step size in us (= inverse sampling frequency)
        self.dt = header['downsamplingFactor']

        # Create memory map to binary file
        self.data = np.memmap(file, dtype=dt_tcp, mode='r', offset=header.nbytes)
    
    def __getitem__(self, val: Union[int, slice]) -> tuple:
        # Use slices consistently
        if type(val) is int:
            val = slice(val, val+1)
        elif type(val) is not slice:
            raise NotImplementedError("val has to be either integer or slice.")
            
        # Calculate time array and convert to numpy datetime (and already apply slicing)
        indices = np.arange(val.stop)[val]
        t = (self.start_timestamp_us + indices*self.dt).astype("datetime64[us]")
        
        # Convert ADC and DAC values to voltage values and save traces in dictionary
        traces = dict()
        for k in self.keys:
            if k.startswith('ADC'): 
                bits = self.adc_bits
            elif k.startswith('DAC'): 
                bits = self.dac_bits
            else:
                warnings.warn(f"Unable to classify channel name. Skipping '{k}'.")
                continue

            traces[k] = ai.data.convert_to_V(self.data[k][val], bits=bits, min=-20, max=20)

        return (t, traces)