import cait as ai
import numpy as np
import warnings

# Has no test case (yet)
class StreamFile():
    """
    Factory class for providing a common access point to stream data. Currently only used when calling class:`StreamViewer`.
    Currently, only vdaq2 files are supported but an extension can be straight forwardly implemented by defining a 'read_stream_xxx' function which takes as inputs: a file `file` (of whatever file extension), a starting index `start_ind` (from which datapoint onwards the stream should be read) and a batch size `n_points` (how many datapoints should be read). The function should return a tuple where the first entry is a time array (preferably of numpy datetimes) and the second entry is a dictionary of the form `{channel_name: voltage_values}`.

    :param file: The stream file (full path including extension).
    :type file: str
    :param hardware: The hardware which was used to record the stream file. Valid options are ['vdaq2']
    :type hardware: str
    """
    def __init__(self, file: str, hardware: str):

        self.file = file

        # possibility to add more hardware options here
        if hardware == 'vdaq2':
            self.conversion_function = read_stream_vdaq2
        else:
            raise NotImplementedError("Only hardware 'vdaq2' is supported.")
        
        # get info about contents
        out = self.conversion_function(self.file, start_ind=0, n_points=1)
        self.start_timestamp_us = out[0] 
        self.available_channel_names = out[1].keys()

    def __getitem__(self, val):
        # Slicing is supported but only with step-size 0
        if type(val) is slice:
            if val.step in [0, None]:
                start = val.start
                n = val.stop - val.start
            else:
                raise ValueError("Step size > 1 is not supported.")
        
        # Single values can be retrieved, too
        elif type(val) is int:
            start = val
            n = 1

        else:
            raise NotImplementedError("Only integer and slice indexing is supported.")

        return self.conversion_function(self.file, start_ind=start, n_points=n) 

# Has no test case (yet)
def read_stream_vdaq2(file: str, start_ind: int, n_points: int) -> tuple:
    """
    Helper function that provides a tuple (time_array, {channel_name: voltage_trace}) for a vdaq2 (.bin) file containing `n_points` data points starting at data-index `start_ind`.

    :param file: The stream file (full path including extension; has to be a .bin-file).
    :type file: str
    :param start_ind: First data-index to be returned.
    :type start_ind: int
    :param n_points: Number of data points to be returned.
    :type n_points: int
    """
    # Get relevant info about file from its header
    header, keys, adc_bits, dac_bits, dt_tcp = ai.trigger.read_header(file)
    # Start timestamp of the file in us (header['timestamp'] is in ns)
    start_timestamp_us = int(header['timestamp']/1000)
    # Temporal step size in us (= inverse sampling frequency)
    dt = header['downsamplingFactor']

    # Read batch of size n_points starting from start_ind
    data = np.fromfile(file, dtype=dt_tcp, count=n_points, offset=header.nbytes + start_ind*dt_tcp.itemsize)
    # Calculate time array and convert to numpy datetime
    t = (start_timestamp_us + ( start_ind + np.arange(n_points) )*dt).astype("datetime64[us]")

    # Convert ADC and DAC values to voltage values and save traces in dictionary
    traces = dict()
    for k in keys:
        if k.startswith('ADC'): 
            bits = adc_bits
        elif k.startswith('DAC'): 
            bits = dac_bits
        else:
            warnings.warn(f"Unable to classify channel name. Skipping '{k}'.")
            continue
            
        traces[k] = ai.data.convert_to_V(data[k], bits=bits, min=-20, max=20)

    return (t, traces)