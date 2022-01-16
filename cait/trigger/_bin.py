# imports

import numpy as np
from ..data._raw import convert_to_V
from ._csmpl import time_to_sample


# functions

def get_record_window_vdaq(path,
                      start_time,  # in s
                      record_length,
                      dtype,
                      key,
                      header_size,
                      sample_duration=0.00004,
                      down=1,
                      bits=16,
                      vswing=39.3216,
                      ):
    """
    Get a record window from a stream *.bin file.

    :param path: The full path of the *.bin file.
    :type path: str
    :param start_time: The start time in seconds, from where we want to read the record window, starting with 0 at
        the beginning of the file.
    :type start_time: float
    :param record_length: The record length to read from the bin file.
    :type record_length: int
    :param dtype: The data type with which we read the *.bin file.
    :type dtype: numpy data type
    :param key: The key of the dtype, corresponding to the channel that we want to read.
    :type key: str
    :param header_size: The size of the file header of the bin file, in bytes.
    :type header_size: int
    :param sample_duration: The duration of a sample, in seconds.
    :type sample_duration: float
    :param down: A factor by which the events are downsampled before they are returned.
    :type down: int
    :param bits: The precision of the digitizer.
    :type bits: int
    :param vswing: The total volt region covered by the ADC.
    :type vswing: float
    :return: List of two 1D numpy arrays: The event read from the *.bin file, and the corresponding time grid.
    :rtype: list
    """

    offset = header_size + dtype.itemsize * time_to_sample(start_time, sample_duration=sample_duration)
    offset = np.maximum(offset, header_size)

    event = np.fromfile(path,
                        offset=int(offset),
                        count=record_length,
                        dtype=dtype)

    event = convert_to_V(event[key], bits=bits, max=vswing/2, min=-vswing/2)

    # handling end of file and fill up with small random values to avoid division by zero
    if len(event) < record_length:
        new_event = np.random.normal(scale=1e-5, size=record_length)
        new_event[:len(event)] = event
        event = np.copy(new_event)
        del new_event

    if down > 1:
        event = np.mean(event.reshape(int(len(event) / down), down), axis=1)
    time = start_time + np.arange(0, record_length / down) * sample_duration * down

    return event, time