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

    TODO
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