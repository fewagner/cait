
import numpy as np
from ..filter._ma import box_car_smoothing

def calc_correlated_ph(events, dominant_channel=0):
    # TODO

    nmbr_channels = events.shape[0]
    length_event = events.shape[1]

    phs = np.empty((nmbr_channels), dtype=float)
    offsets = np.mean(events[:, :int(length_event / 8)], axis=1)

    # smoothing or downsampling
    for i in range(nmbr_channels):
        events[i] = box_car_smoothing(events[i] - offsets[i])

    # get the maximal pulse height and the time of the maximum
    phs[dominant_channel] = np.max(events[dominant_channel])
    maximum_index = np.argmax(events[dominant_channel])

    for i in range(nmbr_channels):
        if i != dominant_channel:
            phs[i] = events[i, maximum_index]

    return phs