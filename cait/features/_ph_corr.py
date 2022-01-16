import numpy as np
from ..filter._ma import box_car_smoothing
import numba as nb

def calc_correlated_ph(events, dominant_channel=0,
                       offset_to_dominant_channel=None,
                       max_search_range=50,
                       ):
    """
    Calculate the correlated pulse heights of the channels.

    :param events: The events of all channels.
    :type events: 2D array of shape (nmbr_channels, record_length)
    :param dominant_channel: Which channel is the one for the primary max search.
    :type dominant_channel: int
    :param offset_to_dominant_channel: The expected offsets of the peaks of pulses to the pesk of the dominant channel.
    :type offset_to_dominant_channel: list of ints
    :param max_search_range: The number of samples that are included in the search range of the maximum search in the
        non-dominant channels.
    :type max_search_range: int
    :return: The evaluated pulse heights.
    :rtype: numpy array of shape (nmbr_channels)
    """

    nmbr_channels = events.shape[0]
    length_event = events.shape[1]

    phs = np.empty((nmbr_channels), dtype=float)
    offsets = np.mean(events[:, :int(length_event / 8)], axis=1)

    # smoothing
    for i in range(nmbr_channels):
        events[i] = box_car_smoothing(events[i] - offsets[i])

    # get the maximal pulse height and the time of the maximum
    phs[dominant_channel] = np.max(events[dominant_channel])
    maximum_index = np.argmax(events[dominant_channel])

    for i in range(nmbr_channels):
        if i != dominant_channel:
            if offset_to_dominant_channel is None:
                start_idx = np.maximum(int(maximum_index - max_search_range / 2), 0)
            else:
                start_idx = np.maximum(int(maximum_index + offset_to_dominant_channel[i] - max_search_range / 2), 0)
            phs[i] = np.max(events[i, start_idx:start_idx + max_search_range])

    return phs
