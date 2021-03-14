# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------

import numpy as np
from numpy.fft import rfft, irfft
import numba as nb

"""
>>>Remarks on the Standardevent and the Noise Power Spectrum:

If there is no standard event available for a detector, it can get constructed like this: The easiest form of a Standardevent is the point-wise mean of many pulses. For better standard events some cleaning algorithm can be employed, e.g. the first constructed Standardevent can be fit back to all event from which it is constructed (fit parameters: height, baseline offset). Then all events that give a high fit-error (i.e. all events that do not look like the standardevent) get thrown away and from all others a new standard event is constructed. This can be repeated several times and results very likely in a very beautiful pulse-shape.

The Noise Power Spectrum is as a mean of the noise power spectra of many baselines. Usually, about 2% of all hardware-taken baselines are not clear, but have artifacts like pulses, spikes, etc. This has to be cleaned first, i.e. by a low pass filter and then a treshhold, all baselines that exceed that treshhold for some sample get thrown away. The 98% clean baselines are then fourier transformed. The noise power spectrum (NPS) is the squared absolut value of the fourier transformation. Then take the mean of the NPS of all baselines.

>>>Remark on the Optimal Filter in General:

The Optimal Filter is the frequency filter which provides the highest possible signal-to-noise ratio for the time tau_M (this is usually chosen as the time of the peak of the standardevent), if: 
    -) the event is exactly shaped like the standard event
    -) the noise has the exact frequency spectrum of the nps we provide 
In applications those requirements will never be perfectly fulfilled, so we will never reach the actual upper limit of the signal-to-noise ratio. Still, it is the best we can do with a frequency filter.
"""


# ---------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------

def normalization_constant(stdevent, nps):
    """
    this function is needed as utility for the function optimal_transition_function and calculates the normalization constant s.t. the Variance of an event is preserved
    Remark: We do not use this function, as we want to preserve the height of a peak and not the RMS.

    :param stdevent:  1D array of the standardevent with length N
    :param nps: 1D array of the noise power spectrum with length N/2 + 1
    :return: integer, the normalization constant
    """

    stdevent_fft = rfft(stdevent)  # fft or the stdevent
    # this is the formula from the paper for the constant
    h = 1 / np.sum(np.abs(stdevent_fft) ** 2 / nps)

    return h


def optimal_transfer_function(stdevent, nps):
    """
    this function calculates the transition function for an optimal filter

    :param stdevent: 1D array, pulse shape standard event with length N
    :param nps: 1D array, the NPS of a baseline, with length N/2 + 1
    :return: 1D complex numpy array of length N/2 + 1, the optimal transfer function
    """

    tau_m = (np.argmax(stdevent)) / (1304 * len(
        stdevent) / 8192)  # index of maximal value, the number 1304 is experimentally evaluated on an event of length 8192
    stdevent_fft = rfft(stdevent)  # do fft of stdevent
    H = np.zeros([len(stdevent_fft)], dtype=complex)  # init transfer func

    for w in range(0, len(stdevent_fft)):
        # this is the actual calculation:
        H[w] = (stdevent_fft[w].conjugate()) * np.exp(-1j * w * tau_m) / nps[w]
        # H[w] = H[w] / normalization_constant(stdevent, nps)  # divide by constant to keep size!

    filtered_standardevent = filter_event(stdevent, H)

    return H / np.max(filtered_standardevent)


def filter_event(event, transfer_function):
    """
    this function filters a single event
    :param event: 1D array of the one event that should be filtered, size N
    :param transition_function: the filter in fourier space, size N/2 +1 complex numpy array
    :return: 1D array length N, the filtered event
    """

    event_fft = rfft(event)  # do fft of event
    # do the filtering, multiplication in fourier space is convolution in time space
    event_filtered = event_fft * transfer_function
    event_filtered = irfft(event_filtered)  # fft back to time space

    return event_filtered


def get_amplitudes(events_array, stdevent, nps, hard_restrict=False, down=1):
    """
    This function determines the amplitudes of several events with optimal sig-noise-ratio.

    :param events_array: 2D array (nmbr_events, rec_length), the events to determine ph
    :param stdevent: 1D array, the standardevent
    :param nps: 1D array, length N/2 + 1, the noise power spectrum
    :param hard_restrict: bool, The maximum search is restricted to 20-30% of the record window.
    :param down: int, a factor by which the events and filter is downsampled before application
    :return: 1D array size (nmbr_events), the phs after of filtering
    """

    length = len(events_array[0])
    events_array = events_array - np.mean(events_array[:, :int(length / 8), np.newaxis], axis=1)

    if down > 1:
        length = int(length/down)
        events_array = np.mean(events_array.reshape(len(events_array), length, down), axis=2)

    # calc transition function
    transition_function = optimal_transfer_function(stdevent, nps)
    # filter events
    events_filtered = np.array([filter_event(event, transition_function) for event in events_array])
    # get maximal heights of filtered events
    if not hard_restrict:
        amplitudes = np.max(events_filtered[:, int(length / 8):-int(length / 8)], axis=1)
    if hard_restrict:
        amplitudes = np.max(events_filtered[:, int(length * 20 / 100):int(length * 30 / 100)], axis=1)

    return amplitudes
