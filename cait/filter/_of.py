# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------

import numpy as np
from numpy.fft import rfft, irfft

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
    this function is needed as utility for the function optimal_transition_function and calculates the normalization constant s.t. the amplitude of a peak is preserved
    stdevent is the array of the standardevent with length N
    nps is the array of the noise power spectrum with length N/2 i guess
    """

    stdevent_fft = rfft(stdevent)  # fft or the stdevent
    # this is the formula from the paper for the constant
    h = 1 / np.sum(np.abs(stdevent_fft) ** 2 / nps)

    return h


def optimal_transfer_function(stdevent, nps):
    """
    this function calculates the transition function for an optimal filter
    stdevent is the pulse shape standard event array with length N
    nps is the NPS of a baseline, with length N/2 i guess
    """

    tau_m = (np.argmax(stdevent) * 0.04) / (52.16 * len(stdevent) / 8192)  # index of maximal value
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
    event is the array of the one event that should be filtered, size N
    transition_function is the filter in fourier space, size N/2 i guess
    """

    event_fft = rfft(event)  # do fft of event
    # do the filtering, multiplication in fourier space is convolution in time space
    event_filtered = event_fft * transfer_function
    event_filtered = irfft(event_filtered)  # fft back to time space

    return event_filtered


def get_amplitudes(events_array, stdevent, nps):
    """
    this function determines the amplitudes of several events with optimal sig-noise-ratio
    events_array is an MxN array of M events each length N
    stdevent is the standardevent, an array of length N
    nps is the noise power spectrum, an array of length N/2 i guess
    """

    # calc transition function
    transition_function = optimal_transfer_function(stdevent, nps)
    # filter events
    events_filtered = np.array([filter_event(event, transition_function) for event in events_array])
    # get maximal heights of filtered events
    amplitudes = np.max(events_filtered, axis=0)

    return amplitudes
