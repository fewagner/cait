import numpy as np

def extend_nps(nps, extend_to, sampling_frequency):
    """
    Extend a noise power spectrum to a given length.

    :param nps: The noise power spectrum.
    :type nps: 1D np.array
    :param extend_to: The desired length of the noise record window.
    :type extend_to: int
    :param sampling_frequency: The sampling frequency.
    :type sampling_frequency: int
    :return: List of the frequencies of the extended NPS and the extended NPS.
    :rtype: list
    """
    rw = 2*int(nps.shape[0] - 1)
    extended_nps = np.sqrt(nps)
    extended_nps = np.fft.irfft(extended_nps)
    extended_nps = np.roll(extended_nps, shift=int(rw/2))
    extended_nps = np.pad(extended_nps, int(extend_to/2 - rw/2), mode='edge')
    assert extended_nps.shape[0] == extend_to
    extended_nps = np.roll(extended_nps, int(extend_to/2))
    extended_nps = np.fft.rfft(extended_nps)
    extended_nps = np.abs(extended_nps)**2
    extended_nps[0] = 0
    return np.fft.rfftfreq(extend_to, 1/sampling_frequency), extended_nps

def extend_filter(filt, extend_to, sampling_frequency):
    """
    Extend a filter to a given length.

    :param nps: The filter.
    :type nps: 1D np.array
    :param extend_to: The desired length of the filter.
    :type extend_to: int
    :param sampling_frequency: The sampling frequency.
    :type sampling_frequency: int
    :return: List of the frequencies of the extended filter and the extended filter.
    :rtype: list
    """
    rw = 2*int(filt.shape[0] - 1)
    extended_filt = np.fft.irfft(filt)
    extended_filt = np.roll(extended_filt, shift=int(rw/2))
    extended_filt = np.pad(extended_filt, int(extend_to/2 - rw/2), mode='edge')
    assert extended_filt.shape[0] == extend_to
    extended_filt = np.roll(extended_filt, int(extend_to/2))
    extended_filt = np.fft.rfft(extended_filt)
    extended_filt[0] = 0
    return np.fft.rfftfreq(extend_to, 1/sampling_frequency), extended_filt
