# imports

import numpy as np

# functions

def rate_cut(timestamps, interval=10, significance=3, min=0, max=60):
    """
    Return a bool array for all timestamps, with true for events that are in intervals with sigma rate

    :param timestamps: the array of the time stamps in minutes
    :type timestamps: 1D array
    :param interval: in minutes, the interval we compare
    :type interval: float
    :param significance: all intervals, that have an event rate not within significance-sigma of mean event rate, are cut
    :type significance: float
    :param min: intervals with lower rate than this are excluded from the calculation
    :type min: float
    :param max: intervals with higher rate than this are excluded from the calculation
    :param max: float
    :return: true if event survives rate cut, false if not
    :rtype: boolean array of same size as timestamps
    """

    bins = np.arange(0, timestamps[-1], interval)
    hist, _ = np.histogram(timestamps, bins=bins)

    intervals = np.empty(shape=(bins-1, 2), dtype=float)

    hist_cut = hist[np.logical_and(hist >= min, hist <= max)]
    mean = np.mean(hist_cut)
    sigma = np.std(hist_cut)
    intervals[:, 0] = bins[:-1]
    intervals[:, 1] = bins[1:]
    intervals = intervals[np.logical_and(hist >= min, hist <= max), :]
    intervals = intervals[np.logical(hist_cut > mean - significance*sigma, hist_cut < mean + significance*sigma), :]

    flag = np.zeros(timestamps)
    for iv in intervals:
        flag[np.logical_and(timestamps >= iv[0],timestamps <= iv[0])] = 1

    return flag
