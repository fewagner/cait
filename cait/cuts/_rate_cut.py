# imports

import numpy as np
import scipy.stats as sc


# functions

def rate_cut(timestamps,
             timestamps_cp=None,
             timestamps_tp=None,
             interval=10,
             significance=3,
             min=0, max=60,
             use_poisson=True,
             intervals=None):
    """
    Return a bool array for all timestamps, with true for events that are in intervals with sigma rate.

    :param timestamps: The array of the event time stamps in minutes.
    :type timestamps: 1D array
    :param timestamps_cp: The array of the control pulse time stamps in minutes.
    :type timestamps: 1D array
    :param timestamps_tp: The array of the test pulse time stamps in minutes.
    :type timestamps: 1D array
    :param interval: In minutes, the interval we compare.
    :type interval: float
    :param significance: All intervals, that have an event rate not within significance-sigma of mean event rate, are cut.
    :type significance: float
    :param min: Intervals with lower rate than this are excluded from the calculation.
    :type min: float
    :param max: Intervals with higher rate than this are excluded from the calculation.
    :param max: float
    :param use_poisson: If this is activated (per default) we use the median and poisson confidence intervals instead
        of standard normal statistics.
    :type use_poisson: bool
    :param intervals: A list of the stable intervals in minutes. If this is handed, these intervals are used instead of
        calculating them from scratch. This is useful e.g. for the cut efficiency.
    :type intervals: list of 2-tuples
    :return: For events, controlpulses and testpulses arrays: True if event survives rate cut, false if not. The list of
        the stable intervals in minutes.
    :rtype: 3 boolean arrays of same size as timestamps, list of tuples
    """
    print('Do Rate Cut.')

    if intervals is None:

        assert timestamps_cp is not None, 'If you hand no intervals, you need to hand timestamps of control pulses!'

        bins = np.arange(0, timestamps[-1], interval)
        hist, _ = np.histogram(timestamps, bins=bins)
        hist_cp, _ = np.histogram(timestamps_cp, bins=bins)  # this is to exclude gaps in the recording

        median_cp = np.median(hist_cp[hist_cp > 0])

        intervals = np.empty(shape=(len(bins) - 1, 2), dtype=float)
        intervals[:, 0] = bins[:-1]
        intervals[:, 1] = bins[1:]

        hist_cut = hist[np.logical_and(hist >= min, hist <= max)]
        hist_cp = hist_cp[np.logical_and(hist >= min, hist <= max)]
        hist_cut = hist_cut[hist_cp > median_cp/2]

        if not use_poisson:
            mean = np.mean(hist_cut)
            sigma = np.std(hist_cut)

            intervals = intervals[np.logical_and(hist >= min, hist <= max), :]
            intervals = intervals[hist_cp > median_cp / 2, :]
            intervals = intervals[
                        np.logical_and(hist_cut >= mean - significance * sigma, hist_cut <= mean + significance * sigma), :]

            print('Rate: ({:.3f} +- {:.3f})/{}m'.format(mean, sigma, interval))
            print('Good Rate per {}m ({} sigma): {:.3f} - {:.3f}'.format(interval,
                                                                         significance,
                                                                         np.max([mean - significance * sigma, 0]),
                                                                         mean + significance * sigma,
                                                                         ))

        else:
            median = np.median(hist_cut)
            low = sc.poisson.ppf(sc.norm.cdf(-significance), mu=median)
            up = sc.poisson.ppf(sc.norm.cdf(significance), mu=median)

            intervals = intervals[np.logical_and(hist >= min, hist <= max), :]
            intervals = intervals[hist_cp > median_cp / 2, :]
            intervals = intervals[
                        np.logical_and(hist_cut >= low, hist_cut <= up), :]

            print('Rate Median: {}/{}m'.format(median, interval))
            print('Good Rate per {}m ({} sigma): {} - {}'.format(interval,
                                                                         significance,
                                                                         low,
                                                                         up,
                                                                         ))

    else:
        print('Using precalculated intervals.')

    # simplify the intervals
    iv_simple = []
    start = intervals[0][0]
    for i, iv in enumerate(intervals[:-1]):
        if not iv[1] == intervals[i+1][0]:
            iv_simple.append([start, iv[1]])
            start = intervals[i+1][0]
    iv_simple.append([start, intervals[-1][1]])
    intervals = iv_simple

    flag_ev = np.zeros(len(timestamps), dtype=bool)
    if timestamps_cp is not None:
        flag_cp = np.zeros(len(timestamps_cp), dtype=bool)
    else:
        flag_cp = None
    if timestamps_tp is not None:
        flag_tp = np.zeros(len(timestamps_tp), dtype=bool)
    else:
        flag_tp = None
    for iv in intervals:
        flag_ev[np.logical_and(timestamps >= iv[0], timestamps <= iv[1])] = 1
        if timestamps_cp is not None:
            flag_cp[np.logical_and(timestamps_cp >= iv[0], timestamps_cp <= iv[1])] = 1
        if timestamps_tp is not None:
            flag_tp[np.logical_and(timestamps_tp >= iv[0], timestamps_tp <= iv[1])] = 1

    if intervals is None:
        print('Good Time: {:.3f}h/{:.3f}h ({:.3f}%)'.format(len(intervals) * interval / 60,
                                                            (len(bins) - 1) * interval / 60,
                                                            100 * len(intervals) / (len(bins) - 1)))
    print('Good Events: {:.3f}/{:.3f} ({:.3f}%)'.format(np.sum(flag_ev), len(flag_ev),
                                                        100 * np.sum(flag_ev) / len(flag_ev)))
    if timestamps_cp is not None:
        print('Good Controlpulses: {:.3f}/{:.3f} ({:.3f}%)'.format(np.sum(flag_cp), len(flag_cp),
                                                                   100 * np.sum(flag_cp) / len(flag_cp)))
    if timestamps_tp is not None:
        print('Good Testpulses: {:.3f}/{:.3f} ({:.3f}%)'.format(np.sum(flag_tp), len(flag_tp),
                                                                100 * np.sum(flag_tp) / len(flag_tp)))
    return flag_ev, flag_cp, flag_tp, intervals
