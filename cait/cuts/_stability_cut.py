# imports

import numpy as np

# functions

def controlpulse_stability(hours_ev, cphs=None, hours_cp=None, significance=3,
                           max_gap=1, lb=0, ub=100, instable_iv=None):
    """
    Find all stable control pulses and events.

    A control pulse is defined as stable, if its height is between the lower and upper bound, and within
    a significant amount of sigmas
    of the mean height of control pulses. Every time interval between two control pulses is defined as stable, if they
    are both stable. A single unstable control pulse is in this definition ignored, so only two consecutive instable
    control pulses lead to an instable time interval. All events in instable time intervals are counted as instable.

    This method is described in "CRESST Collaboration, First results from the CRESST-III low-mass dark matter program"
    (10.1103/PhysRevD.100.102002).

    :param cphs: The pulse heights of the control pulses.
    :type cphs: 1D float array
    :param hours_cp: The hours time stamps of the control pulses.
    :type hours_cp: 1D float array
    :param hours_ev: The hours time stamps of the events.
    :type hours_ev: 1D float array
    :param significance: The multiplcator of the standard deviation of the pulse heights sigma, that is used to exclude
        all pulses outside significance*sigma of the mean pulse height.
    :type significance: float
    :param max_gap: For control pulses that are further apart than this value, the interval in between is automatically
        counted as instable.
    :type max_gap: float
    :param lb: All control pulse heights below this value are counted as instable.
    :type lb: float
    :param ub: All control pulse heights above this value are counted as instable.
    :type ub: float
    :param instable_iv: A list of the instable intervals. If this is handed, the instable intervals are not calculated
        but those are used. Useful for e.g. the cut efficiency.
    :type instable_iv: list
    :return: The flag that specifies which events are stable. The flag that specifies which control pulses are stable.
        The list of instable intervals in hours.
    :rtype: two 1D bool arrays, one list
    """

    print('Do Controlpulse Stability Cut')

    if instable_iv is None:

        assert cphs is not None and hours_cp is not None, 'If you hand no instable_iv, you need to hand control pulse' \
                                                          'heights and hours!'

        cphs = np.array(cphs)
        hours_cp = np.array(hours_cp)
        hours_ev = np.array(hours_ev)

        nmbr_cps = len(cphs)

        cond = np.logical_and(cphs > lb, cphs < ub)

        # get all hours that are not within significance standard deviations
        mu = np.mean(cphs[cond])
        sigma = np.std(cphs[cond])

        cond = np.logical_and(cond, np.abs(cphs - mu) < significance * sigma)
        print('Control Pulse PH {:.3f} +- {:.3f}, within {} sigma: {:.3f} %'.format(mu, sigma, significance,
                                                                                    100 * np.sum(
                                                                                        cond) / len(
                                                                                        cond)))

        print('Good Control Pulses: {}/{} ({:.3f}%)'.format(np.sum(cond), nmbr_cps, 100 * np.sum(cond) / nmbr_cps))

        flag_cp = cond

        cond[0] = True  # such that we do not exceed boundaries below
        cond[-1] = True

        # make the exclusion intervals
        # both control pulses before and after must have condition true
        # also, gap between control pulses must not exceed the max_gap duration!

        instable_iv = []
        i = 1
        while i < len(cond):
            if not cond[i]:
                lb = hours_cp[i]
                while cond[i] == False:
                    i += 1
                ub = hours_cp[i]
                instable_iv.append([lb, ub])
            else:
                if hours_cp[i] - hours_cp[i - 1] > max_gap:
                    instable_iv.append([hours_cp[i - 1], hours_cp[i]])
            i += 1
    else:
        print('Using pre-calculated instable intervals.')
        flag_cp = None

    # simplify the intervals
    iv_simple = []
    start = instable_iv[0][0]
    for i, iv in enumerate(instable_iv[:-1]):
        if not iv[1] == instable_iv[i + 1][0]:
            iv_simple.append([start, iv[1]])
            start = instable_iv[i + 1][0]
    iv_simple.append([start, instable_iv[-1][1]])
    instable_iv = iv_simple

    # exclude the events in instable intervals
    flag_ev = np.ones(len(hours_ev), dtype=bool)
    excluded_hours = 0
    for lb, ub in instable_iv:
        flag_ev[np.logical_and((hours_ev > lb), (hours_ev < ub))] = False
        excluded_hours += ub - lb

    # exclude also the events before first and after last tp!
    if instable_iv is None:
        flag_ev[hours_ev < hours_cp[0]] = False
        flag_ev[hours_ev > hours_cp[-1]] = False
        if cond[1] == True:
            excluded_hours += hours_cp[0]
            excluded_hours += np.max([0, hours_ev[-1] - hours_cp[-1]])

    print('Good Events: {}/{} ({:.3f}%)'.format(np.sum(flag_ev), len(flag_ev), 100 * np.sum(flag_ev) / len(flag_ev)))
    if hours_cp is not None:
        print('Good Time: {:.3f}h/{:.3f}h ({:.3f}%)'.format(hours_cp[-1] - excluded_hours, hours_cp[-1],
                                                            100 * (hours_cp[-1] - excluded_hours) / hours_cp[-1]))

    return flag_ev, flag_cp, instable_iv


def testpulse_stability(tpas, tphs, hours_tp, hours_ev,
                        significance=3, noise_level=0.005, max_gap=1,
                        lb=None, ub=None):
    """
    Find all test pulses and events.

    The stability is evaluated for all TPA values independently. A test pulse is defined as stable, if its height is
    between the lower and upper bound, and within a significant amount of sigmas
    of the mean height of test pulses with this TPA value. Every time interval between two test pulses is
    defined as stable, if they
    are both stable. All events in instable time intervals are counted as instable.

    :param tpas: The TPA values of the test pulses.
    :type tpas: 1D float array
    :param tphs: The pulse heights of the test pulses.
    :type tphs: 1D float array
    :param hours_tp: The hours time stamps of the test pulses.
    :type hours_tp: 1D float array
    :param hours_ev: The hours time stamps of the events.
    :type hours_ev: 1D float array
    :param significance: The multiplcator of the standard deviation of the pulse heights sigma, that is used to exclude
        all pulses outside significance*sigma of the mean pulse height.
    :type significance: float
    :param noise_level: The
    :type noise_level: float
    :param max_gap: For control pulses that are further apart than this value, the interval in between is automatically
        counted as instable.
    :type max_gap: float
    :param lb: All control pulse heights below this value are counted as instable.
    :type lb: float
    :param ub: All control pulse heights above this value are counted as instable.
    :type ub: float
    :return: ( The flag that specifies which events are stable; The flag that specifies which test pulses are stable.)
    :rtype: 2-tuple of 1D bool arrays
    """

    print('Do Test Pulse Stability Cut')

    unique_tpas = np.unique(tpas)
    print('Unique TPAs: ', unique_tpas)

    if lb is not None:
        if len(lb) != len(unique_tpas):
            raise KeyError('The argument lb needs to be a list of same length as the unique tpas!')
    if ub is not None:
        if len(ub) != len(unique_tpas):
            raise KeyError('The argument ub needs to be a list of same length as the unique tpas!')

    tpas = np.array(tpas)
    tphs = np.array(tphs)
    hours_tp = np.array(hours_tp)
    hours_ev = np.array(hours_ev)

    nmbr_tps = len(tphs)

    # cleaning
    cond_noise = tphs > noise_level
    print('Testpulses after Noise Cut: {}/{} ({:.3f}%)'.format(np.sum(cond_noise), len(cond_noise),
                                                               100 * np.sum(cond_noise) / len(cond_noise)))

    tpas = tpas[cond_noise]
    tphs = tphs[cond_noise]
    cond = np.ones(len(tpas), dtype=bool)

    for i, val in enumerate(unique_tpas):
        # apply the bounds
        if lb is not None:
            cond[tpas == val] = np.logical_and(cond[tpas == val], tphs[tpas == val] > lb[i])
        if ub is not None:
            cond[tpas == val] = np.logical_and(cond[tpas == val], tphs[tpas == val] < ub[i])
        if lb is not None or ub is not None:
            print('TPA {:.3f} after lb/ub cut: {}/{}'.format(val, np.sum(cond[tpas == val]), len(cond[tpas == val])))

        # get all hours that are not within significance standard deviations
        mu = np.mean(tphs[np.logical_and(tpas == val, cond)])
        sigma = np.std(tphs[np.logical_and(tpas == val, cond)])

        cond[tpas == val] = np.logical_and(cond[tpas == val], np.abs(tphs[tpas == val] - mu) < significance * sigma)
        print('TPA {:.3f} with PH {:.3f} +- {:.3f}, within {} sigma: {:.3f} %'.format(val, mu, sigma, significance,
                                                                                      100 * np.sum(
                                                                                          cond[tpas == val]) / len(
                                                                                          cond[tpas == val])))

    print('Good Testpulses: {}/{} ({:.3f}%)'.format(np.sum(cond), nmbr_tps, 100 * np.sum(cond) / nmbr_tps))

    cond_noise[cond_noise] = cond
    flag_tp = cond_noise

    cond[0] = True  # such that we do not excees boundaries below
    cond[-1] = True

    # make the exclusion intervals
    # both testpulses before and after must have condition true
    # also, gap between test pulses must not exceed the max_gap duration!
    exclusion = []
    i = 1
    while i < len(cond):
        if not cond[i]:
            lb = hours_tp[i]
            while cond[i] == False:
                i += 1
            ub = hours_tp[i]
            exclusion.append([lb, ub])
        else:
            if hours_tp[i] - hours_tp[i - 1] > max_gap:
                exclusion.append([hours_tp[i - 1], hours_tp[i]])
        i += 1

    # exclude the events in instable intervals
    flag_ev = np.ones(len(hours_ev), dtype=bool)
    excluded_hours = 0
    for lb, ub in exclusion:
        flag_ev[np.logical_and((hours_ev > lb), (hours_ev < ub))] = False
        excluded_hours += ub - lb

    # exclude also the events before first and after last tp!
    flag_ev[hours_ev < hours_tp[0]] = False
    flag_ev[hours_ev > hours_tp[-1]] = False
    if cond[1] == True:
        excluded_hours += hours_tp[0]
        excluded_hours += np.max([0, hours_ev[-1] - hours_tp[-1]])

    print('Good Events: {}/{} ({:.3f}%)'.format(np.sum(flag_ev), len(flag_ev), 100 * np.sum(flag_ev) / len(flag_ev)))
    print('Good Time: {:.3f}h/{:.3f}h ({:.3f}%)'.format(hours_tp[-1] - excluded_hours, hours_tp[-1],
                                                        100 * (hours_tp[-1] - excluded_hours) / hours_tp[-1]))

    return flag_ev, flag_tp
