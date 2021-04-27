# imports

import numpy as np


# functions

def controlpulse_stability(cphs, hours_cp, hours_ev, significance=3, max_gap=1, lb=0, ub=100):
    """
    Return all event indices, that are between two stable control pulses

    TODO

    :param cphs:
    :type cphs:
    :param hours_cp:
    :type hours_cp:
    :param hours_ev:
    :type hours_ev:
    :param significance:
    :type significance:
    :param max_gap:
    :type max_gap:
    :param lb:
    :type lb:
    :param ub:
    :type ub:
    :return:
    :rtype:
    """

    print('Do Testpulse Stability Cut')

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
    exclusion = []
    i = 1
    while i < len(cond):
        if not cond[i]:
            lb = hours_cp[i]
            while cond[i] == False:
                i += 1
            ub = hours_cp[i]
            exclusion.append([lb, ub])
        else:
            if hours_cp[i] - hours_cp[i - 1] > max_gap:
                exclusion.append([hours_cp[i - 1], hours_cp[i]])
        i += 1

    # exclude the events in instable intervals
    flag_ev = np.ones(len(hours_ev), dtype=bool)
    excluded_hours = 0
    for lb, ub in exclusion:
        flag_ev[np.logical_and((hours_ev > lb), (hours_ev < ub))] = False
        excluded_hours += ub - lb

    # exclude also the events before first and after last tp!
    flag_ev[hours_ev < hours_cp[0]] = False
    flag_ev[hours_ev > hours_cp[-1]] = False
    if cond[1] == True:
        excluded_hours += hours_cp[0]
        excluded_hours += np.max([0, hours_ev[-1] - hours_cp[-1]])

    print('Good Events: {}/{} ({:.3f}%)'.format(np.sum(flag_ev), len(flag_ev), 100 * np.sum(flag_ev) / len(flag_ev)))
    print('Good Time: {:.3f}h/{:.3f}h ({:.3f}%)'.format(hours_cp[-1] - excluded_hours, hours_cp[-1],
                                                        100 * (hours_cp[-1] - excluded_hours) / hours_cp[-1]))

    return flag_ev, flag_cp


def testpulse_stability(tpas, tphs, hours_tp, hours_ev,
                        significance=3, noise_level=0.005, max_gap=1,
                        lb=None, ub=None):
    """
    Return all event indices, that are between two stable testpulses

    TODO

    :param tpas:
    :type tpas:
    :param tphs:
    :type tphs:
    :param hours_tp:
    :type hours_tp:
    :param hours_ev:
    :type hours_ev:
    :param significance:
    :type significance:
    :param noise_level:
    :type noise_level:
    :param max_gap:
    :type max_gap:
    :param lb:
    :type lb:
    :param ub:
    :type ub:
    :return:
    :rtype:
    """

    print('Do Control Pulse Stability Cut')

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
