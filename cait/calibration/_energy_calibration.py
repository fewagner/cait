# imports

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import linregress


# functions

def poly(x, a1, a2, a3, a4, a5):
    return a5 * x ** 5 + a4 * x ** 4 + a3 * x ** 3 + a2 * x ** 2 + a1 * x


def energy_calibration(evhs,
                       ev_hours,
                       tphs,
                       tpas,
                       tp_hours,
                       start_saturation,  # in V Pulse Height
                       max_dist,
                       cpe_factor,
                       smoothing_factor=0.95,
                       exclude_tpas=[],
                       plot=False
                       ):
    """
    Attention! This function is deprecated! Please use the PulserModel class instead!
    """

    raise Warning('This function is deprecated, please use the PulseModel instead!')

    unique_tpas = np.unique(tpas)
    unique_tpas = unique_tpas[np.logical_not(np.in1d(unique_tpas, exclude_tpas))]
    # unique_tpas = unique_tpas[np.logical_and(unique_tpas > linear_tpa_range[0],
    #                                          unique_tpas < linear_tpa_range[1])]
    print('Unique TPAs: ', unique_tpas)

    # create the interpolation and exclusion intervals
    intervals = []
    lb = tp_hours[0]
    for i in range(1, len(tp_hours)):
        if np.abs(tp_hours[i] - tp_hours[i - 1]) > max_dist:
            intervals.append([lb, tp_hours[i]])
            lb = tp_hours[i]
    intervals.append([lb, tp_hours[-1]])

    print('Intervals seperated by {} h: {}'.format(max_dist, intervals))

    # do a 1D spline fit for every TPA and exclusion interval to get continuous estimates of the PH/TPA factor
    all_spls = []
    all_linear_tpas = []
    for iv in intervals:
        spls = []
        linear_tpas = []
        for i in unique_tpas:
            cond = tpas == i
            this_tp_hours = tp_hours[cond]
            this_tph = tphs[cond]

            if np.mean(this_tph) < start_saturation:
                linear_tpas.append(i)
                spls.append(UnivariateSpline(this_tp_hours, this_tph))
                spls[-1].set_smoothing_factor(smoothing_factor)
            else:
                break

        all_linear_tpas.append(linear_tpas)
        all_spls.append(spls)  # all_spls[n][m] gives now the spline in the n'th interval for the m'th TPA

    if plot:
        # plot the splines
        plt.close()
        plt.scatter(tp_hours, tphs, s=1, marker='.', color='blue')
        for i, iv in enumerate(intervals):
            t = np.linspace(iv[0], iv[1], 100)
            for m in range(len(all_linear_tpas[i])):
                plt.plot(t, all_spls[i][m](t), color='orange', linewidth=2)
        plt.show()

        # plot the polynomials
        plt.close()
        for i, iv in enumerate(intervals):
            h = np.linspace(0, start_saturation, 100)
            xdata=[s((iv[1] - iv[0]) / 2) for s in all_spls[i]]
            ydata=all_linear_tpas[i]
            popt, pcov = curve_fit(poly,
                                   xdata=xdata,
                                   ydata=ydata)
            plt.plot(h, poly(h, *popt), color='orange', linewidth=2, zorder=1)
            plt.scatter(xdata, ydata, s=30, marker='x', color='blue', linewidths=2, zorder=10)
        plt.show()

    # for each event in the interpolation intervals define a ph/tpa factor, for other events take closes TP time value
    # this is a polynomial fit (order 5 or so) of the the ph/tpa values for fixed tpas

    energies = np.zeros(len(evhs))
    for e in range(len(evhs)):
        for i, iv in enumerate(intervals):
            if i == 0 and ev_hours[e] <= iv[0]:
                popt, pcov = curve_fit(poly,
                                       xdata=[s(ev_hours[e]) for s in all_spls[i]],
                                       ydata=all_linear_tpas[i])
                energies[e] = cpe_factor * poly(evhs[e], *popt)
                break
            elif ev_hours[e] > iv[0] and ev_hours[e] <= iv[1]:
                popt, pcov = curve_fit(poly,
                                       xdata=[s(ev_hours[e]) for s in all_spls[i]],
                                       ydata=all_linear_tpas[i])
                energies[e] = cpe_factor * poly(evhs[e], *popt)
                break

    return energies
