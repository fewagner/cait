# imports

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ..styles._plt_styles import use_cait_style, make_grid
from tqdm.auto import tqdm
from ._pulser_model import LinearModel, PolyModel
import warnings

# functions

def energy_calibration_linear(evhs,
                              ev_hours,
                              tphs,
                              tpas,
                              tp_hours,
                              start_saturation,  # in V Pulse Height
                              max_dist,
                              cpe_factor,
                              exclude_tpas=[],
                              plot=False,
                              poly_order=5,
                              dpi=150,
                              plot_only_first_poly=True
                              ):
    """
    Attention! This function is depricated!
    """

    warnings.warn('Attention, this function is depricated!')

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

    # do a 1D linear reg for every TPA and exclusion interval to get continuous estimates of the PH/TPA factor
    all_regs = []
    all_linear_tpas = []
    for iv in intervals:
        regs = []
        linear_tpas = []
        for i in unique_tpas:
            cond = tpas == i
            this_tp_hours = tp_hours[cond]
            this_tph = tphs[cond]

            if np.mean(this_tph) < start_saturation:
                linear_tpas.append(i)
                regs.append(LinearModel(this_tp_hours, this_tph))
            else:
                break

        all_linear_tpas.append(linear_tpas)
        all_regs.append(regs)  # all_regs[n][m] gives now the lin reg parameters in the n'th interval for the m'th TPA

    if plot:
        use_cait_style(dpi=dpi)
        # plot the regressions
        plt.close()
        plt.scatter(tp_hours, tphs, s=5, marker='.', color='blue', zorder=10)
        for i, iv in enumerate(intervals):
            if i == 0:
                plt.axvline(iv[0], color='green', linewidth=1, zorder=15)
            t = np.linspace(iv[0], iv[1], 100)
            for m in range(len(all_linear_tpas[i])):
                lower, y, upper = all_regs[i][m].y_sigma(t)
                plt.plot(t, y, color='red', linewidth=2, zorder=15)
                plt.fill_between(t, lower, upper, color='black', alpha=0.3, zorder=5)
                plt.axvline(iv[1], color='green', linewidth=1, zorder=15)
        make_grid()
        plt.ylim([0, start_saturation])
        plt.xlabel('Hours (h)')
        plt.ylabel('Pulse Height (V)')
        plt.show()

        # plot the polynomials
        for i, iv in enumerate(intervals):
            plt.close()
            x_data, x_sigma = [], []
            plot_timestamp = (iv[1] - iv[0]) / 2
            print('Plot Regression Polynomial at {:.3} hours.'.format(plot_timestamp))
            for s in all_regs[i]:
                xl, x, xu = s.y_sigma(plot_timestamp)
                x_data.append(x)
                x_sigma.append(xu - xl)
            y_data = all_linear_tpas[i]
            model = PolyModel(xd=x_data, yd=y_data, x_sigma=x_sigma, order=poly_order)

            h = np.linspace(0, start_saturation, 100)
            yl, y, yu = model.y_pred(h)

            plt.plot(h, y, color='red', linewidth=2, zorder=15)
            plt.fill_between(h, yl, yu, color='black', alpha=0.3, zorder=5)
            plt.plot(x_data, y_data, 'b.', markersize=3.5, zorder=10)
            plt.errorbar(x_data, y_data, ecolor='b', xerr=x_sigma, fmt=" ", linewidth=1, capsize=0, zorder=20)
            make_grid()
            plt.ylim([0, y[-1]])
            plt.ylabel('Testpulse Amplitude (V)')
            plt.xlabel('Pulse Height (V)')
            plt.show()

            if plot_only_first_poly:
                break

    # for each event in the interpolation intervals define a ph/tpa factor, for other events take closest TP time value
    # this is a polynomial fit (order 5 or so) of the the ph/tpa values for fixed tpas

    energies = np.zeros(len(evhs))
    energies_sigma = np.zeros(len(evhs))
    tpa_equivalent = np.zeros(len(evhs))
    tpa_equivalent_sigma = np.zeros(len(evhs))
    for e in tqdm(range(len(evhs))):
        for i, iv in enumerate(intervals):
            if (i == 0 and ev_hours[e] <= iv[0]) or (  # events befor the first interval
                    ev_hours[e] > iv[0] and ev_hours[e] <= iv[1]) or (  # events inside an interval
                    i == len(intervals) - 1 and ev_hours[e] > iv[1]):  # events after the last interval
                x_data, x_sigma = [], []
                for s in all_regs[i]:
                    xl, x, xu = s.y_sigma(ev_hours[e])
                    x_data.append(x)
                    x_sigma.append(xu - xl)
                y_data = all_linear_tpas[i]
                model = PolyModel(xd=x_data, yd=y_data, x_sigma=x_sigma, order=poly_order)
                yl, y, yu = model.y_pred(evhs[e])

                energies[e] = cpe_factor * y
                energies_sigma[e] = cpe_factor * (yu - yl)
                tpa_equivalent[e] = y
                tpa_equivalent_sigma[e] = yu - yl
                break

    return energies, energies_sigma, tpa_equivalent, tpa_equivalent_sigma
