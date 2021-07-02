# imports

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from ..styles._plt_styles import use_cait_style, make_grid
from tqdm.auto import tqdm
from ._energy_calibration_linear import PolyModel
import warnings


# functions

def energy_calibration_tree(evhs,
                            ev_hours,
                            tphs,
                            tpas,
                            tp_hours,
                            start_saturation,  # in V Pulse Height
                            cpe_factor,
                            exclude_tpas=[],
                            poly_order=5,
                            plot=False,
                            dpi=150,
                            ):
    """
    Attention! This function is depricated!
    """

    warnings.warn('Attention, this function is depricated!')

    unique_tpas = np.unique(tpas)
    unique_tpas = unique_tpas[np.logical_not(np.in1d(unique_tpas, exclude_tpas))]

    print('Unique TPAs: ', unique_tpas)

    # regression tree for all TPAs and upper lower prediction intervals
    LOWER_ALPHA = 0.16
    MIDDLE_ALPHA = 0.5
    UPPER_ALPHA = 0.84
    upper_regs = []
    mean_regs = []
    lower_regs = []
    linear_tpas = []
    for tpa in unique_tpas:
        if np.mean(tphs[tpas == tpa]) < start_saturation:
            linear_tpas.append(tpa)

            lower_regs.append(GradientBoostingRegressor(loss="quantile",
                                                        alpha=LOWER_ALPHA))
            mean_regs.append(GradientBoostingRegressor(loss="quantile",
                                                       alpha=MIDDLE_ALPHA))
            upper_regs.append(GradientBoostingRegressor(loss="quantile",
                                                        alpha=UPPER_ALPHA))

            # Fit models
            lower_regs[-1].fit(tp_hours[tpas == tpa].reshape(-1, 1), tphs[tpas == tpa])
            mean_regs[-1].fit(tp_hours[tpas == tpa].reshape(-1, 1), tphs[tpas == tpa])
            upper_regs[-1].fit(tp_hours[tpas == tpa].reshape(-1, 1), tphs[tpas == tpa])

    linear_tpas = np.array(linear_tpas)

    if plot:
        use_cait_style(dpi=dpi)
        # plot the regressions
        plt.close()
        plt.scatter(tp_hours, tphs, s=5, marker='.', color='blue', zorder=10)
        t = np.linspace(0, tp_hours[-1], 100)
        for m in range(len(linear_tpas)):
            lower = lower_regs[m].predict(t.reshape(-1, 1))
            y = mean_regs[m].predict(t.reshape(-1, 1))
            upper = upper_regs[m].predict(t.reshape(-1, 1))
            plt.plot(t, y, color='red', linewidth=2, zorder=15)
            plt.fill_between(t, lower, upper, color='black', alpha=0.3, zorder=5)
        make_grid()
        plt.ylim(0, start_saturation)
        plt.ylabel('Pulse Height (V)')
        plt.xlabel('Time (h)')
        plt.show()

        # plot the polynomials
        plt.close()
        x_data, x_sigma = [], []
        plot_timestamp = tp_hours[-1] / 2
        for l, m, u in zip(lower_regs, mean_regs, upper_regs):
            xl, x, xu = l.predict(np.array([[plot_timestamp]])), m.predict(np.array([[plot_timestamp]])), u.predict(
                np.array([[plot_timestamp]]))
            x_data.append(x)
            x_sigma.append(xu - xl)
        x_data = np.array(x_data).reshape(-1)
        x_sigma = np.array(x_sigma).reshape(-1)
        y_data = linear_tpas
        model = PolyModel(xd=x_data, yd=y_data, x_sigma=x_sigma, order=poly_order)

        h = np.linspace(0, start_saturation, 100)
        yl, y, yu = model.y_pred(h)

        print('Plot Regression Polynomial at {:.3} hours.'.format(plot_timestamp))

        plt.plot(h, y, color='red', linewidth=2, zorder=15)
        plt.fill_between(h, yl, yu, color='black', alpha=0.3, zorder=5)
        plt.plot(x_data, y_data, 'b.', markersize=3.5, zorder=10)
        plt.errorbar(x_data, y_data, ecolor='b', xerr=x_sigma, fmt=" ", linewidth=0.5, capsize=0, zorder=20)
        make_grid()
        plt.ylim([0, y[-1]])
        plt.ylabel('Testpulse Amplitude (V)')
        plt.xlabel('Pulse Height (V)')
        plt.show()

    # for each event in the interpolation intervals define a ph/tpa factor, for other events take closes TP time value
    # this is a polynomial fit (order 5 or so) of the the ph/tpa values for fixed tpas

    energies = np.zeros(len(evhs))
    energies_sigma = np.zeros(len(evhs))
    tpa_equivalent = np.zeros(len(evhs))
    tpa_equivalent_sigma = np.zeros(len(evhs))
    for e in tqdm(range(len(evhs))):
        x_data, x_sigma = [], []
        for l, m, u in zip(lower_regs, mean_regs, upper_regs):
            xl, x, xu = l.predict(ev_hours[e].reshape(-1, 1)), m.predict(ev_hours[e].reshape(-1, 1)), u.predict(
                ev_hours[e].reshape(-1, 1))
            x_data.append(x)
            x_sigma.append(xu - xl)
        x_data = np.array(x_data).reshape(-1)
        x_sigma = np.array(x_sigma).reshape(-1)
        y_data = linear_tpas
        model = PolyModel(xd=x_data, yd=y_data, x_sigma=x_sigma, order=poly_order)
        yl, y, yu = model.y_pred(evhs[e])

        energies[e] = cpe_factor * y
        energies_sigma[e] = cpe_factor * (yu - yl)
        tpa_equivalent[e] = y
        tpa_equivalent_sigma[e] = yu - yl

    return energies, energies_sigma, tpa_equivalent, tpa_equivalent_sigma
