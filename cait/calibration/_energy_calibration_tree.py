# imports

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import linregress, t, norm
from scipy import odr
from sklearn.ensemble import GradientBoostingRegressor
from scipy.interpolate import interp1d


# functions

class PolyModel:
    def __init__(self, xd, yd, x_sigma=None, y_sigma=None, order=5):
        # todo https://docs.scipy.org/doc/scipy/reference/odr.html !!
        self.xd = xd
        self.yd = yd
        self.x_sigma = x_sigma
        self.x_sigma_interp = interp1d(self.xd, self.x_sigma, fill_value="extrapolate")
        self.y_sigma = y_sigma
        self.order = order
        self.poly_model = odr.polynomial(order)
        self.fix = np.ones(order + 1)
        self.fix[0] = 0
        self.data = odr.RealData(self.xd, self.yd, sx=x_sigma, sy=y_sigma)
        self.odr = odr.ODR(data=self.data, model=self.poly_model)
        self.out = self.odr.run()
        self.y = np.poly1d(self.out.beta[::-1])
        self.dydx = np.polyder(self.y, m=1)

    def y_pred(self, x):
        y = self.y(x)
        if len(x.shape) == 0:
            x = np.array([x])
        x_mat = np.array([x ** (i) for i in range(self.order + 1)]).T

        # process covarianvce matrix and derivatives
        d = np.dot(np.dot(x_mat, self.out.cov_beta), np.transpose(x_mat))
        d = np.diag(d)

        lower, upper = y - np.sqrt(self.dydx(x) * self.x_sigma_interp(x)**2 + self.out.res_var * d), \
                       y + np.sqrt(self.dydx(x) * self.x_sigma_interp(x)**2 + self.out.res_var * d)
        return lower, y, upper


def energy_calibration_tree(evhs,
                            ev_hours,
                            tphs,
                            tpas,
                            tp_hours,
                            start_saturation,  # in V Pulse Height
                            cpe_factor,
                            exclude_tpas=[],
                            plot=False
                            ):
    """
    TODO

    :param evhs:
    :type evhs:
    :param ev_hours:
    :type ev_hours:
    :param tphs:
    :type tphs:
    :param tpas:
    :type tpas:
    :param tp_hours:
    :type tp_hours:
    :param linear_tpa_range:
    :type linear_tpa_range:
    :param max_dist:
    :type max_dist:
    :param cpe_factor:
    :type cpe_factor:
    :param smoothing_factor:
    :type smoothing_factor:
    :return:
    :rtype:
    """

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
        # plot the regressions
        plt.close()
        plt.scatter(tp_hours, tphs, s=1, marker='.', color='blue')
        t = np.linspace(0, tp_hours[-1], 100)
        for m in range(len(linear_tpas)):
            lower = lower_regs[m].predict(t.reshape(-1, 1))
            y = mean_regs[m].predict(t.reshape(-1, 1))
            upper = upper_regs[m].predict(t.reshape(-1, 1))
            plt.plot(t, y, color='red', linewidth=2, zorder=9)
            plt.fill_between(t, lower, upper, color='black', alpha=0.3)
        plt.show()

        # plot the polynomials
        plt.close()
        x_data, x_sigma = [], []
        for l, m, u in zip(lower_regs, mean_regs, upper_regs):
            xl, x, xu = l.predict(np.array([[tp_hours[-1] / 2]])), m.predict(np.array([[tp_hours[-1] / 2]])), u.predict(
                np.array([[tp_hours[-1] / 2]]))
            x_data.append(x)
            x_sigma.append(xu - xl)
        x_data = np.array(x_data).reshape(-1)
        x_sigma = np.array(x_sigma).reshape(-1)
        y_data = linear_tpas
        model = PolyModel(xd=x_data, yd=y_data, x_sigma=x_sigma)

        h = np.linspace(0, start_saturation, 100)
        yl, y, yu = model.y_pred(h)

        plt.plot(h, y, color='red', linewidth=2, zorder=1)
        plt.fill_between(h, yl, yu, color='black', alpha=0.3)
        plt.plot(x_data, y_data, 'b.', markersize=3.5)
        plt.errorbar(x_data, y_data, ecolor='b', xerr=x_sigma, fmt=" ", linewidth=0.5, capsize=0)
        plt.ylim([0, y[-1]])
        plt.ylabel('Testpulse Amplitude (V)')
        plt.xlabel('Pulse Height (V)')
        plt.show()

    # for each event in the interpolation intervals define a ph/tpa factor, for other events take closes TP time value
    # this is a polynomial fit (order 5 or so) of the the ph/tpa values for fixed tpas

    energies = np.zeros(len(evhs))
    energies_sigma = np.zeros(len(evhs))
    for e in range(len(evhs)):
        x_data, x_sigma = [], []
        for l, m, u in zip(lower_regs, mean_regs, upper_regs):
            xl, x, xu = l.predict(ev_hours[e].reshape(-1, 1)), m.predict(ev_hours[e].reshape(-1, 1)), u.predict(
                ev_hours[e].reshape(-1, 1))
            x_data.append(x)
            x_sigma.append(xu - xl)
        x_data = np.array(x_data).reshape(-1)
        x_sigma = np.array(x_sigma).reshape(-1)
        y_data = linear_tpas
        model = PolyModel(xd=x_data, yd=y_data, x_sigma=x_sigma)
        yl, y, yu = model.y_pred(evhs[e])

        energies[e] = cpe_factor * y
        energies_sigma[e] = cpe_factor * (yu - yl)

    return energies, energies_sigma
