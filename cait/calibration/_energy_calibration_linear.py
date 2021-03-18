# imports

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import linregress, t, norm
from scipy import odr
from scipy.interpolate import interp1d
from ..styles._plt_styles import use_cait_style, make_grid
from tqdm.auto import tqdm


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

        lower, upper = y - np.sqrt(np.abs(self.dydx(x)) * self.x_sigma_interp(x) ** 2 + self.out.res_var * d), \
                       y + np.sqrt(np.abs(self.dydx(x)) * self.x_sigma_interp(x) ** 2 + self.out.res_var * d)
        return lower, y, upper


class LinearModel:
    def __init__(self, xd, yd):
        self.xd = xd
        self.yd = yd
        self.a, self.b, self.r, self.p, self.err = linregress(self.xd, self.yd)
        self.n = self.xd.size
        self.sd = np.sqrt(1. / (self.n - 2.) * np.sum((self.yd - self.a * self.xd - self.b) ** 2))
        self.sxd = np.sum((self.xd - self.xd.mean()) ** 2)
        self.sum_errs = np.sum((self.yd - self.y(self.xd)) ** 2)
        self.stdev = np.sqrt(1 / (len(self.yd) - 2) * self.sum_errs)

    def y(self, x):
        return x * self.a + self.b

    def confidence_interval(self, x, conf=0.95):
        alpha = 1. - conf

        y = self.y(x)
        sx = (x - self.xd.mean()) ** 2

        q = t.ppf(1. - alpha / 2, self.n - 2)  # quantile
        dy = q * self.sd * np.sqrt(1. / self.n + sx / self.sxd)
        yl = y - dy
        yu = y + dy

        return yl, y, yu

    def y_sigma(self, x):
        yl, y, yu = self.confidence_interval(x, conf=0.68)

        lower, upper = y - np.sqrt((y - yl) ** 2 + self.stdev ** 2), y + np.sqrt((y - yu) ** 2 + self.stdev ** 2)
        return lower, y, upper

    def prediction_interval(self, x, pi=.68):
        y = self.y(x)

        one_minus_pi = 1 - pi
        ppf_lookup = 1 - (one_minus_pi / 2)
        z_score = norm.ppf(ppf_lookup)
        interval = z_score * self.stdev  # generate prediction interval lower and upper bound
        lower, upper = y - interval, y + interval
        return lower, y, upper


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
                break

    return energies, energies_sigma
