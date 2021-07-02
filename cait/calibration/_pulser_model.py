# imports

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.stats import linregress, t, norm
from scipy import odr
from scipy.interpolate import interp1d
from ..styles import make_grid, use_cait_style
from sklearn.ensemble import GradientBoostingRegressor


# functions

def light_yield_correction(phonon_energy, light_energy, scintillation_efficiency):
    """
    Return the recoil energy, corrected for the energy loss to scintillation light.

    TODO add citation

    :param phonon_energy: The recoil energies of the phonon channel, in keV.
    :type phonon_energy: 1D float array
    :param light_energy: The recoil energies of the light channel, in keV electron equivalent.
    :type light_energy: 1D float array
    :param scintillation_efficiency: The scintillation efficiency of the crystal.
    :type scintillation_efficiency: float
    :return: The corrected recoil energies.
    :rtype: 1D float array
    """
    return scintillation_efficiency * light_energy + (1 - scintillation_efficiency) * phonon_energy


# classes

class PolyModel:
    """
    Fit a polynomial to given x and y data an orthogonal distance regression and uncertainties.

    :param xd: The x values for the fit.
    :type xd: 1D array
    :param yd: The y values for the fit.
    :type yd: 1D array
    :param x_sigma: The uncertainties on the x values for the fit.
    :type x_sigma: 1D array
    :param y_sigma: The uncertainties on the y values for the fit.
    :type y_sigma: 1D array
    :param order: The order of the polynomial.
    :type order: int
    """

    def __init__(self, xd, yd, x_sigma=None, y_sigma=None, order=5):
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
        """
        Evaluate the fitted polynomial for a given x value and return the +/- 1 sigma prediction interval.

        :param x: The x values for which we evaluate the fitted polynomial.
        :type x: 1D array
        :return: (predicted values - 1 sigma uncertainty, predicted value, predicted values + 1 sigma uncertainty)
        :rtype: 3-tuple of 1D arrays
        """
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
    """
    Fit a linear model to given x and y data with ordinary least squares regression and prediction intervals.

    :param xd: The x values for the fit.
    :type xd: 1D array
    :param yd: The y values for the fit.
    :type yd: 1D array
    """

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
        """
        Evaluate the fitted linear model for a given x value.

        :param x: The x values for which we evaluate the fitted linear model.
        :type x: 1D array
        :return: The predicted values.
        :rtype: 1D array
        """
        return x * self.a + self.b

    def confidence_interval(self, x, conf=0.95):
        """
        Return predicted values and their confidence interval for given x data.

        :param x: The x values for which we evaluate the fitted polynomial.
        :type x: 1D array
        :param conf: The confidence level of the confidence interval.
        :type conf: float between 0 and 1
        :return: (predicted values - confidence, predicted value, predicted values + confidence)
        :rtype: 3-tuple of 1D arrays
        """
        alpha = 1. - conf

        y = self.y(x)
        sx = (x - self.xd.mean()) ** 2

        q = t.ppf(1. - alpha / 2, self.n - 2)  # quantile
        dy = q * self.sd * np.sqrt(1. / self.n + sx / self.sxd)
        yl = y - dy
        yu = y + dy

        return yl, y, yu

    def y_sigma(self, x):
        """
        Evaluate the fitted linear model for a given x value and return the +/- 1 sigma accumulated uncertainties (confidence and prediction uncertainty).

        :param x:  The x values for which we evaluate the fitted polynomial.
        :type x: 1D array
        :return: (predicted values - 1 sigma prediction interval, predicted value, predicted values + 1 sigma prediction interval)
        :rtype: 3-tuple of 1D arrays
        """
        yl, y, yu = self.confidence_interval(x, conf=0.68)

        lower, upper = y - np.sqrt((y - yl) ** 2 + self.stdev ** 2), y + np.sqrt((y - yu) ** 2 + self.stdev ** 2)
        return lower, y, upper

    def prediction_interval(self, x, pi=.68):
        """
        Evaluate the fitted polynomial for a given x value and return the prediction interval.

        :param x: The x values for which we evaluate the fitted polynomial.
        :type x: 1D array
        :param pi: The
        :type pi: float between 0 and 1
        :return: (predicted values - prediction interval, predicted value, predicted values + prediction interval)
        :rtype: 3-tuple of 1D arrays
        """
        y = self.y(x)

        one_minus_pi = 1 - pi
        ppf_lookup = 1 - (one_minus_pi / 2)
        z_score = norm.ppf(ppf_lookup)
        interval = z_score * self.stdev  # generate prediction interval lower and upper bound
        lower, upper = y - interval, y + interval
        return lower, y, upper


# pulse model class

class PulserModel:
    """
    A model to unfold the detector effects of cryogenic detectos.

    The model is to predict the equivalent test pulse values for particle event recoils within the detector crystal.

    # todo add citation

    :param start_saturation: Test pulses with average pulse heights above this value are excluded from the fit.
    :type start_saturation: float
    :param max_dist: The maximal distance between two test pulses such that they are included still in the same regression.
        For test pulses further apart, a new regression is started.
    :type max_dist: float
    """

    def __init__(self,
                 start_saturation,
                 max_dist,
                 ):
        self.start_saturation = start_saturation
        self.max_dist = max_dist
        print('PulserModel instance created.')

    def fit(self,
            tphs,
            tpas,
            tp_hours,
            exclude_tpas=[],
            interpolation_method='linear',
            LOWER_ALPHA=0.16,
            MIDDLE_ALPHA=0.5,
            UPPER_ALPHA=0.84,
            ):
        """
        Fit the model to given test pulse heights and amplitudes.

        This fits a linear model or a gradient boosted regression tree for each test pulse amplitude in the
        pulse height - time plane.

        :param tphs: The pulse heights of the test pulses.
        :type tphs: 1D array
        :param tpas: The test pulse amplitudes.
        :type tpas: 1D array
        :param tp_hours: The hours time stamps of the test pulses.
        :type tp_hours: 1D array
        :param exclude_tpas: A list of TPA values that are excluded from the fit.
        :type exclude_tpas: list
        :param interpolation_method: Either 'linear' or 'tree'. Either a linear model or a regression tree is used for
            the interpolation of test pulse pulse height.
        :type interpolation_method: string
        :param LOWER_ALPHA: For the regression tree, this gives the quantile for the upper prediction band.
        :type LOWER_ALPHA: float
        :param MIDDLE_ALPHA: For the regression tree, this gives the quantile for the central prediction band.
        :type MIDDLE_ALPHA: float
        :param UPPER_ALPHA: For the regression tree, this gives the quantile for the lower prediction band.
        :type UPPER_ALPHA: float
        """

        self.tphs = tphs
        self.tp_hours = tp_hours

        unique_tpas = np.unique(tpas)
        unique_tpas = unique_tpas[np.logical_not(np.in1d(unique_tpas, exclude_tpas))]
        print('Unique TPAs: ', unique_tpas)

        if interpolation_method == 'linear':

            # create the interpolation and exclusion intervals
            self.intervals = []
            lb = tp_hours[0]
            for i in range(1, len(tp_hours)):
                if np.abs(tp_hours[i] - tp_hours[i - 1]) > self.max_dist:
                    self.intervals.append([lb, tp_hours[i]])
                    lb = tp_hours[i]
            self.intervals.append([lb, tp_hours[-1]])

            print('Intervals seperated by {} h: {}'.format(self.max_dist, self.intervals))

            # do a 1D linear reg for every TPA and exclusion interval to get continuous estimates of the PH/TPA factor
            self.all_regs = []
            self.all_linear_tpas = []
            for iv in self.intervals:
                regs = []
                linear_tpas = []
                for i in unique_tpas:
                    cond = tpas == i
                    this_tp_hours = tp_hours[cond]
                    this_tph = tphs[cond]

                    if np.mean(this_tph) < self.start_saturation:
                        linear_tpas.append(i)
                        regs.append(LinearModel(this_tp_hours, this_tph))
                    else:
                        break

                self.all_linear_tpas.append(linear_tpas)
                self.all_regs.append(
                    regs)  # all_regs[n][m] gives now the lin reg parameters in the n'th interval for the m'th TPA

        elif interpolation_method == 'tree':

            # regression tree for all TPAs and upper lower prediction intervals
            self.upper_regs = []
            self.mean_regs = []
            self.lower_regs = []
            self.linear_tpas = []
            for tpa in unique_tpas:
                if np.mean(tphs[tpas == tpa]) < self.start_saturation:
                    self.linear_tpas.append(tpa)

                    self.lower_regs.append(GradientBoostingRegressor(loss="quantile",
                                                                     alpha=LOWER_ALPHA))
                    self.mean_regs.append(GradientBoostingRegressor(loss="quantile",
                                                                    alpha=MIDDLE_ALPHA))
                    self.upper_regs.append(GradientBoostingRegressor(loss="quantile",
                                                                     alpha=UPPER_ALPHA))

                    # Fit models
                    self.lower_regs[-1].fit(tp_hours[tpas == tpa].reshape(-1, 1), tphs[tpas == tpa])
                    self.mean_regs[-1].fit(tp_hours[tpas == tpa].reshape(-1, 1), tphs[tpas == tpa])
                    self.upper_regs[-1].fit(tp_hours[tpas == tpa].reshape(-1, 1), tphs[tpas == tpa])

            self.linear_tpas = np.array(self.linear_tpas)

        else:
            raise NotImplementedError('This method is not implemented.')

    def predict(self,
                evhs,
                ev_hours,
                poly_order,
                cpe_factor=None,
                interpolation_method='linear',
                ):
        """
        Predict the equivalent test pulse value for given pulse heights.

        :param evhs: The pulse heights of the events to predict equivalent TPA values for.
        :type evhs: 1D array
        :param ev_hours: The hours time stamps of the events to predict equivalent TPA values for.
        :type ev_hours: 1D array
        :param poly_order: The order of the polynomial to fit in the TPA/PH plane.
        :type poly_order: int
        :param cpe_factor: The CPE factor, which linearly maps TPA values to recoil energies.
        :type cpe_factor: float
        :param interpolation_method: Either 'linear' or 'tree'. Either a linear model or a regression tree is used for
            the interpolation of test pulse pulse height.
        :type interpolation_method: string
        :return: (the recoil energies, the 1 sigma uncertainties on the recoil energies, the equivalent tpa values, the
            2 sigma uncertainties on the equivalent tpa values)
        :rtype: 4-tuple of 1D arrays
        """

        # for each event in the interpolation intervals define a ph/tpa factor, for other events take closest TP time value
        # this is a polynomial fit (order 5 or so) of the the ph/tpa values for fixed tpas

        if interpolation_method == 'linear':

            tpa_equivalent = np.zeros(len(evhs))
            tpa_equivalent_sigma = np.zeros(len(evhs))
            for e in tqdm(range(len(evhs))):
                for i, iv in enumerate(self.intervals):
                    if (i == 0 and ev_hours[e] <= iv[0]) or (  # events befor the first interval
                            ev_hours[e] > iv[0] and ev_hours[e] <= iv[1]) or (  # events inside an interval
                            i == len(self.intervals) - 1 and ev_hours[e] > iv[1]):  # events after the last interval
                        x_data, x_sigma = [], []
                        for s in self.all_regs[i]:
                            xl, x, xu = s.y_sigma(ev_hours[e])
                            x_data.append(x)
                            x_sigma.append(xu - xl)
                        y_data = self.all_linear_tpas[i]
                        model = PolyModel(xd=x_data, yd=y_data, x_sigma=x_sigma, order=poly_order)
                        yl, y, yu = model.y_pred(evhs[e])

                        tpa_equivalent[e] = y
                        tpa_equivalent_sigma[e] = yu - yl
                        break

        elif interpolation_method == 'tree':

            tpa_equivalent = np.zeros(len(evhs))
            tpa_equivalent_sigma = np.zeros(len(evhs))
            for e in tqdm(range(len(evhs))):
                x_data, x_sigma = [], []
                for l, m, u in zip(self.lower_regs, self.mean_regs, self.upper_regs):
                    xl, x, xu = l.predict(ev_hours[e].reshape(-1, 1)), m.predict(ev_hours[e].reshape(-1, 1)), u.predict(
                        ev_hours[e].reshape(-1, 1))
                    x_data.append(x)
                    x_sigma.append(xu - xl)
                x_data = np.array(x_data).reshape(-1)
                x_sigma = np.array(x_sigma).reshape(-1)
                y_data = self.linear_tpas
                model = PolyModel(xd=x_data, yd=y_data, x_sigma=x_sigma, order=poly_order)
                yl, y, yu = model.y_pred(evhs[e])

                tpa_equivalent[e] = y
                tpa_equivalent_sigma[e] = yu - yl

        else:
            raise NotImplementedError('This method is not implemented.')

        if cpe_factor is not None:
            energies = cpe_factor * tpa_equivalent
            energies_sigma = cpe_factor * tpa_equivalent_sigma
        else:
            energies = tpa_equivalent
            energies_sigma = tpa_equivalent_sigma

        return energies, energies_sigma, tpa_equivalent, tpa_equivalent_sigma

    def plot(self,
             dpi=None,
             plot_only_first_poly=True,
             plot_poly_timestamp=None,
             interpolation_method='linear',
             poly_order=3,
             ylim=None,
             xlim=None,
             ):
        """
        Plot a scatter plot of the test pulse pulse heights vs time and the fitted polynomial in the TPA/PH plane.

        :param dpi: The dots per inch of the plots.
        :type dpi: int
        :param plot_only_first_poly: If true, only the polynomial from the middle of the first consecutive record
            interval is plotted, otherwise all are plotted.
        :type plot_only_first_poly: bool
        :param plot_poly_timestamp: Time stamp at which the polynomial is plotted.
        :type plot_poly_timestamp: float
        :param interpolation_method: Either 'linear' or 'tree'. Either a linear model or a regression tree is used for
            the interpolation of test pulse pulse height.
        :type interpolation_method: string
        :param poly_order: The order of the polynomial to fit in the TPA/PH plane.
        :type poly_order: int
        :param ylim: The limits on the y axis.
        :type ylim: 2-tuple
        :param xlim: The limits on the x axis.
        :type xlim: 2-tuple
        """

        if interpolation_method == 'linear':
            use_cait_style(dpi=dpi)
            # plot the regressions
            plt.close()
            plt.scatter(self.tp_hours, self.tphs, s=5, marker='.', color='blue', zorder=10)
            for i, iv in enumerate(self.intervals):
                if i == 0:
                    plt.axvline(iv[0], color='green', linewidth=1, zorder=15)
                t = np.linspace(iv[0], iv[1], 100)
                for m in range(len(self.all_linear_tpas[i])):
                    lower, y, upper = self.all_regs[i][m].y_sigma(t)
                    plt.plot(t, y, color='red', linewidth=2, zorder=15)
                    plt.fill_between(t, lower, upper, color='black', alpha=0.3, zorder=5)
                    plt.axvline(iv[1], color='green', linewidth=1, zorder=15)
            make_grid()
            if ylim is None:
                plt.ylim([0, self.start_saturation])
            else:
                plt.ylim(ylim)
            plt.xlim(xlim)
            plt.xlabel('Hours (h)')
            plt.ylabel('Pulse Height (V)')
            plt.show()

            # plot the polynomials
            for i, iv in enumerate(self.intervals):
                plt.close()
                x_data, x_sigma = [], []
                if plot_poly_timestamp is None:
                    plot_timestamp = (iv[1] - iv[0]) / 2
                else:
                    plot_timestamp = np.copy(plot_poly_timestamp)
                print('Plot Regression Polynomial at {:.3} hours.'.format(plot_timestamp))
                for s in self.all_regs[i]:
                    xl, x, xu = s.y_sigma(plot_timestamp)
                    x_data.append(x)
                    x_sigma.append(xu - xl)
                y_data = self.all_linear_tpas[i]
                model = PolyModel(xd=x_data, yd=y_data, x_sigma=x_sigma, order=poly_order)

                h = np.linspace(0, self.start_saturation, 100)
                yl, y, yu = model.y_pred(h)

                plt.plot(h, y, color='red', linewidth=2, zorder=15)
                plt.fill_between(h, yl, yu, color='black', alpha=0.3, zorder=5)
                plt.plot(x_data, y_data, 'b.', markersize=3.5, zorder=10)
                plt.errorbar(x_data, y_data, ecolor='b', xerr=x_sigma, fmt=" ", linewidth=1, capsize=0, zorder=20)
                make_grid()
                if ylim is None:
                    plt.ylim([0, y[-1]])
                else:
                    plt.xlim(ylim)
                plt.ylabel('Testpulse Amplitude (V)')
                plt.xlabel('Pulse Height (V)')
                plt.show()

                if plot_only_first_poly or plot_poly_timestamp is not None:
                    break

        elif interpolation_method == 'tree':

            use_cait_style(dpi=dpi)
            # plot the regressions
            plt.close()
            plt.scatter(self.tp_hours, self.tphs, s=5, marker='.', color='blue', zorder=10)
            t = np.linspace(0, self.tp_hours[-1], 100)
            for m in range(len(self.linear_tpas)):
                lower = self.lower_regs[m].predict(t.reshape(-1, 1))
                y = self.mean_regs[m].predict(t.reshape(-1, 1))
                upper = self.upper_regs[m].predict(t.reshape(-1, 1))
                plt.plot(t, y, color='red', linewidth=2, zorder=15)
                plt.fill_between(t, lower, upper, color='black', alpha=0.3, zorder=5)
            make_grid()
            if ylim is None:
                plt.ylim([0, self.start_saturation])
            plt.xlim(xlim)
            plt.ylabel('Pulse Height (V)')
            plt.xlabel('Time (h)')
            plt.show()

            # plot the polynomials
            plt.close()
            x_data, x_sigma = [], []
            if plot_poly_timestamp is None:
                plot_timestamp = self.tp_hours[-1] / 2
            else:
                plot_timestamp = np.copy(plot_poly_timestamp)
            for l, m, u in zip(self.lower_regs, self.mean_regs, self.upper_regs):
                xl, x, xu = l.predict(np.array([[plot_timestamp]])), m.predict(np.array([[plot_timestamp]])), u.predict(
                    np.array([[plot_timestamp]]))
                x_data.append(x)
                x_sigma.append(xu - xl)
            x_data = np.array(x_data).reshape(-1)
            x_sigma = np.array(x_sigma).reshape(-1)
            y_data = self.linear_tpas
            model = PolyModel(xd=x_data, yd=y_data, x_sigma=x_sigma, order=poly_order)

            h = np.linspace(0, self.start_saturation, 100)
            yl, y, yu = model.y_pred(h)

            print('Plot Regression Polynomial at {:.3} hours.'.format(plot_timestamp))

            plt.plot(h, y, color='red', linewidth=2, zorder=15)
            plt.fill_between(h, yl, yu, color='black', alpha=0.3, zorder=5)
            plt.plot(x_data, y_data, 'b.', markersize=3.5, zorder=10)
            plt.errorbar(x_data, y_data, ecolor='b', xerr=x_sigma, fmt=" ", linewidth=0.5, capsize=0, zorder=20)
            make_grid()
            if ylim is None:
                plt.ylim([0, y[-1]])
            else:
                plt.xlim(ylim)
            plt.ylabel('Testpulse Amplitude (V)')
            plt.xlabel('Pulse Height (V)')
            plt.show()

        else:
            raise NotImplementedError('This method is not implemented.')
