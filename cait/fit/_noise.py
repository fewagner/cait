# imports

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize, curve_fit
from scipy.stats import norm, expon
import matplotlib.pyplot as plt
from ..styles import use_cait_style, make_grid
import numba as nb


# ------------------------------------------
# functions for binned least squares gaussian fit
# ------------------------------------------

def noise_trigger_template(x_max, d, sigma):
    """
    A template for purely Gaussian noise.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution.
    :type sigma: float
    :return: The template.
    :rtype: 1D array
    """
    P = (d / np.sqrt(2 * np.pi) / sigma)
    P *= np.exp(-(x_max / np.sqrt(2) / sigma) ** 2)
    P *= (1 / 2 + erf(x_max / np.sqrt(2) / sigma) / 2) ** (d - 1)
    return P


def wrapper(x_max, d, sigma):
    sigma_lower_bound = 0.0001
    if sigma < sigma_lower_bound:
        P = 1000 * np.abs(sigma_lower_bound - sigma) + 1000
    else:
        P = noise_trigger_template(x_max, d, sigma)

    return P


def get_noise_parameters_binned(counts,
                                bins,
                                ):
    """
    Return the least squares fit parameters to the purely Gaussian noise model. You need to calculate a histogram of
    the maxima of the empty baselines before already, e.g. with np.hist.

    :param counts: The counts within the bins.
    :type counts: 1D array
    :param bins: The bin edges. This array is one number longer than the counts array.
    :type bins: 1D array
    :return: The fitted parameters (d, sigma).
    :rtype: 2-tuple
    """

    x_data = bins[:-1] + (bins[1] - bins[0]) / 2
    ydata = counts

    pars, _ = curve_fit(f=wrapper,
                        xdata=x_data,
                        ydata=ydata,
                        check_finite=True,
                        )

    print('Fitted Noise Trigger Template Parameters: d {},  sigma {:.3} mV'.format(pars[0], pars[1]))

    return pars


# ------------------------------------------
# utils functions for unbinned fit
# ------------------------------------------

def gauss_noise(x_max, d, sigma):
    """
    A template for purely Gaussian noise. Opposite to noise_trigger_template, this function uses the scipy
    implementation of the norm distribution.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution
    :type sigma: float
    :return: The template.
    :rtype: 1D array
    """

    P = d * norm.pdf(x_max, loc=0, scale=sigma) * norm.cdf(x_max, loc=0, scale=sigma) ** (d - 1)
    P[P < 10e-8] = 10e-8
    return P


def pollution_exponential_noise(x_max, d, sigma, lamb):
    """
    A template for purely Gaussian noise with a pollution that follows an exponential distribution.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution
    :type sigma: float
    :param lamb: The rate parameter of the exponential distribution.
    :type lamb: float
    :return: The template.
    :rtype: 1D array
    """

    P = expon.pdf(x_max, scale=1 / lamb) * norm.cdf(x_max, loc=0, scale=sigma) ** (d - 1)
    P += expon.cdf(x_max, scale=1 / lamb) * (d - 1) * norm.pdf(x_max, loc=0, scale=sigma) * norm.cdf(x_max, loc=0,
                                                                                                     scale=sigma) ** (
                 d - 2)
    P[P < 10e-10] = 10e-10
    return P


def fraction_exponential_noise(x_max, d, sigma, lamb, w):
    """
    A template for a Gaussian-exponential noise mixture.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution.
    :type sigma: float
    :param lamb: The rate parameter of the exponential distribution.
    :type lamb: float
    :param w: The weight of the Gaussian fraction.
    :type w: float
    :return: The template.
    :rtype: 1D array
    """

    P = d * (w * norm.pdf(x_max, loc=0, scale=sigma) + (1 - w) * expon.pdf(x_max, scale=1 / lamb)) * \
        (w * norm.cdf(x_max, loc=0, scale=sigma) + (1 - w) * expon.cdf(x_max, scale=1 / lamb)) ** (d - 1)
    P[P < 10e-8] = 10e-8
    return P


def pollution_gauss_noise(x_max, d, sigma, mu, sigma_2):
    """
    A template for purely Gaussian noise with a pollution that follows another Gaussian distribution.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution
    :type sigma: float
    :param mu: The mean of the Gaussian pollution.
    :type mu: float
    :param sigma_2: The standard deviation of the Gaussian pollution.
    :type sigma_2: float
    :return: The template.
    :rtype: 1D array
    """

    P = norm.pdf(x_max, loc=mu, scale=sigma_2) * norm.cdf(x_max, loc=0, scale=sigma) ** (d - 1)
    P += (d - 1) * norm.cdf(x_max, loc=mu, scale=sigma_2) * norm.pdf(x_max, loc=0, scale=sigma) * norm.cdf(x_max, loc=0,
                                                                                                           scale=sigma) ** (
                 d - 2)
    P[P < 10e-8] = 10e-8
    return P


def fraction_gauss_noise(x_max, d, sigma, mu, sigma_2, w):
    """
    A template for a two component Gaussian noise mixture.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution.
    :type sigma: float
    :param mu: The mean of the second component.
    :type mu: float
    :param sigma: The standard deviation of the second component.
    :type sigma: float
    :param w: The weight of the first Gaussian fraction.
    :type w: float
    :return: The template.
    :rtype: 1D array
    """

    P = d * (w * norm.pdf(x_max, loc=0, scale=sigma) + (1 - w) * norm.pdf(x_max, loc=mu, scale=sigma_2)) * \
        (w * norm.cdf(x_max, loc=0, scale=sigma) + (1 - w) * norm.cdf(x_max, loc=mu, scale=sigma_2)) ** (d - 1)
    P[P < 10e-8] = 10e-8
    return P


# ------------------------------------------
# only gauss components
# ------------------------------------------

def pollution_exponential_noise_gauss_comp(x_max, d, sigma, lamb):
    """
    A partial template for purely Gaussian noise with a pollution that follows an exponential distribution. Only the
    Gaussian component is returned.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution
    :type sigma: float
    :param lamb: The rate parameter of the exponential distribution.
    :type lamb: float
    :return: The template of the Gaussian component.
    :rtype: 1D array
    """

    P = expon.cdf(x_max, scale=1 / lamb) * (d - 1) * norm.pdf(x_max, loc=0, scale=sigma) * norm.cdf(x_max, loc=0,
                                                                                                    scale=sigma) ** (
                d - 2)
    return P


def fraction_exponential_noise_gauss_comp(x_max, d, sigma, lamb, w):
    """
    A partial template for a Gaussian-exponential noise mixture. Only the
    Gaussian component is returned.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution.
    :type sigma: float
    :param lamb: The rate parameter of the exponential distribution.
    :type lamb: float
    :param w: The weight of the Gaussian fraction.
    :type w: float
    :return: The template of the Gaussian component.
    :rtype: 1D array
    """
    P = d * (w * norm.pdf(x_max, loc=0, scale=sigma)) * \
        (w * norm.cdf(x_max, loc=0, scale=sigma) + (1 - w) * expon.cdf(x_max, scale=1 / lamb)) ** (d - 1)
    return P


def pollution_gauss_noise_gauss_comp(x_max, d, sigma, mu, sigma_2):
    """
    A partial template for purely Gaussian noise with a pollution that follows another Gaussian distribution. Only the
    Gaussian component is returned.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution
    :type sigma: float
    :param mu: The mean of the Gaussian pollution.
    :type mu: float
    :param sigma_2: The standard deviation of the Gaussian pollution.
    :type sigma_2: float
    :return: The template of the Gaussian part.
    :rtype: 1D array
    """
    P = (d - 1) * norm.cdf(x_max, loc=mu, scale=sigma_2) * norm.pdf(x_max, loc=0, scale=sigma) * norm.cdf(x_max, loc=0,
                                                                                                          scale=sigma) ** (
                d - 2)
    return P


def fraction_gauss_noise_gauss_comp(x_max, d, sigma, mu, sigma_2, w):
    """
    A partial template for a two component Gaussian noise mixture. Only the
    first Gaussian component is returned.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution.
    :type sigma: float
    :param mu: The mean of the second component.
    :type mu: float
    :param sigma: The standard deviation of the second component.
    :type sigma: float
    :param w: The weight of the first Gaussian fraction.
    :type w: float
    :return: The template of the first Gaussian component.
    :rtype: 1D array
    """
    P = d * (w * norm.pdf(x_max, loc=0, scale=sigma)) * \
        (w * norm.cdf(x_max, loc=0, scale=sigma) + (1 - w) * norm.cdf(x_max, loc=mu, scale=sigma_2)) ** (d - 1)
    return P


# ------------------------------------------
# only pollution components
# ------------------------------------------


def pollution_exponential_noise_poll_comp(x_max, d, sigma, lamb):
    """
    A partial template for purely Gaussian noise with a pollution that follows an exponential distribution. Only the
    polluting component is returned.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution
    :type sigma: float
    :param lamb: The rate parameter of the exponential distribution.
    :type lamb: float
    :return: The template of the pollution.
    :rtype: 1D array
    """
    P = expon.pdf(x_max, scale=1 / lamb) * norm.cdf(x_max, loc=0, scale=sigma) ** (d - 1)
    return P


def fraction_exponential_noise_poll_comp(x_max, d, sigma, lamb, w):
    """
    A partial template for a Gaussian-exponential noise mixture. Only the
    exponential component is returned.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution.
    :type sigma: float
    :param lamb: The rate parameter of the exponential distribution.
    :type lamb: float
    :param w: The weight of the Gaussian fraction.
    :type w: float
    :return: The template of the exponential component.
    :rtype: 1D array
    """
    P = d * ((1 - w) * expon.pdf(x_max, scale=1 / lamb)) * \
        (w * norm.cdf(x_max, loc=0, scale=sigma) + (1 - w) * expon.cdf(x_max, scale=1 / lamb)) ** (d - 1)
    return P


def pollution_gauss_noise_poll_comp(x_max, d, sigma, mu, sigma_2):
    """
    A partial template for purely Gaussian noise with a pollution that follows another Gaussian distribution. Only the
    polluting component is returned.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution
    :type sigma: float
    :param mu: The mean of the Gaussian pollution.
    :type mu: float
    :param sigma_2: The standard deviation of the Gaussian pollution.
    :type sigma_2: float
    :return: The template of the pollution.
    :rtype: 1D array
    """
    P = norm.pdf(x_max, loc=mu, scale=sigma_2) * norm.cdf(x_max, loc=0, scale=sigma) ** (d - 1)
    return P


def fraction_gauss_noise_poll_comp(x_max, d, sigma, mu, sigma_2, w):
    """
    A partial template for a two component Gaussian noise mixture. Only the
    second Gaussian component is returned.

    :param x_max: The maxima of the empty noise baselines.
    :type x_max: 1D array
    :param d: The number of independent samples.
    :type d: float
    :param sigma: The baseline resolution.
    :type sigma: float
    :param mu: The mean of the second component.
    :type mu: float
    :param sigma: The standard deviation of the second component.
    :type sigma: float
    :param w: The weight of the first Gaussian fraction.
    :type w: float
    :return: The template of the second Gaussian component.
    :rtype: 1D array
    """
    P = d * ((1 - w) * norm.pdf(x_max, loc=mu, scale=sigma_2)) * \
        (w * norm.cdf(x_max, loc=0, scale=sigma) + (1 - w) * norm.cdf(x_max, loc=mu, scale=sigma_2)) ** (d - 1)
    return P


# ------------------------------------------
# negative likelihood functions for unbinned fit
# ------------------------------------------


def nll_gauss(pars, x):
    d, sigma = pars
    if sigma < 1e-5:
        return 10e6 + 10e3 * np.abs(sigma - 1e-5)
    elif d < 1:
        return 10e6 + 10e3 * np.abs(d - 1)
    else:
        return -np.sum(np.log(gauss_noise(x, d, sigma)))


def nll_pollution_exponential(pars, x):
    d, sigma, lamb = pars
    if sigma < 1e-5:
        return 10e6 + 10e3 * np.abs(sigma - 1e-5)
    elif d < 1:
        return 10e6 + 10e3 * np.abs(d - 1)
    else:
        return -np.sum(np.log(pollution_exponential_noise(x, d, sigma, lamb)))


def nll_fraction_exponential(pars, x):
    d, sigma, lamb, w = pars
    if sigma < 1e-5:
        return 10e6 + 10e3 * np.abs(sigma - 1e-5)
    elif d < 1:
        return 10e6 + 10e3 * np.abs(d - 1)
    elif w < 0 or w > 1:
        return 10e6
    else:
        return -np.sum(np.log(fraction_exponential_noise(x, d, sigma, lamb, w)))


def nll_pollution_gauss(pars, x):
    d, sigma, mu, sigma_2 = pars
    if sigma < 1e-5:
        return 10e6 + 10e3 * np.abs(sigma - 1e-5)
    elif d < 1:
        return 10e6 + 10e3 * np.abs(d - 1)
    elif sigma_2 < 1e-5:
        return 10e6 + 10e3 * np.abs(sigma_2 - 1e-5)
    else:
        return -np.sum(np.log(pollution_gauss_noise(x, d, sigma, mu, sigma_2)))


def nll_fraction_gauss(pars, x):
    d, sigma, mu, sigma_2, w = pars
    if sigma < 1e-5:
        return 10e6 * 10e3 * np.abs(sigma - 1e-5)
    elif d < 1:
        return 10e6 + 10e3 * np.abs(d - 1)
    elif sigma_2 < 1e-5:
        return 10e6 * 10e3 * np.abs(sigma_2 - 1e-5)
    elif w < 0 or w > 1:
        return 10e6
    else:
        return -np.sum(np.log(fraction_gauss_noise(x, d, sigma, mu, sigma_2, w)))


# ------------------------------------------
# actual function for the fit
# ------------------------------------------


def get_noise_parameters_unbinned(events,
                                  model='gauss',
                                  sigma_x0=2,
                                  ):
    """
    Find the maximum likelihood estimators of all noise trigger model parameters in an unbinned maximum likelihood fit.

    :param events: The array of the unbinned noise baseline maxima.
    :type events: 1D array
    :param model: The model that is used to fit the noise maxima.
        - 'gauss': Purely Gaussian noise model.
        - 'pollution_exponential': Gaussian noise model with an exponentially distributed pollution.
        - 'fraction_exponential': Gaussian-exponential mixture noise model.
        - 'pollution_gauss': Gaussian noise model with an Gaussian distributed pollution.
        - 'fraction_gauss': Gaussian mixture noise model.
    :type model: string
    :param sigma_x0: The start value for the baseline resolution.
    :type sigma_x0: float
    :return: The fitted parameters. These are different for each model and compatible with the functions gauss_noise,
        pollution_exponential_noise, fraction_exponential_noise and pollution_gauss_noise.
    :rtype: 1D array
    """

    if model == 'gauss':
        # d, sigma
        minimizer = nll_gauss
        x0 = np.array([400, sigma_x0])
    elif model == 'pollution_exponential':
        # d, sigma, lamb
        minimizer = nll_pollution_exponential
        x0 = np.array([400, sigma_x0, 2])
    elif model == 'fraction_exponential':
        # d, sigma, lamb, w
        minimizer = nll_fraction_exponential
        x0 = np.array([400, sigma_x0, 2, 0.9])
    elif model == 'pollution_gauss':
        # d, sigma, mu, sigma_2
        minimizer = nll_pollution_gauss
        x0 = np.array([400, sigma_x0, 0, 2])
    elif model == 'fraction_gauss':
        # d, sigma, mu, sigma_2, w
        minimizer = nll_fraction_gauss
        x0 = np.array([400, sigma_x0, 0, 2, 0.5])
    else:
        raise NotImplementedError('This model is not implemented!')

    res = minimize(minimizer,
                   args=(events,),
                   x0=x0,
                   method='nelder-mead')

    print('Independent samples d: ', res.x[0])
    print('Baseline resolution sigma (mV): ', res.x[1])
    if model == 'pollution_exponential':
        print('Exponential pollution rate parameter lambda (1/mV): ', res.x[2])
    if model == 'fraction_exponential':
        print('Exponential fraction rate parameter lambda (1/mV): ', res.x[2])
        print('Weight gaussian baseline parameter w: ', res.x[3])
    if model == 'pollution_gauss':
        print('Gaussian pollution mean parameter mu (mV): ', res.x[2])
        print('Gaussian pollution standard deviation parameter sigma_2 (mV): ', res.x[3])
    if model == 'fraction_gauss':
        print('Gaussian fraction mean parameter mu (mV): ', res.x[2])
        print('Gaussian fraction standard deviation parameter sigma_2 (mV): ', res.x[3])
        print('Weight gaussian baseline parameter w: ', res.x[4])

    return res.x


# ------------------------------------------
# plot
# ------------------------------------------

def plot_noise_trigger_model(bins_hist,
                             counts_hist,
                             x_grid,
                             trigger_window,
                             ph_distribution,
                             xran_hist,
                             noise_trigger_rate,
                             polluted_trigger_rate,
                             threshold,
                             polluted_ph_distribution,
                             nmbr_pollution_triggers,
                             model='gauss',
                             title=None,
                             yran=None,
                             allowed_noise_triggers=1,
                             xran=None,
                             ylog=False,
                             only_histogram=False,
                             save_path=None,
                             ):
    """
    Plot the noise trigger model.

    :param bins_hist: The bins edges of the noise maxima histogram. This array is one number longer than the counts_hist
        array.
    :type bins_hist: 1D array
    :param counts_hist: The counts of all histogram bins.
    :type counts_hist: 1D array
    :param x_grid: The grid to evaluate the noise trigger models on.
    :type x_grid: 1D array
    :param trigger_window: The exposure of the trigger window, in kg days.
    :type trigger_window: float
    :param ph_distribution: The evaluated noise template on x_grid, normalized to the exposure.
    :type ph_distribution: 1D array
    :param xran_hist: The range of the x axis on the histogram plot.
    :type xran_hist: tuple of two floats
    :param noise_trigger_rate: The number of noise triggers, depending on the threshold, given by the x_grid.
    :type noise_trigger_rate: 1D array
    :param polluted_trigger_rate: The number of pollution triggers, depending on the threshold, given by the x_grid.
    :type polluted_trigger_rate: 1D array
    :param threshold: The threshold in mV.
    :type threshold: float
    :param polluted_ph_distribution: The polluted component of the evaluated noise template on x_grid,
        normalized to the exposure.
    :type polluted_ph_distribution: 1D array
    :param nmbr_pollution_triggers: The number of pollution triggers.
    :type nmbr_pollution_triggers: float
    :param model: Which model was fit to the noise.
        - 'gauss': Model of purely Gaussian noise.
        - 'pollution_exponential': Model of Gaussian noise with one exponentially distributed sample on each baseline.
        - 'fraction_exponential': Mixture model of Gaussian and exponentially distributed noise.
        - 'pollution_gauss': Model of Gaussian noise and one sample in each baseline that follows another, also Gaussian distribution.
        - 'fraction_gauss':  Mixture model of two Gaussian noise components.
    :param title: A title for both plots.
    :type title: string
    :param yran: The range of the y axis on both plots.
    :type yran: tuple of two floats
    :param allowed_noise_triggers: The number of noise triggers that are allowed per kg day exposure.
        :type allowed_noise_triggers: float
    :param xran: The range of the x axis on the noise trigger estimation plot.
    :type xran: tuple of two floats
    :param ylog: If set, the y axis is plotted logarithmically on the histogram plot.
    :type ylog: bool
    :param only_histogram: Plot only the histogram, not the noise trigger model.
    :type only_histogram: bool
    :param save_path: A path to save the plots.
    :type save_path: string
    """

    # plot the counts
    plt.close()
    use_cait_style()
    xdata = bins_hist[:-1] + (bins_hist[1] - bins_hist[0]) / 2
    plt.hist(xdata, bins_hist, weights=counts_hist / trigger_window,
             zorder=8, alpha=0.8, label='Counts')
    if model == 'pollution_exponential' or model == 'pollution_gauss':
        plt.plot(x_grid, ph_distribution, linewidth=2, zorder=12, color='C2', label='Gaussian Component')
        plt.plot(x_grid, polluted_ph_distribution, linewidth=2, zorder=12, color='yellow', label='Pollution Component')
        plt.plot(x_grid, ph_distribution + polluted_ph_distribution, linewidth=2, zorder=14, color='black',
                 label='Cumulative Model')
    else:
        plt.plot(x_grid, ph_distribution, linewidth=2, zorder=12, color='black', label='Fit')
    make_grid()
    if title is not None:
        plt.title(title)
    if xran_hist is not None:
        plt.xlim(xran_hist)
    if yran is not None:
        plt.ylim(yran)
    else:
        plt.ylim(bottom=0.1)
    if ylog:
        plt.yscale('log')
        plt.ylim(bottom=np.min(counts_hist[counts_hist > 0] / trigger_window) / 10)
    plt.legend()
    plt.xlabel('Pulse Height (mV)')
    plt.ylabel('Counts (1 / kg days mV)')
    if save_path is not None:
        name = save_path + '_counts.pdf'
        plt.savefig(name)
    plt.show()

    if not only_histogram:

        # plot the threshold
        plt.close()
        use_cait_style()
        plt.plot(x_grid, noise_trigger_rate, linewidth=2, zorder=16, color='black', label='Noise Triggers')
        if model == 'pollution_exponential' or model == 'pollution_gauss':
            plt.plot(x_grid, polluted_trigger_rate, linewidth=2, zorder=17, color='grey', linestyle='dotted',
                     label='Pollution Triggers')
        plt.vlines(x=threshold, ymin=0.01, ymax=allowed_noise_triggers, color='tab:red',
                   linewidth=2, zorder=20, label='{} / kg days'.format(allowed_noise_triggers))
        plt.hlines(y=allowed_noise_triggers, xmin=0, xmax=threshold, color='tab:red',
                   linewidth=2, zorder=20)
        if model == 'pollution_exponential' or model == 'pollution_gauss':
            plt.vlines(x=threshold, ymin=0.01, ymax=nmbr_pollution_triggers, linestyles='dotted', color='tab:red',
                       linewidth=2, zorder=21, label='{:.2f} / kg days'.format(nmbr_pollution_triggers))
            plt.hlines(y=nmbr_pollution_triggers, xmin=0, xmax=threshold, linestyles='dotted', color='tab:red',
                       linewidth=2, zorder=21)
        make_grid()
        if yran is None:
            plt.ylim(bottom=0.1)
        else:
            plt.ylim(yran)
        if xran is not None:
            plt.xlim(xran)
        plt.yscale('log')
        if title is not None:
            plt.title(title)
        plt.legend()
        plt.xlabel('Threshold (mV)')
        plt.ylabel('Noise Trigger Rate (1 / kg days)')
        if save_path is not None:
            name = save_path + '_ntr.pdf'
            plt.savefig(name)
        plt.show()


# ------------------------------------------
# calc the threshold
# ------------------------------------------

def calc_threshold(record_length, sample_length, detector_mass, interval_restriction, ul, ll, model,
                   pars, allowed_noise_triggers):
    """
    Calculate the treshold for given noise baseline maxima, to obtain a defined number of noise triggers.

    This method was described in "M. Mancuso et. al., A method to define the energy threshold depending on
    noise level for rare event searches" (doi 10.1016/j.nima.2019.06.030).

    :param record_length: The number of samples in a record window.
    :type record_length: int
    :param sample_length: The length of a sample in seconds.
    :type sample_length: float
    :param detector_mass: The mass of the detector in kg.
    :type detector_mass: float
    :param interval_restriction: A value between 0 and 1. Only this share of the trigger window is used in the maximum
        search. This is typically 0.75 if filters are applied, to avoid border effects.
    :type interval_restriction: float
    :param ul: The upper limit of the interval that is used to search a threshold, in mV.
    :type ul: float
    :param ll: The lower limit of the interval that is used to search a threshold, in mV.
    :type ll: float
    :param model: Determine which model is fit to the noise.
            - 'gauss': Model of purely Gaussian noise.
            - 'pollution_exponential': Model of Gaussian noise with one exponentially distributed sample on each baseline.
            - 'fraction_exponential': Mixture model of Gaussian and exponentially distributed noise.
            - 'pollution_gauss': Model of Gaussian noise and one sample in each baseline that follows another, also Gaussian distribution.
            - 'fraction_gauss':  Mixture model of two Gaussian noise components.
    :type model: string
    :param pars: The fitted parameters for the model. They have to match the chosen model
    :type pars: tuple
    :param allowed_noise_triggers: The number of noise triggers that are allowed per kg day exposure.
    :type allowed_noise_triggers: float
    :return: 8-tuple
            - The grid to evaluate the noise trigger models on.
            - The exposure of the trigger window, in kg days.
            - The evaluated noise template on x_grid, normalized to the exposure.
            - The polluted component of the evaluated noise template on x_grid,
        normalized to the exposure.
            - The number of noise triggers, depending on the threshold, given by the x_grid.
            - The number of pollution triggers, depending on the threshold, given by the x_grid.
            - The threshold in mV.
            - The number of pollution triggers.
    :rtype: tuple
    """

    if model == 'gauss':
        d, sigma = pars
    elif model == 'pollution_exponential':
        d, sigma, lamb = pars
    elif model == 'fraction_exponential':
        d, sigma, lamb, w = pars
    elif model == 'pollution_gauss':
        d, sigma, mu, sigma_2 = pars
    elif model == 'fraction_gauss':
        d, sigma, mu, sigma_2, w = pars
    else:
        raise NotImplementedError('This model is not implemented!')

    # calc the exposure in kg days
    trigger_window = record_length * sample_length * detector_mass / 3600 / 24 * interval_restriction

    # get the noise trigger rate
    num = 3000
    h = (ul - ll) / num
    x_grid = np.linspace(start=ll, stop=ul, num=num)
    if model == 'gauss':
        ph_distribution = noise_trigger_template(x_max=x_grid, d=d, sigma=sigma)

    elif model == 'fraction_exponential':
        ph_distribution = fraction_exponential_noise(x_grid, *pars)

    elif model == 'fraction_gauss':
        ph_distribution = fraction_gauss_noise(x_grid, *pars)

    else:
        # get the correct template
        if model == 'pollution_exponential':
            template_noise = pollution_exponential_noise_gauss_comp
            template_pollution = pollution_exponential_noise_poll_comp
        elif model == 'pollution_gauss':
            template_noise = pollution_gauss_noise_gauss_comp
            template_pollution = pollution_gauss_noise_poll_comp

        # get the noise trigger rate
        ph_distribution = template_noise(x_grid, *pars)

        # get the polluted noise trigger rate
        polluted_ph_distribution = template_pollution(x_grid, *pars)
        polluted_trigger_rate = np.array(
            [h * np.sum(polluted_ph_distribution[i:]) for i in range(len(polluted_ph_distribution))])
        polluted_ph_distribution /= trigger_window
        polluted_trigger_rate /= trigger_window

    # get the noise trigger rate
    noise_trigger_rate = np.array([h * np.sum(ph_distribution[i:]) for i in range(len(ph_distribution))])
    ph_distribution /= trigger_window
    noise_trigger_rate /= trigger_window

    # calc the threshold
    try:
        threshold = x_grid[noise_trigger_rate < allowed_noise_triggers][0]
        if threshold == ul:
            raise IndexError
    except IndexError:
        if model == 'pollution_exponential' or model == 'pollution_gauss':
            retval = (x_grid, \
                      trigger_window, \
                      ph_distribution, \
                      polluted_ph_distribution, \
                      noise_trigger_rate, \
                      polluted_trigger_rate, \
                      None, \
                      None)
        else:
            retval = (x_grid, \
                      trigger_window, \
                      ph_distribution, \
                      None, \
                      noise_trigger_rate, \
                      None, \
                      None, \
                      None)
        raise IndexError('The threshold for the wanted number of noise triggers is above the set interval. '
                         'Either increase the interval or allow for more noise triggers!', retval)
    print('Threshold for {} Noise Trigger per kg day: {:.3f} mV'.format(allowed_noise_triggers, threshold))
    if model == 'pollution_exponential' or model == 'pollution_gauss':
        nmbr_pollution_triggers = polluted_trigger_rate[x_grid == threshold][0]
        print('Pollution Triggers per kg day for this Threshold: {:.3f}'.format(nmbr_pollution_triggers))

    if model == 'pollution_exponential' or model == 'pollution_gauss':
        return x_grid, \
               trigger_window, \
               ph_distribution, \
               polluted_ph_distribution, \
               noise_trigger_rate, \
               polluted_trigger_rate, \
               threshold, \
               nmbr_pollution_triggers
    else:
        return x_grid, \
               trigger_window, \
               ph_distribution, \
               None, \
               noise_trigger_rate, \
               None, \
               threshold, \
               None
