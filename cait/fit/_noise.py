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
    # TODO
    P = (d / np.sqrt(2 * np.pi) / sigma)
    P *= np.exp(-(x_max / np.sqrt(2) / sigma) ** 2)
    P *= (1 / 2 + erf(x_max / np.sqrt(2) / sigma) / 2) ** (d - 1)
    return P


def wrapper(x_max, d, sigma):
    # TODO

    sigma_lower_bound = 0.0001
    if sigma < sigma_lower_bound:
        P = 1000 * np.abs(sigma_lower_bound - sigma) + 1000
    else:
        P = noise_trigger_template(x_max, d, sigma)

    return P


def get_noise_parameters_binned(counts,
                                bins,
                                ):
    # TODO

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
    P = d * norm.pdf(x_max, loc=0, scale=sigma) * norm.cdf(x_max, loc=0, scale=sigma) ** (d - 1)
    P[P < 10e-8] = 10e-8
    return P


def pollution_exponential_noise(x_max, d, sigma, lamb):
    P = expon.pdf(x_max, scale=1 / lamb) * norm.cdf(x_max, loc=0, scale=sigma) ** (d - 1)
    P += expon.cdf(x_max, scale=1 / lamb) * (d - 1) * norm.pdf(x_max, loc=0, scale=sigma) * norm.cdf(x_max, loc=0,
                                                                                                     scale=sigma) ** (
                 d - 2)
    P[P < 10e-10] = 10e-10
    return P


def fraction_exponential_noise(x_max, d, sigma, lamb, w):
    P = d * (w * norm.pdf(x_max, loc=0, scale=sigma) + (1 - w) * expon.pdf(x_max, scale=1 / lamb)) * \
        (w * norm.cdf(x_max, loc=0, scale=sigma) + (1 - w) * expon.cdf(x_max, scale=1 / lamb)) ** (d - 1)
    P[P < 10e-8] = 10e-8
    return P


def pollution_gauss_noise(x_max, d, sigma, mu, sigma_2):
    P = norm.pdf(x_max, loc=mu, scale=sigma_2) * norm.cdf(x_max, loc=0, scale=sigma) ** (d - 1)
    P += (d - 1) * norm.cdf(x_max, loc=mu, scale=sigma_2) * norm.pdf(x_max, loc=0, scale=sigma) * norm.cdf(x_max, loc=0,
                                                                                                           scale=sigma) ** (
                 d - 2)
    P[P < 10e-8] = 10e-8
    return P


def fraction_gauss_noise(x_max, d, sigma, mu, sigma_2, w):
    P = d * (w * norm.pdf(x_max, loc=0, scale=sigma) + (1 - w) * norm.pdf(x_max, loc=mu, scale=sigma_2)) * \
        (w * norm.cdf(x_max, loc=0, scale=sigma) + (1 - w) * norm.cdf(x_max, loc=mu, scale=sigma_2)) ** (d - 1)
    P[P < 10e-8] = 10e-8
    return P


# ------------------------------------------
# only gauss components
# ------------------------------------------

def pollution_exponential_noise_gauss_comp(x_max, d, sigma, lamb):
    P = expon.cdf(x_max, scale=1 / lamb) * (d - 1) * norm.pdf(x_max, loc=0, scale=sigma) * norm.cdf(x_max, loc=0,
                                                                                                    scale=sigma) ** (
                d - 2)
    return P


def fraction_exponential_noise_gauss_comp(x_max, d, sigma, lamb, w):
    P = d * (w * norm.pdf(x_max, loc=0, scale=sigma)) * \
        (w * norm.cdf(x_max, loc=0, scale=sigma) + (1 - w) * expon.cdf(x_max, scale=1 / lamb)) ** (d - 1)
    return P


def pollution_gauss_noise_gauss_comp(x_max, d, sigma, mu, sigma_2):
    P = (d - 1) * norm.cdf(x_max, loc=mu, scale=sigma_2) * norm.pdf(x_max, loc=0, scale=sigma) * norm.cdf(x_max, loc=0,
                                                                                                          scale=sigma) ** (
                d - 2)
    return P


def fraction_gauss_noise_gauss_comp(x_max, d, sigma, mu, sigma_2, w):
    P = d * (w * norm.pdf(x_max, loc=0, scale=sigma)) * \
        (w * norm.cdf(x_max, loc=0, scale=sigma) + (1 - w) * norm.cdf(x_max, loc=mu, scale=sigma_2)) ** (d - 1)
    return P


# ------------------------------------------
# only pollution components
# ------------------------------------------


def pollution_exponential_noise_poll_comp(x_max, d, sigma, lamb):
    P = expon.pdf(x_max, scale=1 / lamb) * norm.cdf(x_max, loc=0, scale=sigma) ** (d - 1)
    return P


def fraction_exponential_noise_poll_comp(x_max, d, sigma, lamb, w):
    P = d * ((1 - w) * expon.pdf(x_max, scale=1 / lamb)) * \
        (w * norm.cdf(x_max, loc=0, scale=sigma) + (1 - w) * expon.cdf(x_max, scale=1 / lamb)) ** (d - 1)
    return P


def pollution_gauss_noise_poll_comp(x_max, d, sigma, mu, sigma_2):
    P = norm.pdf(x_max, loc=mu, scale=sigma_2) * norm.cdf(x_max, loc=0, scale=sigma) ** (d - 1)
    return P


def fraction_gauss_noise_poll_comp(x_max, d, sigma, mu, sigma_2, w):
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
                                  ):
    # TODO

    if model == 'gauss':
        # d, sigma
        minimizer = nll_gauss
        x0 = np.array([400, 2])
    elif model == 'pollution_exponential':
        # d, sigma, lamb
        minimizer = nll_pollution_exponential
        x0 = np.array([400, 2, 2])
    elif model == 'fraction_exponential':
        # d, sigma, lamb, w
        minimizer = nll_fraction_exponential
        x0 = np.array([400, 2, 2, 0.9])
    elif model == 'pollution_gauss':
        # d, sigma, mu, sigma_2
        minimizer = nll_pollution_gauss
        x0 = np.array([400, 2, 0, 2])
    elif model == 'fraction_gauss':
        # d, sigma, mu, sigma_2, w
        minimizer = nll_fraction_gauss
        x0 = np.array([400, 2, 0, 2, 0.5])
    else:
        raise NotImplementedError('This model is not implemented!')

    res = minimize(minimizer,
                   args=(events,),
                   x0=x0,
                   method='nelder-mead')

    print('Independent samples d: ', res.x[0])
    print('Baseline resolution sigma (mV): ', res.x[1])
    if model == 'pollution_exponential':
        print('Exponential pollution rate parameter lambda (mV): ', res.x[2])
    if model == 'fraction_exponential':
        print('Exponential fraction rate parameter lambda (mV): ', res.x[2])
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
                             model,
                             polluted_ph_distribution,
                             title,
                             xran_hist,
                             noise_trigger_rate,
                             polluted_trigger_rate,
                             threshold,
                             yran,
                             allowed_noise_triggers,
                             nmbr_pollution_triggers,
                             xran,
                             ylog,
                             only_histogram,
                             save_path,
                             ):
    # TODO

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
    # TODO

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
