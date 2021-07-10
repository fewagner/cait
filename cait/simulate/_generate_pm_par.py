
import scipy.stats as sci
import numpy as np
import warnings

def norm(x, mu, sigma):
    return (x - mu) / sigma


def denorm(x, mu, sigma):
    return sigma * x + mu


def generate_ps_par(h):
    """
    Simulate PS parameters modelled after the TUM40 phonon detector.

    :param h: Array of pulse heights.
    :type h: 1D array
    :return: (t0, An, At, tau_n, tau_in, tau_t)
    :rtype: list of 6 1D arrays
    """

    warnings.warn('Watch out, this function is not fully tested!')

    SAMPLE_SIZE = len(h)

    h_mean = 0.89
    h_std = 0.57
    tau_n_mean = 112.64
    tau_n_std = 38.04
    t0_mean = -1.36
    t0_std = 0.29
    # An_mean = 1.27
    # An_std = 0.81
    # At_mean = 1.03
    # At_std = 0.66
    sigma_tn_mean = 0.718
    sigma_tn_std = 0.709
    tau_in_mean = 2.52
    tau_in_std = 0.058
    tau_t_mean = 21.45
    tau_t_std = 1.46

    # these equations are found with genetic algorithms

    # independent var
    h_ = norm(h, h_mean, h_std)

    # semidependent vars
    tau_n_ = sci.uniform.rvs(size=SAMPLE_SIZE,
                             loc=-0.134 - np.sqrt(3) * denorm(-0.312 * h_, sigma_tn_mean, sigma_tn_std),
                             scale=2 * np.sqrt(3) * denorm(-0.312 * h_, sigma_tn_mean,
                                                           sigma_tn_std))  # underscore means the normed variable

    # dependent vars
    # An_ = np.copy(h_)
    # At_ = np.copy(h_)
    t0_ = (h_ - 0.455) / (0.504 * h_ + 1.103) + sci.norm.rvs(size=SAMPLE_SIZE, loc=0,
                                                             scale=0.644)
    tau_in_ = 0.23 * tau_n_ ** 2 - tau_n_ - 0.331 + sci.norm.rvs(size=SAMPLE_SIZE, loc=0, scale=0.5454978 / 2)
    tau_t_ = -0.468 * tau_n_ ** 2 + tau_n_ + 0.372 + sci.norm.rvs(size=SAMPLE_SIZE, loc=0, scale=1.0354906 / 2)

    # transform back
    t0 = denorm(t0_, t0_mean, t0_std)
    An = h / 0.705  # denorm(denorm(An_,h_mean,h_std), An_mean, An_std)
    At = h / 0.858  # denorm(denorm(At_,h_mean,h_std), At_mean, At_std)
    tau_n = denorm(tau_n_, tau_n_mean, tau_n_std)
    tau_in = denorm(tau_in_, tau_in_mean, tau_in_std)
    tau_t = denorm(tau_t_, tau_t_mean, tau_t_std)

    return np.array([t0, An, At, tau_n, tau_in, tau_t])
