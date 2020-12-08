# imports
import h5py
import numpy as np
from scipy.stats import gaussian_kde
from ..fit._templates import baseline_template_cubic
from ..data._baselines import get_nps, get_cc_noise


# Simulate Baselines
def simulate_baselines(path_h5,
                       size,
                       rms_thresholds,
                       lamb=0.01,
                       kde=True,
                       sim_poly=True,
                       verb=False):
    """
    Creates fake baselines with given nps and drift structure

    :param path_h5: string, path to the file from which nps and bl drifts come
    :param size: int, nmbr of baselines to simulate
    :param rms_thresholds: list of two ints, threshold for the bl rms fit
        error above which they get not included in the nps and drifts
    :param lamb: float, parameter for the noise simulation method
    :param verb: bool, if true feedback about progress in code
    :return: tuple (3D array - (ch_nmbr, size bl, rec_len) of the simulated baselines,
                    3D array - (ch_nmbr, size bl, rec_len) of the simulated polynomials)
    """

    # kernel density estimator
    h5f = h5py.File(path_h5, 'r')
    record_length = len(h5f['noise']['event'][0, 0])
    fitpar = np.array(h5f['noise']['fit_coefficients'])
    nmbr_channels = len(fitpar)
    rms = np.array(h5f['noise']['fit_rms'])
    nps = h5f['noise']['nps']

    t = np.linspace(0, record_length - 1, record_length)

    # simulate polynomials
    polynomials = np.zeros((nmbr_channels, size, record_length))
    if sim_poly:
        if verb:
            print('Simulating Polynomials.')
        for i in range(nmbr_channels):
            p = fitpar[i]
            p = p[rms[i] < rms_thresholds[i]]
            if kde:
                kde = gaussian_kde(p.T)
                simpar = kde.resample(size=size).T
            else:
                mean = np.mean(p, axis=0)
                cov = np.cov(p.T)
                simpar = np.random.multivariate_normal(mean, cov, size=size)
            for j in range(size):
                polynomials[i, j, :] = baseline_template_cubic(t,
                                                               c0=simpar[j, 0],
                                                               c1=simpar[j, 1],
                                                               c2=simpar[j, 2],
                                                               c3=simpar[j, 3])

        # calculate polynomial nps
        if verb:
            print('Calculating Polynomial NPS.')
        mnps_poly = np.zeros((nmbr_channels, int(record_length / 2 + 1)))
        for c in range(nmbr_channels):
            for i, p in enumerate(polynomials[c]):
                mnps_poly[c] += get_nps(p)
        mnps_poly /= size

        nps -= mnps_poly
        nps[nps <= 0] = 0

    # simulate noise with difference nps
    if verb:
        print('Simulating Noise with difference NPS.')
    baselines = np.zeros((nmbr_channels, size, record_length))
    for c in range(nmbr_channels):
        baselines[c] = polynomials[c] + get_cc_noise(nmbr_noise=size,
                                                     nps=nps[c],
                                                     lamb=lamb,
                                                     verb=True)
    h5f.close()
    if verb:
        print('Baseline Simulation done.')

    return baselines, polynomials
