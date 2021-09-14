# imports
import h5py
import numpy as np
from scipy.stats import gaussian_kde
from ..fit._templates import baseline_template_cubic
from ..data._baselines import get_nps, get_cc_noise
from tqdm.auto import trange


# Simulate Baselines
def simulate_baselines(path_h5,
                       size,
                       rms_thresholds,
                       lamb=0.01,
                       kde=True,
                       sim_poly=True,
                       verb=False):
    """
    Creates fake baselines with given nps and drift structure.

    :param path_h5: Path to the file from which nps and bl drifts come.
    :type path_h5: string
    :param size: Nmbr of baselines to simulate.
    :type size: int
    :param rms_thresholds: Threshold for the bl rms fit
        error above which they get not included in the nps and drifts.
    :type rms_thresholds: list of two ints
    :param lamb: Parameter for the noise simulation method.
    :type lamb: float
    :param kde: If True we sample the coefficients of the bl fit with a kernel density estimation.
    :type kde: bool
    :param sim_poly: If True we simulate the polynomials for the baselines.
    :type sim_poly: pool
    :param verb: If true feedback about progress in code.
    :type verb: bool
    :return: (ch_nmbr, size bl, rec_len) of the simulated baselines, (ch_nmbr, size bl, rec_len) of the simulated polynomials.
    :type: tuple (3D array, 3D array)
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

    batchsize = 50
    nmbr_batches = int(size / batchsize)
    rest = int(size - nmbr_batches * batchsize)

    for c in range(nmbr_channels):

        for i in trange(nmbr_batches):
            baselines[c, i * batchsize:(i + 1) * batchsize] = polynomials[c, i * batchsize:(i + 1) * batchsize] + get_cc_noise(nmbr_noise=batchsize,
                                                                                            nps=nps[c],
                                                                                            lamb=lamb)
        if rest > 0:
            baselines[c, -rest:] = polynomials[c, -rest:] + get_cc_noise(nmbr_noise=rest,
                                                                 nps=nps[c],
                                                                 lamb=lamb)

    h5f.close()
    if verb:
        print('Baseline Simulation done.')

    return baselines, polynomials
