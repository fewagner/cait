# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
from ..fit._templates import baseline_template_quad, baseline_template_cubic
from scipy.optimize import curve_fit
from scipy.stats import norm, uniform
from tqdm.auto import tqdm

# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------


def get_nps(x):
    """
    Calculates the Noise Power Spectrum (NPS) of a given array.

    :param x: The time series.
    :type x: 1D numpy array of size N
    :return: The noise power spectrum.
    :rtype: 1D numpy array of size N/2 + 1
    """
    x = np.fft.rfft(x)
    x = np.abs(x) ** 2
    return x


def noise_function(nps):
    """
    Simulates the function f from CC-Noise Algo
    with a given Noise Power Spectrum (NPS)
    see arXiv:1006.3289, eq (4) and (5)

    :param f: The noise power spectrum.
    :type f: real valued 1D array of size N/2 + 1
    :return: Noise Baselines.
    :rtype: real valued 1D array of size N
    """
    f = np.sqrt(nps)
    f = np.array(f, dtype='complex')  # create array for frequencies
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi  # create random phases
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np + 1] *= phases
    f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])
    return np.fft.irfft(f)


def get_cc_noise(nmbr_noise,
                 nps,
                 lamb=0.01):
    """
    Simulation of a noise baseline, according to Carretoni Cremonesi: arXiv:1006.3289

    :param nmbr_noise: Number of noise baselines to simulate.
    :type nmbr_noise: int > 0
    :param nps: Noise power spectrum of the baselines,
        e.g. generated with scipy.fft.rfft().
    :type nps: 1D array of odd size
    :param lamb: Parameter of the method (overlap between ).
    :type lamb: integer > 0
    :return: The simulated baselines.
    :rtype: 2D array of size (nmbr_noise, 2*(len(nps)-1))
    """

    T = (len(nps) - 1)*2

    # alpha is fixed by the function we use to 1
    a = np.sqrt(1/lamb/T)

    noise = np.zeros((nmbr_noise, T))

    for i in tqdm(range(nmbr_noise)):
        t = 0
        while t < T:
            noise[i] += norm.rvs(scale=a) * np.roll(noise_function(nps), t)
            t = int(t - np.log(1 - uniform.rvs())/lamb)

    return noise

def calculate_mean_nps(baselines,
                       order_polynom=3,
                       clean=True,
                       percentile=50,
                       down=1,
                       sample_length = 0.00004,
                       rms_baselines=None,
                       rms_cutoff=None):
    """
    Calculates the mean Noise Power Spectrum (mNPS) of a set of baselines,
    after cleaning them from artifacts with a polynomial fit.

    :param baselines: The baselines for the mean NPS.
    :type baselines: 2D array of size (nmbr_baselines, record_length)
    :param order_polynom: 2 or 3, the order of the fitted polynomial.
    :type order_polynom: int
    :param clean: If True the baselines are cleaned from artifacts with a poly fit.
    :type clean: boolean
    :param percentile: The percentile of the Fit RMS that is still used
        for the calculation of the mNPS.
    :type percentile: float between 0 and 1
    :param down: Downsample the baselines before the calculation of the NPS - must be 2^x.
    :type down: int
    :param sample_length: The length of one sample from the time series in seconds.
    :type sample_length: float
    :param rms_cutoff: Only baselines with a fit rms below this values are included in the NPS calculation. This
            will overwrite the percentile argument, if it is not set to None.
    :type rms_cutoff: float
    :return: Tuple of (the mean NPS, the cleaned baselines).
    :rtype: (1D array of size (record_length/2 + 1), 2D array of size (percentile*nmbr_baselines, record_length))
    """

    # downsample the baselines
    if down > 1:
        baselines = np.mean(baselines.reshape(len(baselines),
                                              int(len(baselines[0])/down),
                                              down), axis=2)

    record_length = baselines.shape[1]

    # substract offset
    baselines -= np.mean(baselines[:, :int(record_length / 8)], axis=1, keepdims=True)

    # clean baselines
    if clean:

        if rms_baselines is None:
            print('Fitting baselines.')

            # choose baseline template
            if order_polynom == 2:
                bl_temp = baseline_template_quad
            elif order_polynom == 3:
                bl_temp = baseline_template_cubic
            else:
                bl_temp = None
                print('PLEASE USE POLYNOMIAL ORDER 2 OR 3!!')

            # fit polynomial coefficients
            coefficients = np.zeros([len(baselines), order_polynom + 1])
            t = np.linspace(0, record_length * sample_length, record_length)

            for i in range(len(baselines)):
                coefficients[i], _ = curve_fit(bl_temp, t, baselines[i])

            # throw high rms
            rms_baselines = []
            for bl, coeff in tqdm(zip(baselines, coefficients)):
                baseline_fit = bl_temp(t, *coeff)
                baseline_fit = np.array(baseline_fit)
                rms_baselines.append(np.sum((bl - baseline_fit) ** 2))

            # baselines_polynomials = np.array(baselines_polynomials)
            rms_baselines = np.array(rms_baselines)

        else:
            print('Using fitted baselines.')

        rms_means = np.array(np.percentile(rms_baselines, percentile))

        idx_keep = []
        for i, rms in enumerate(rms_baselines):
            if rms_cutoff is None:
                if rms < rms_means:
                    idx_keep.append(i)
            else:
                if rms < rms_cutoff:
                    idx_keep.append(i)

        baselines = baselines[idx_keep]

    # get mean NPS
    counter_baselines = 0
    mean_nps = np.zeros(int(record_length/2) + 1)

    for i, bl in enumerate(baselines):
        nps = get_nps(bl)

        counter_baselines += 1
        mean_nps += nps

    mean_nps /= counter_baselines

    return mean_nps, baselines
