# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
from ..fit._templates import baseline_template_quad, baseline_template_cubic
from scipy.optimize import curve_fit
from scipy.stats import norm, uniform

# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------


def get_nps(x):
    """
    Calculates the Noise Power Spectrum (NPS) of a given array

    :param x: time series, 1D numpy array of size N
    :return: NPS, 1D numpy array of size N/2 + 1
    """
    x = np.fft.rfft(x)
    x = np.abs(x) ** 2
    return x


def noise_function(nps):
    """
    Simulates the function f from CC-Noise Algo
    with a given Noise Power Spectrum (NPS)
    see arXiv:1006.3289, eq (4) and (5)

    :param f: NPS, real valued 1D array of size N/2 + 1
    :return: Noise Baselines, real valued 1D array of size N
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
                 lamb=0.01,
                 verb=False):
    """
    Simulation of a noise baseline, according to Carretoni Cremonesi: arXiv:1006.3289

    :param nmbr_noise: int > 0, number of noise baselines to simualte
    :param nps: 1D array of odd size, Noise power spectrum of the baselines,
        e.g. generated with scipy.fft.rfft()
    :param lamb: integer > 0, parameter of the method (overlap between )
    :param verb: bool, verbal feedback about progress of code
    :return: 2D array of size (nmbr_noise, 2*(len(nps)-1)), the simulated baselines
    """

    T = (len(nps) - 1)*2

    alpha = lamb*T

    noise = np.zeros((nmbr_noise, T))

    for i in range(nmbr_noise):
        if verb and i%100==0:
            print('Baselines simulated: ', i)
        t = 0
        while t < T:
            noise[i] += norm.rvs() * np.roll(noise_function(nps), t)
            t = int(t - np.log(1 - uniform.rvs())/lamb)

    noise /= np.sqrt(alpha)  # normalization constant

    return noise

def calculate_mean_nps(baselines,
                       order_polynom=3,
                       clean=True,
                       percentile=50):
    """
    Calculates the mean Noise Power Spectrum (mNPS) of a set of baselines,
    after cleaning them from artifacts with a polynomial fit

    :param baselines: 2D array of size (nmbr_baselines, record_length),
        the baselines for the mNPS
    :param order_polynom: 2 or 3, the order of the fitted polynomial
    :param clean: boolean, if True the baselines are cleaned from artifacts with a poly fit
    :param percentile: float between 0 and 1, the percentile of the Fit RMS that is still used
        for the calculation of the mNPS
    :return: (1D array of size (record_length/2 + 1) - the mNPS,
            2D array of size (percentile*nmbr_baselines, record_length) - the cleaned baselines)
    """

    record_length = len(baselines[0])

    # clean baselines
    if clean:

        # choose baseline template
        if order_polynom == 2:
            bl_temp = baseline_template_quad
        elif order_polynom == 3:
            bl_temp = baseline_template_cubic
        else:
            bl_temp = None
            print('PLEASE USE POLYNOMIAL ORDER 2 OR 3!!')

        for bl in baselines:
            bl = bl - np.mean(bl)

        # fit polynomial coefficients
        coefficients = np.zeros([len(baselines), order_polynom + 1])
        t = np.linspace(0, record_length * 0.00004, record_length)

        for i in range(len(baselines)):
            coefficients[i], _ = curve_fit(bl_temp, t, baselines[i])

        # throw high rms
        baselines_polynomials = []
        rms_baselines = []
        for bl, coeff in zip(baselines, coefficients):
            baseline_fit = bl_temp(t, *coeff)
            baseline_fit = np.array(baseline_fit)
            # baselines_polynomials.append(baseline_fit)
            rms_baselines.append(np.sum((bl - baseline_fit) ** 2))

        # baselines_polynomials = np.array(baselines_polynomials)
        rms_baselines = np.array(rms_baselines)

        rms_means = np.array(np.percentile(rms_baselines, percentile))

        idx_keep = []
        for i, rms in enumerate(rms_baselines):
            if rms < rms_means:
                idx_keep.append(i)

        baselines = baselines[idx_keep]

    # get mean NPS
    counter_baselines = 0
    mean_nps = np.zeros(int(record_length/2) + 1)

    for i, bl in enumerate(baselines):
        bl = bl - np.mean(bl[:500])
        nps = get_nps(bl)

        counter_baselines += 1
        mean_nps += nps

    mean_nps /= counter_baselines

    return mean_nps, baselines
