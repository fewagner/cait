# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
from cait.fit._templates import baseline_template_quad, baseline_template_cubic
from scipy.optimize import curve_fit


# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------


def get_nps(x):
    x = np.fft.rfft(x)
    x = np.abs(x) ** 2
    return x


def fftnoise(f):
    f = np.array(f, dtype='complex')  # create array for frequencies
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi  # create random phases
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np + 1] *= phases
    f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])
    return np.fft.irfft(f)


def calculate_mean_nps(baselines, order_polynom=3, clean = True, record_length=16384):

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

        rms_means = np.array(np.percentile(rms_baselines, 90))

        idx_keep = []
        for i, rms in enumerate(rms_baselines):
            if rms < rms_means:
                idx_keep.append(i)

        baselines = baselines[idx_keep]

    # get mean NPS
    counter_baselines = 0
    mean_nps = np.zeros(8193)

    for i, bl in enumerate(baselines):
        bl = bl - np.mean(bl[:500])
        nps = get_nps(bl)

        counter_baselines += 1
        mean_nps += nps

    mean_nps /= counter_baselines

    return mean_nps, baselines