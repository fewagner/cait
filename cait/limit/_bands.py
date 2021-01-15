# imports

import numpy as np
from scipy.special import erf


# functions

def mean_electron_band(E, L0, L1, L2, L3):
    """
    Fit model for the mean of the election recoil band.

    :param E: Recoil energy.
    :type E: Scalar or numpy array
    :param L0: Fit parameter
    :type L0: Scalar
    :param L1: parameter
    :type L1: Scalar
    :param L2: parameter
    :type L2: Scalar
    :param L3: parameter
    :type L3: Scalar
    :return: Value of the the electron recoil band at point E
    :rtype: Scalar or numpy array
    """

    return (L0 * E + L1 * E ** 2) * (1 - L2 * np.exp(-E / L3))


def mean_gamma_band(E, L0, L1, L2, L3, Q1, Q2):
    """
    Fit model for the mean of the gamma recoil band.

    :param E: Recoil energy.
    :type E: Scalar or numpy array
    :param L0: Fit parameter
    :type L0: Scalar
    :param L1: parameter
    :type L1: Scalar
    :param L2: parameter
    :type L2: Scalar
    :param L3: parameter
    :type L3: Scalar
    :param Q1: First quentching factor
    :type Q1: Scalar
    :param Q2: Second quentching factor
    :type Q2: Scalar
    :return: Value of the the electron recoil band at point E
    :rtype: Scalar or numpy array
    """

    return mean_electron_band(E * (Q1 + E * Q2), L0, L1, L2, L3)


def light_resolution(L, sigma_L0, S1, S2):
    """
    Resolution of the light detector

    :param L: Recoil energy of the light detector
    :type L: Scalar or numpy array
    :param sigma_L0: Uncertainty due to baseline noise in the light detector
    :type sigma_L0: Scalar
    :param S1: Describes the finite energy required to produce a single photon
    :type S1: Scalar
    :param S2: Describes the position dependence in the crystal
    :type S2: Scalar
    :return: Value of the resolution of the light detector at point L
    :rtype: Scalar or numpy array
    """

    return np.sqrt(sigma_L0 ** 2 + S1 * L + S2 * L ** 2)


def phonon_resolution(E, sigma_P0, sigma_P1):
    """
    Resolution of the phonon detector

    :param L: Recoil energy of the phonon detector
    :type L: Scalar or numpy array
    :param sigma_L0: Uncertainty due to baseline noise in the phonon detector
    :type sigma_L0: Scalar
    :param S1: Describes the finite energy required to produce a single phonon
    :type S1: Scalar
    :return: Value of the resolution of the phonon detector at point E
    :rtype: Scalar or numpy array
    """

    return np.sqrt(sigma_P0 ** 2 + sigma_P1 * E)


def differential_mean_electron_band(E, L0, L1, L2, L3):
    """
    Fit model for the differential mean of the election recoil band.

    :param E: Recoil energy.
    :type E: Scalar or numpy array
    :param L0: Fit parameter
    :type L0: Scalar
    :param L1: parameter
    :type L1: Scalar
    :param L2: parameter
    :type L2: Scalar
    :param L3: parameter
    :type L3: Scalar
    :return: Value of the the electron recoil band at point E
    :rtype: Scalar or numpy array
    """

    return (L0 * E + L1 * E ** 2) * L2 * np.exp(-E / L3) / L3 + (L0 + 2 * L1 * E) * (1 - L2 * np.exp(-E / L3))


def differential_mean_gamma_band(E, L0, L1, L2, L3, Q1, Q2):
    """
    TODO

    :param E:
    :type E:
    :param L0:
    :type L0:
    :param L1:
    :type L1:
    :param L2:
    :type L2:
    :param L3:
    :type L3:
    :param Q1:
    :type Q1:
    :param Q2:
    :type Q2:
    :return:
    :rtype:
    """

    return differential_mean_electron_band(E * (Q1 + E * Q2), L0, L1, L2, L3)


def total_resolution_electron_band(E, sigma_L0, S1, S2, L0, L1, L2, L3, sigma_P0, sigma_P1):
    """
    TODO

    :param E:
    :type E:
    :param sigma_L0:
    :type sigma_L0:
    :param S1:
    :type S1:
    :param S2:
    :type S2:
    :param L0:
    :type L0:
    :param L1:
    :type L1:
    :param L2:
    :type L2:
    :param L3:
    :type L3:
    :param sigma_P0:
    :type sigma_P0:
    :param sigma_P1:
    :type sigma_P1:
    :return:
    :rtype:
    """

    return np.sqrt(light_resolution(mean_electron_band(E, L0, L1, L2, L3), sigma_L0, S1, S2) +
                   differential_mean_electron_band(E, L0, L1, L2, L3) * phonon_resolution(E, sigma_P0, sigma_P1))


def total_resolution_gamma_band(E, sigma_L0, S1, S2, L0, L1, L2, L3, Q1, Q2, sigma_P0, sigma_P1):
    """
    TODO

    :param E:
    :type E:
    :param sigma_L0:
    :type sigma_L0:
    :param S1:
    :type S1:
    :param S2:
    :type S2:
    :param L0:
    :type L0:
    :param L1:
    :type L1:
    :param L2:
    :type L2:
    :param L3:
    :type L3:
    :param Q1:
    :type Q1:
    :param Q2:
    :type Q2:
    :param sigma_P0:
    :type sigma_P0:
    :param sigma_P1:
    :type sigma_P1:
    :return:
    :rtype:
    """

    return np.sqrt(light_resolution(mean_gamma_band(E, L0, L1, L2, L3, Q1, Q2), sigma_L0, S1, S2) +
                   differential_mean_gamma_band(E, L0, L1, L2, L3, Q1, Q2) * phonon_resolution(E, sigma_P0, sigma_P1))


def excess_light(E, L, Elamp, Elwidth, Eldec, L0, L1, L2, L3, sigma_L0, S1, S2):
    """
    TODO

    :param E:
    :type E:
    :param L:
    :type L:
    :param Elamp:
    :type Elamp:
    :param Elwidth:
    :type Elwidth:
    :param Eldec:
    :type Eldec:
    :param L0:
    :type L0:
    :param L1:
    :type L1:
    :param L2:
    :type L2:
    :param L3:
    :type L3:
    :param sigma_L0:
    :type sigma_L0:
    :param S1:
    :type S1:
    :param S2:
    :type S2:
    :return:
    :rtype:
    """

    sle = light_resolution(mean_electron_band(E, L0, L1, L2, L3), sigma_L0, S1, S2)

    return Elamp * np.exp(-E / Eldec) * (1 / 2 / Elwidth * np.exp(-L / Elwidth + sle ** 2 / 2 / Elwidth ** 2) * (
                1 + erf(L / np.sqrt(2) / sle - sle / np.sqrt(2) / Elwidth)))

def mean_nuclear_band(E, L0, L1, eps, QFx, fx, lambx):
    """
    TODO

    :param E:
    :type E:
    :param L0:
    :type L0:
    :param L1:
    :type L1:
    :param QFx:
    :type QFx:
    :param fx:
    :type fx:
    :param lambx:
    :type lambx:
    :return:
    :rtype:
    """

    return (L0*E + L1*E**2)*eps*QFx*(1 + fx*np.exp(-E/lambx))

def get_quench(which):
    """
    TODO

    :param which:
    :type which:
    :return:
    :rtype:
    """
    if which == 'O':
        QFx, Fx, lambx = 0.0739, 0.708, 567.1
    elif which == 'Ca':
        QFx, Fx, lambx = 0.0556, 0.1887, 801.3
    elif which == 'W':
        QFx, Fx, lambx = 0.0196, 0, np.inf
    return QFx, Fx, lambx

