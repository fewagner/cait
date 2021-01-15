# imports

import numpy as np
from scipy import special


# form factors

def form():
    # TODO
    return 0

# bands and cut efficiency

def B_N():
    # TODO
    return 0

def cut_efficiency():
    # TODO
    return 0

# functions

def rho_N(E, L, Q, M, sigma_0, m_chi, m_N, form):
    # TODO
    return Q*m_N/M*dNdE(E, sigma_0, m_chi, m_N, form) * B_N() * cut_efficiency()


def dNdE(E, sigma_0, m_chi, m_N, form, rho_chi=0.3):
    # TODO
    return rho_chi / 2 / m_N / red_m(m_N, m_chi) ** 2 * sigma_0 * form * I(vmin(E, m_N, red_m(m_N, m_chi)))


def red_m(m_N, m_chi):
    # TODO
    return m_N * m_chi / (m_N + m_chi)


def I(v_min, z=np.sqrt(3 / 2) * 544 / 270, w=270, v_esc=544):
    # TODO
    return 1 / N() / eta() * np.sqrt(3 / 2 / np.pi / w ** 2) * (np.sqrt(np.pi) / 2 * (
                special.erf(x_min(v_min=v_min) - eta()) - special.erf(x_min(v_min=v_min) + eta())) - 2 * eta() * np.exp(
        -z ** 2))


def vmin(E, m_N, red_m):
    # TODO
    return np.sqrt(E * m_N / 2 / red_m ** 2)


def N(z=np.sqrt(3 / 2) * 544 / 270):
    # TODO
    return special.erf(z) - 2 / np.sqrt(np.pi) * z * np.exp(-z ** 2)


def eta(w=270, v_center=220 * 1.05):
    # TODO
    return 2 * v_center ** 2 / 2 / w ** 2


def x_min(v_min, w=270):
    # TODO
    return 3 * v_min ** 2 / 2 / w ** 2
