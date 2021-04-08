import math
import numpy as np
import scipy.stats
import numba
from math import erf

@numba.njit
def pol1(E, p):
    """
    Linear fit-component.
    """
    return p[0] + p[1] * E

@numba.njit
def gaussn(x, mean, sigma):
    """
    Gauss function.
    """
    return (1 / np.sqrt(2.0 * math.pi)) / sigma * np.exp(-(x-mean)**2 / 2.0 / sigma**2)

@numba.njit
def peak(E, p, res):
    """
    Gausspeak fit-component.
    """
    return gaussn(E, p[1], res) * p[0]

@numba.njit
def expspec(E, p):
    """
    Exponential decay fit-component.
    """
    return p[0] * np.exp(-E / p[1])

@numba.njit
def gbspec(energy, p, sigma):
    """
    Energy spectrum of gamma/betapeaks 0 = FB_C; 1 = FB_M; 2 = FB_D.
    """
    E=energy-p[1]
    QmE = p[2]-E
    z=1.0/(sigma * np.sqrt(2.0))
    QmEz=QmE * z
    Ez=E * z
    retval= p[0] * ((1.0 - E/p[2]) * (erf(QmEz) - erf(-Ez)) + (np.exp(-QmEz * QmEz )-np.exp(-Ez * Ez ))/(z * np.sqrt(math.pi) * p[2])) /p[2]
    if retval >=0.0:
        return retval
    else:
        return 0.0

@numba.njit
def ielspec(E, p, preslower, presupper):
    """
    Energy spectrum of inelastics 0 = IE_S; 1 = IE_E; 2 = IE_p0; 3 = IE_p1.
    """
    lowersigmasqrt2=np.sqrt(2.0)*preslower
    uppersigmasqrt2=np.sqrt(2.0)*presupper
    return (p[2] + (E-p[0])*p[3]) * 0.5 * ( erf( (E-p[0])/lowersigmasqrt2 ) - erf( (E-p[1])/uppersigmasqrt2 ) )

@numba.njit
def pressq(E, p, thr):
    """
    Energy-dependent resolution of phonon detector 0 = sigma_p0; 1 = sigma_p1.
    """
    return (p[0] ** 2) + (p[1] ** 2) * ( (E ** 2) - (thr ** 2) )

@numba.njit
def lressq(L, p):
    """
    Light-dependent resolution of light detector 0 = sigma_l0; 1 = S1; 2 = S2.
    """
    return p[0] ** 2 + p[1] * L + p[2] * (L ** 2)

@numba.njit
def bressq(E, L, pp, pl, thr, slope):
    """
    Energy- and light-dependent resolution of band.
    """
    return pressq(E, pp, thr) * (slope ** 2) + lressq(L, pl)

@numba.njit
def meane(E, p):
    """
    Meanline for the e-Band 0 = L0; 1 = L1; 2 = np_fract; 3 = np_decay.
    """
    return (p[0] * E + p[1] * (E ** 2) ) * (1 - p[2] * np.exp( -E / p[3]))

@numba.njit
def slopee(E, p):
    """
    Slope for the e-Band 0 = L0; 1 = L1; 2 = np_fract; 3 = np_decay.
    """
    return (((p[0] * E + p[1] * E ** 2) * p[2] * np.exp(-E / p[3])) / p[3]) + (p[0] + 2 * p[1] * E) * (1 - p[2] * np.exp(-E / p[3]))

@numba.njit
def meanenn(E, p):
    """
    Meanline for the e-Band without non-proportionality 0 = L0; 1 = L1.
    """
    return (p[0] * E + p[1] * (E ** 2) )

@numba.njit
def slopeenn(E, p):
    """
    Slope for the e-Band without non-proportionality 0 = L0; 1 = L1.
    """
    return (p[0] + 2 * p[1] * E)

@numba.njit
def meang(E, p, pe):
    """
    Meanline for the e-Band 0 = QF_y; 1 = QF_ye.
    """
    qfg = p[0] + E * p[1]
    return meane(E * qfg, pe)

@numba.njit
def slopeg(E, p, pe):
    """
    Meanline for the e-Band 0 = QF_y; 1 = QF_ye.
    """
    qfg = p[0] + E * p[1]
    return p[1]*slopee(E * qfg, pe)

@numba.njit
def meann(E, p, eps, mnn):
    """
    Meanline for the neutron-band 0 = QF; 1 = es_f; 1 = es_lb.
    """
    return (eps * p[0] * mnn) * (1 + (p[1] * np.exp(-E / p[2])))

@numba.njit
def slopen(E, p, pe, eps):
    """
    Slope for the neutron-band 0 = QF; 1 = es_f; 1 = es_lb.
    """
    return (eps * p[0]) * ((pe[0] + 2 * pe[1] * E) * (1 + (p[1] * np.exp(-E / p[2]))) - (pe[0] * E + pe[1] * E ** 2) * p[1] * np.exp(-E / p[2]) / p[2])

@numba.njit
def meanb(E, p, pe, pg):
    """
    Meanline for the beta-bands p = FB_M.
    """
    ee = E-p
    meanlight=meane(ee, pe)
    meanlightg=meang(p, pg, pe)
    if meanlight<0.0:
        return meanlightg
    else:
        return meanlight+meanlightg

@numba.njit
def slopeb(E, p, pe):
    """
    Slope for the beta-bands p = FB_M.
    """
    ee = E-p
    return slopee(ee, pe)

@numba.njit
def meanie(E, p, pe, pn, eps):
    """
    Meanline for the inelastic-bands p = IE_S.
    """
    ee = E-p
    meanlightnn = meanenn(ee, pe)
    meanlightn=meann(ee, pn, eps, meanlightnn)
    meanlight=meane(p, pe)
    return meanlight+meanlightn

@numba.njit
def slopeie(E, p, pe, pn, eps):
    """
    Slope for the inelastic-bands p = IE_S.
    """
    ee = E-p
    return slopen(ee, pn, pe, eps)

@numba.njit
def probability(L, meanlight, resolution):
    """
    Calls the gaussn function.
    """
    ret = gaussn(L, meanlight, resolution)
    if ret < 10e-8:
        ret = 10e-8
    return ret

@numba.njit
def probexli(E, L, p, meanlight, res, ressq):
    """
    Excess light distribution; 0 = el_amp; 1 = el_decay; 2 = el_width.
    """
    L -= meanlight
    return p[0] * np.exp(-E / p[1]) * (1 / (2 * p[2])) * np.exp(-(L / p[2]) + (1 / 2) * (ressq / (p[2]**2))) * (1 + erf(L / (np.sqrt(2) * res) - (res / (np.sqrt(2) * p[2]))))

@numba.njit
def intoverlight(lower, upper, meanlight, resolution):
    """
    Returns the integral over the light plane, weighted with a given recoil band.
    """
    return 0.5 * (erf((upper - meanlight) / (np.sqrt(2) * resolution)) - erf((lower - meanlight) / (np.sqrt(2) * resolution)))

@numba.njit
def exliintoverlight(E, lower, upper, p, meanlight, res, ressq):
    """
    Excess light integral; 0 = el_amp; 1 = el_decay; 2 = el_width.
    """
    sigmasqrt2 = res * np.sqrt(2)
    return (((p[0] * np.exp(-(E / p[1]) - ((upper - meanlight) / p[2])) * ((np.exp((upper - meanlight) / p[2]) * erf((upper - meanlight) / sigmasqrt2)) + (np.exp(ressq / (2 * (p[2])**2)) * (-1 + erf((ressq - (p[2] * (upper - meanlight))) / (sigmasqrt2 * p[2])))))) - (p[0] * np.exp(-(E / p[1]) - ((lower - meanlight) / p[2])) * ((np.exp((lower - meanlight) / p[2]) * erf((lower - meanlight) / sigmasqrt2)) + (np.exp(ressq / (2 * (p[2])**2)) * (-1 + erf((ressq - (p[2] * (lower - meanlight))) / (sigmasqrt2 * p[2]))))))) / 2)
