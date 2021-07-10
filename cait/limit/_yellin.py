# imports

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.optimize import brenth
from scipy.integrate import quad
from typing import Union, Iterable
from tqdm.auto import tqdm
from math import erf
from ..styles import use_cait_style, make_grid

# functions

@nb.njit
def I(m_N, speed_light, e_recoil, mu_N):
    """
    Integral of velocity distribution.
    """
    v_esc = 544.  # in km s^-1
    w = 220. * np.sqrt(3. / 2.)  # in km s^-1
    v_sun = 231.  # in km s^-1

    z = np.sqrt(3. / 2.) * v_esc / w
    eta = np.sqrt(3. / 2.) * v_sun / w
    norm = erf(z) - 2. / np.sqrt(np.pi) * z * np.exp(-(z ** 2))

    ret = 1. / (norm * eta) * np.sqrt(3. / (2. * np.pi * (w ** 2)))

    # ↓↓↓ from https://arxiv.org/pdf/hep-ph/9803295.pdf, eq. (A3)
    x_min = np.sqrt(3. * (speed_light ** 2) / 4.e12) * \
            np.sqrt(m_N * e_recoil / ((mu_N ** 2) * (w ** 2)))

    if x_min < z - eta:
        ret *= np.sqrt(np.pi) / 2. * (erf(x_min + eta) - erf(x_min - eta)) - 2. * eta * np.exp(-z ** 2)
    elif x_min < z + eta:
        ret *= np.sqrt(np.pi) / 2. * (erf(z) - erf(x_min - eta)) - (z + eta - x_min) * np.exp(-z ** 2)
    else:
        ret = 0.

    return ret


@nb.njit
def F(e_recoil_: Union[int, float],
      m_N,
      a_nucleons):
    """
    Form factor.
    """
    if e_recoil_ <= 0.:
        return 1.

    # q is the transferred momentum, hbar * c = 197.326960
    q = np.sqrt(2. * m_N * e_recoil_) / 197.326960
    F_a = .52  # in fm
    F_s = .9  # in fm
    F_c = 1.23 * a_nucleons ** (1. / 3.) - .6  # in fm

    R_0 = np.sqrt((F_c ** 2) + 7. / 3. * (np.pi ** 2) * (F_a ** 2) - 5. * (F_s ** 2))

    # ret = spherical_jn(1, q * R_0) / (q * R_0)
    ret = q * R_0
    ret = np.sin(ret) / ret ** 2 - np.cos(ret) / ret
    ret /= (q * R_0)
    ret *= 3. * np.exp(-.5 * (q ** 2) * (F_s ** 2))

    return ret


@nb.njit
def expected_interaction_rate(e_recoil: Union[int, float],
                              m_chi: Union[int, float],
                              a_nucleons: int,
                              ):
    """
    Yields the expected differential interaction rate between Dark Matter particles and a compound-nucleus.

    :param e_recoil: Given in keV.
    :param m_chi: Considered mass of the Dark Matter particle in GeV.
    :param a_nucleons: Number A of nucleons in the compound-nucleus.
    :return: dR/dE in kg^-1 d^-1 keV^-1.
    """
    # constants

    # speed of light in m s^-1
    speed_light = 299792458.
    # elementary charge in C
    e_charge = 1.60217662e-19
    # mass of the proton in GeV c^-2
    m_p = 0.9382720
    # mass of the nucleus
    m_N = m_p * a_nucleons
    # reduced mass of WIMP and proton
    mu_p = m_p * m_chi / (m_p + m_chi)
    # reduced mass of WIMP and nucleus
    mu_N = m_N * m_chi / (m_N + m_chi)
    # WIMP density on earth in GeV c^-2 cm^-3
    rho_chi = .3

    pre_factor = .5 * (speed_light ** 4) / (1.e12 * e_charge) * 1.e-40 * 86400. * (a_nucleons ** 2) / (m_chi * (mu_p ** 2))

    dN_per_dE = pre_factor  # * (a_nucleons ** 2)
    dN_per_dE *= rho_chi
    dN_per_dE *= (F(e_recoil, m_N, a_nucleons) ** 2)
    dN_per_dE *= I(m_N, speed_light, e_recoil, mu_N)

    return dN_per_dE


@nb.njit
def calc_c_0(x, mu):
    """
    Probability of the maximum gap size being smaller than a particular value of x.

    :param mu: The expected value of recoils in the gap.
    :param x: The value with that we compare the maximum gap size.
    """
    if mu < 0.:
        if x < 1.:
            return 0.
        else:
            return 1.

    elif x > mu:
        return 1.

    elif (x < .03 * mu and mu < 60.) \
            or (x < 1.8 and mu >= 60.) \
            or (x < 4. and mu > 700.):
        return 0.

    else:
        x1 = x
        mu1 = mu
        m = mu1 / x1
        A1 = np.exp(-x1)

        C00 = 1.
        if m >= 1.:
            C00 = 1. - (1. + mu1 - x1) * A1

        if m > 1.:
            C = 1.
            for k in range(2, int(m) + 1):
                B = k * x1 - mu1
                C /= k  # => 1/k!
                Delta = (A1 ** k) * (B ** (k - 1)) * C * (B - k)
                C00 += Delta
                if np.abs(Delta) < 1e-10:
                    break
        return C00


@nb.njit
def find_nearest(array: np.ndarray,
                 value: Union[int, float],
                 ) -> int:
    """
    Helper function, returns the index of array of the element closest to a given value.
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx - 1]) < np.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


@nb.njit
def expected_recoil_rate(e_recoil: Union[int, float],
                         m_chi: Union[int, float],
                         component_mass: list,
                         component_nucleons: list,
                         component_efficiencies: list,
                         exposure: Union[int, float],
                         ):
    """
    Yields the expected differential recoil rate between Dark Matter particles and all compound-nuclei.

    :param e_recoil: Given in keV.
    :param m_chi: Considered mass of the Dark Matter particle in GeV.
    :param component_mass: Mass of all atom nucleons in atomic units (u).
    :param component_nucleons: Number nucleons in all atoms.
    :param component_efficiencies: List of cut efficiencies of the atom nucleons. Must all have same length.
    :param exposure: The exposure in kg days.
    :return: dN/dE in keV^-1.
    """
    # constants

    component_mass *= .931494102  # convert 1 u to GeV
    # length_efficiency_array = len(component_efficiencies[0])

    dR_per_dE = 0.
    for i in range(len(component_mass)):
        mass_ratio = component_mass[i] / np.sum(component_mass)
        dN_per_dE = expected_interaction_rate(e_recoil, m_chi, component_nucleons[i])

        idx = find_nearest(component_efficiencies[i, :, 0], e_recoil)
        efficiency = component_efficiencies[i, idx, 1]

        dR_per_dE += exposure * mass_ratio * dN_per_dE * efficiency

    return dR_per_dE


# the limit calculation class

class Limit():
    """
    A class for the calculation of Yellin Maximum Gap Limits.

    :param exposure: The exposure in kg days.
    :type exposure: float
    :param component_mass: The masses of all components within the detector material.
    :type component_mass: array of floats
    :param component_nucleons: The number of nucleons of all components within the detector material.
    :type component_nucleons: array of ints
    :param confidence: Confidence level of limit.
    :type confidence: float
    """

    def __init__(self,
                 exposure: float,
                 component_mass: Union[int, float] = np.array([112.411, 183.84, 4 * 15.999]),
                 component_nucleons: Union[int, float] = np.array([112, 184, 16]),
                 confidence=0.9,
                 ):
        self.confidence = confidence
        self.exposure = exposure
        self.component_mass = component_mass
        self.component_nucleons = component_nucleons

    # public

    def import_efficiencies(self,
                            paths: list,
                            ):
        """
        Imports the efficiencies of all components from a given list of paths.

        The file has to be in an xy format, first column is the energies, second the efficiency.

        :param paths: list of string, the paths to the efficiencies of all components
        """

        self.component_efficiencies = []
        for p in paths:
            self.component_efficiencies.append(np.genfromtxt(p))
        self.component_efficiencies = np.array(self.component_efficiencies)

        print('Efficiencies from {} files imported.'.format(len(paths)))

    def import_observations(self, path: str):
        """
        Import the observed recoils of a detector.

        File is in x file format with one recoil energy per line.

        :param path: string, the path to the file with the observed energies
        """

        self.detector_AR = np.genfromtxt(path)

        print('Observed Recoils imported.')

    def detector_upper_limit(self,
                             m_chi: Union[int, float]):
        """
        Calculated the cross section upper limit for given detector observations and a given wimp mass.

        :param m_chi: The WIMP mass.
        """

        recoil_cdf, cdf_norm = self._calc_recoil_cum_spec(m_chi)

        detector_AR_uniform = np.zeros_like(self.detector_AR)

        for i in range(len(self.detector_AR)):
            detector_AR_uniform[i] = recoil_cdf[find_nearest(recoil_cdf[:, 0], self.detector_AR[i]), 1]

        return self._max_gap_upper_limit(detector_AR_uniform, cdf_norm), cdf_norm

    def calc_upper_limit(self,
                         ll: Union[int, float],
                         ul: Union[int, float],
                         steps: int,
                         plot: bool = True,
                         ):
        """
        Calculate the wimp cross section upper limit for a given detector.

        :param ll: float, The lowest mass to calculate a limit for.
        :param ul: float, The highest mass to calculate a limit for.
        :param steps: The number of values for that we want to calculate the limit.
        :param plot: bool, If True we plot the limit.
        :return x: The wimp masses.
        :return y: The exclusion cross sections.
        """
        x, y = [], []
        a = np.log(ul) - np.log(ll)

        counter = 0
        print('Start calculation.')
        for m_chi in tqdm(ll * np.exp(a * np.linspace(0., 1., steps))):
            counter += 1
            mu_ul, mu_old = self.detector_upper_limit(m_chi)

            x.append(m_chi)
            if mu_old > 0:
                y.append(mu_ul / mu_old * 1e-36)  # conversion from pb to cm^-2
            else:
                y.append(np.inf)

        if plot:
            print('Plot the limit.')
            plt.close()
            use_cait_style()
            plt.plot(x, y)
            make_grid()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("$m_\chi$ [GeV c${^-2}$]")
            plt.ylabel("$\sigma_0$ [cm$^{-2}$]")
            plt.show()

        self.x = x
        self.limit = y

        return x, y

    # plots

    def plot_wimp_distribution(self,
                               wimp_masses: list = [3., 5., 8.],
                               ):
        """
        Plot the expected WIMP energy distribution.

        :param wimp_masses: list of float, the wimp masses for that we want to do the plot
        """
        plt.close()
        use_cait_style()
        x = np.linspace(.1, 5., 100)
        for i, m in enumerate(wimp_masses):
            y = [expected_interaction_rate(e, m, 16) for e in x]
            plt.plot(x, y, label="$m_\chi$ = {} GeV".format(m), zorder=10 + 2*i)
        make_grid()
        plt.xlabel("Recoil Energy (keV)")
        plt.ylabel("dR/dE (kg$^{-1}$ d$^{-1}$ keV$^{-1}$)")
        plt.legend()
        plt.show()

    def plot_recoil_distribution(self,
                                 wimp_masses=[3., 5., 8.],
                                 ):
        """
        Plot the expected recoil spectra, including cut efficiencies.

        :param wimp_masses: list of float, the wimp masses for that we want to do the plot
        """
        x = np.linspace(.1, 20., 1000)

        plt.close()
        use_cait_style()
        for i, m in enumerate(wimp_masses):
            y = [expected_recoil_rate(e,
                                      m,
                                      self.component_mass,
                                      self.component_nucleons,
                                      self.component_efficiencies,
                                      self.exposure) for e in x]
            plt.plot(x, y, label="$m_\chi$ = {:.1} GeV".format(m), zorder=10 + 2*i)
        make_grid()
        plt.xlabel("Recoil Energy (keV)")
        plt.ylabel("dN/dE (keV$^{-1}$)")
        plt.legend()
        plt.show()

    def plot_observations(self,
                          bins=1000,
                          title=None,
                          ):
        """
        Plot the observed recoil histogram.

        :param bins: int, the number of bins in the histogram
        :param title: string, the title of the plot
        """
        plt.close()
        use_cait_style()
        plt.hist(self.detector_AR, bins=bins, zorder=15)
        make_grid()
        plt.xlabel("Recoil Energy (keV)")
        plt.ylabel("Counts")
        plt.title(title)
        plt.show()

    # private

    def _max_gap_0(self,
                   spec: Union[np.ndarray, Iterable, float, int],
                   ):
        """
        Returns largest gap in energy list.
        """
        spec_ = np.sort(spec)
        spec_ = np.concatenate(([0.], spec_))

        return np.max(spec_[1:] - spec_[:-1])

    def _max_gap_upper_limit(self,
                             obs_spec: Union[np.ndarray, int, float],
                             cdf_norm: float,
                             verb: bool = False,
                             ):
        """
        Finds the exclusion cross section in simplified units.

        :par obs_spec: List of observed recoil energies.
        :par c: Significance level of the limit calculation.
        """

        def f(mu_: Union[int, float]):
            """
            Minimizer for root finding to calculate the limit.
            """
            if mu_ < 2.:
                return 0.
            elif mu_ > 10e10:
                return 0.1
            else:
                return calc_c_0(self._max_gap_0(mu_ * obs_spec), mu_) - self.confidence

        a, b = 0., 0.
        initial_guess = 1.25
        while True:
            initial_guess *= 2.
            b = initial_guess
            if f(initial_guess) > 0.:
                break
            else:
                a = initial_guess

        if verb:
            print(f"found initial guess, searching in [{a}, {b}]")

        if cdf_norm > 0:
            sol = brenth(f, a, b, xtol=1e-4)
        else:
            sol = 0

        return sol

    def _calc_recoil_cum_spec(self,
                              m_chi: Union[int, float],
                              ):
        """
        Calculate the normed CDF of the expected wimp spectrum and the normalization constant.

        :param m_chi: The wimp mass.
        """
        steps = 200

        cdf = np.zeros((steps, 2))
        x = np.linspace(.01, 20., steps)
        y = np.zeros(x.size)

        for i in range(1, steps):
            y[i] = y[i - 1] + quad(expected_recoil_rate, a=x[i - 1], b=x[i], args=(m_chi,
                                                                                   self.component_mass,
                                                                                   self.component_nucleons,
                                                                                   self.component_efficiencies,
                                                                                   self.exposure),
                                   limit=20,
                                   full_output=1)[0]

        cdf[:, 0] = x
        norm = y[-1]
        if norm > 0:
            cdf[:, 1] = y / norm

        return cdf, norm
