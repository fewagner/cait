# imports

import numpy as np
from scipy import special
from scipy.interpolate import interp1d
from scipy.special import spherical_jn


# calculation class

class LimitCalculation:
    """
    TODO
    """
    def __init__(self, Q, m_N, atom_counts, A, Ep=None, Lp=None):
        """
        TODO

        :param Q:
        :type Q:
        :param m_N:
        :type m_N:
        :param atom_counts:
        :type atom_counts:
        :param A:
        :type A:
        :param Ep:
        :type Ep:
        :param Lp:
        :type Lp:
        """
        self.Q = Q
        self.m_N = m_N  # list of all masses
        self.atom_counts = atom_counts  # list of the number of each nuclei in the material
        self.M = np.sum([m * n for m, n in zip(m_N, atom_counts)])
        self.A = A  # list of all nuclear mass numbers
        self.Ep = Ep  # data
        self.Lp = Lp  # data
        self.cut_efficiency = None
        self.nuclei_cut_eff = None

    def add_data(self, Ep=None, Lp=None):
        """
        TODO

        :param Ep:
        :type Ep:
        :param Lp:
        :type Lp:
        :return:
        :rtype:
        """
        self.Ep = Ep
        self.Lp = Lp

    def add_cut_efficiency(self, E, cut_efficiency):
        """
        TODO

        :param E:
        :type E:
        :param cut_efficiency:
        :type cut_efficiency:
        :return:
        :rtype:
        """
        self.cut_efficiency = interp1d(x=E, y=cut_efficiency)

    def add_nuclei_cut_eff(self, E, nuclei_cut_eff):
        """
        TODO

        :param E:
        :type E:
        :param nuclei_cut_eff:
        :type nuclei_cut_eff:
        :return:
        :rtype:
        """
        # TODO
        # the nuclei cut eff is a list with a cut eff for each nuclei
        self.nuclei_cut_eff = [interp1d(E, nce) for nce in nuclei_cut_eff]

    def get_wimp_spectrum(self, E, sigma_0, m_chi):  # units ?
        """
        TODO

        :param E:
        :type E:
        :param sigma_0:
        :type sigma_0:
        :param m_chi:
        :type m_chi:
        :return:
        :rtype:
        """
        res = np.zeros(len(E))
        if self.nuclei_cut_eff is not None:
            cut_eff = self.nuclei_cut_eff
        elif self.cut_efficiency is not None:
            cut_eff = [self.cut_efficiency for i in range(len(self.m_N))]
        else:
            cut_eff = None
        for i in range(len(self.m_N)):
            res += self._rho_N(E, sigma_0, m_chi, self.m_N[i], self.A[i], self.atom_counts[i],
                               cut_efficiency=cut_eff[i])
        return res

    def calc_yellin_limit(self):
        raise NotImplementedError("This method is not yet implemented.")

    # private --------------------------------------------------------------

    def _helm_form_factor(self, E_R, m_N, A, s=1):
        """
        TODO

        :param E_R:
        :type E_R:
        :param m_N:
        :type m_N:
        :param A:
        :type A:
        :param s:
        :type s:
        :return:
        :rtype:
        """
        R0 = self._R0(A)
        q = np.sqrt(2 * m_N * E_R)
        return 3 * spherical_jn(1, q * R0) / (q * R0) * np.exp(-q ** 2 * s ** 2 / 2)

    def _R0(self, A):
        """
        TODO

        :param A:
        :type A:
        :return:
        :rtype:
        """
        a = 0.52
        s = 0.9
        c = 1.23 * A ** (1 / 3) - 0.6
        return np.sqrt(c ** 2 + 7 / 3 * np.pi ** 2 * a ** 2 - 5 * s ** 2)

    def _rho_N(self, E, sigma_0, m_chi, m_N, A, atom_counts, cut_efficiency=None):
        """
        TODO

        :param E:
        :type E:
        :param sigma_0:
        :type sigma_0:
        :param m_chi:
        :type m_chi:
        :param m_N:
        :type m_N:
        :param A:
        :type A:
        :param atom_counts:
        :type atom_counts:
        :param cut_efficiency:
        :type cut_efficiency:
        :return:
        :rtype:
        """
        # cut_efficiency must be array of same length as E or None
        res = self.Q * atom_counts * m_N / self.M * self._dNdE(E, sigma_0, m_chi, m_N, A)
        if cut_efficiency is not None:
            res *= cut_efficiency(E)
        return res

    def _dNdE(self, E, sigma_0, m_chi, m_N, A, rho_chi=0.3):
        """
        TODO

        :param E:
        :type E:
        :param sigma_0:
        :type sigma_0:
        :param m_chi:
        :type m_chi:
        :param m_N:
        :type m_N:
        :param A:
        :type A:
        :param rho_chi:
        :type rho_chi:
        :return:
        :rtype:
        """
        return rho_chi / 2 / m_N / self._red_m(m_N, m_chi) ** 2 * sigma_0 * self._helm_form_factor(E, m_N,
                                                                                                   A) ** 2 * self._I(
            self._vmin(E, m_N, self._red_m(m_N, m_chi)))

    def _red_m(self, m_N, m_chi):
        """
        TODO

        :param m_N:
        :type m_N:
        :param m_chi:
        :type m_chi:
        :return:
        :rtype:
        """
        return m_N * m_chi / (m_N + m_chi)

    def _I(self, v_min, z=np.sqrt(3 / 2) * 544 / 270, w=270, v_esc=544):
        """
        TODO

        :param v_min:
        :type v_min:
        :param z:
        :type z:
        :param w:
        :type w:
        :param v_esc:
        :type v_esc:
        :return:
        :rtype:
        """
        return 1 / self._N() / self._eta() * np.sqrt(3 / 2 / np.pi / w ** 2) * (np.sqrt(np.pi) / 2 * (
                special.erf(self._x_min(v_min=v_min) - self._eta()) - special.erf(
            self._x_min(v_min=v_min) + self._eta())) - 2 * self._eta() * np.exp(
            -z ** 2))

    def _vmin(self, E, m_N, red_m):
        """
        TODO

        :param E:
        :type E:
        :param m_N:
        :type m_N:
        :param red_m:
        :type red_m:
        :return:
        :rtype:
        """
        return np.sqrt(E * m_N / 2 / red_m ** 2)

    def _N(self, z=np.sqrt(3 / 2) * 544 / 270):
        """
        TODO

        :param z:
        :type z:
        :return:
        :rtype:
        """
        return special.erf(z) - 2 / np.sqrt(np.pi) * z * np.exp(-z ** 2)

    def _eta(self, w=270, v_center=220 * 1.05):
        """
        TODO

        :param w:
        :type w:
        :param v_center:
        :type v_center:
        :return:
        :rtype:
        """

        return np.sqrt(3 / 2) * v_center / w

    def _x_min(self, v_min, w=270):
        """
        TODO

        :param v_min:
        :type v_min:
        :param w:
        :type w:
        :return:
        :rtype:
        """

        return np.sqrt(3 / 2) * v_min / w
