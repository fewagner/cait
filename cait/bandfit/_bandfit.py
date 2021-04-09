# imports

import numpy as np
import scipy
import scipy.stats
import scipy.interpolate
import scipy.integrate
from scipy.special import erfinv
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from ._likelihood import *
from ._bandfunctions import *
from ..styles import use_cait_style, make_grid
from tqdm.auto import tqdm, trange


# class

class Bandfit():

    def __init__(self,
                 values_module_independent,
                 lbounds_module_independent=None,
                 ubounds_module_independent=None,
                 fixed_module_independent=None,
                 values_nuclear=None,
                 lbounds_nuclear=None,
                 ubounds_nuclear=None,
                 fixed_nuclear=None,
                 values_gamma=None,
                 lbounds_gamma=None,
                 ubounds_gamma=None,
                 fixed_gamma=None,
                 values_beta=None,
                 lbounds_beta=None,
                 ubounds_beta=None,
                 fixed_beta=None,
                 values_inelastic=None,
                 lbounds_inelastic=None,
                 ubounds_inelastic=None,
                 fixed_inelastic=None,
                 ):

        # detector independent required parameters

        names = ["np_decay", "np_fract", "L0", "L1", "sigma_l0", "S1", "S2", "el_amp", "el_decay", "el_width",
                 "sigma_p0", "sigma_p1", "E_p0", "E_p1", "E_fr", "E_dc", "L_lee", "QF_y", "QF_ye", "eps", "kg_d", "thr"]

        values = list(values_module_independent)

        if lbounds_module_independent is None:
            lbounds = [0.0, 0.0, 0.5, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.5,
                       -0.03,
                       0.3, 0.0, 0.0]
        else:
            lbounds = list(lbounds_module_independent)

        if ubounds_module_independent is None:
            ubounds = [10.0, 1.0, 1.5, 0.1, 0.3, 1.0, 0.1, 100.0, 25.0, 10.0, 0.03, 0.05, 1000.0, 20.0, 1.0e6, 10.0,
                       0.3,
                       1.5, 0.03, 1.0, 1.0e6, 1.0e6]
        else:
            ubounds = list(ubounds_module_independent)

        if fixed_module_independent is None:
            fixed = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]
        else:
            fixed = list(fixed_module_independent)

        # nuclear recoil band parameters
        if values_nuclear is not None:
            self.nmbr_nuclei = len(values_nuclear)
            if len(values_nuclear[0]) != 5:
                raise ValueError('Each nuclear band need exactly 5 parameters!')
            for i in range(self.nmbr_nuclei):
                names.append("QF_nuc{}".format(i))
                names.append("es_nuc{}_f".format(i))
                names.append("es_nuc{}_lb".format(i))
                names.append("nc_nuc{}_p0".format(i))
                names.append("nc_nuc{}_p1".format(i))
                for v, l, u, f in zip(values_nuclear[i], lbounds_nuclear[i], ubounds_nuclear[i], fixed_nuclear[i]):
                    values.append(v)
                    lbounds.append(l)
                    ubounds.append(u)
                    fixed.append(f)
        else:
            self.nmbr_nuclei = 0

        # gamma peak parameters
        if values_gamma is not None:
            self.nmbr_gamma = len(values_gamma)
            if len(values_gamma[0]) != 2:
                raise ValueError('Each gamma band need exactly 2 parameters!')
            for i in range(self.nmbr_gamma):
                names.append("FG_{}_C".format(i))
                names.append("FG_{}_M".format(i))
                for v, l, u, f in zip(values_gamma[i], lbounds_gamma[i], ubounds_gamma[i], fixed_gamma[i]):
                    values.append(v)
                    lbounds.append(l)
                    ubounds.append(u)
                    fixed.append(f)
        else:
            self.nmbr_gamma = 0

        # beta peak parameters
        if values_beta is not None:
            self.nmbr_beta = len(values_beta)
            if len(values_beta[0]) != 3:
                raise ValueError('Each beta band need exactly 3 parameters!')
            for i in range(self.nmbr_beta):
                names.append("B_{}_C".format(i))
                names.append("B_{}_M".format(i))
                names.append("B_{}_D".format(i))
                for v, l, u, f in zip(values_beta[i], lbounds_beta[i], ubounds_beta[i], fixed_beta[i]):
                    values.append(v)
                    lbounds.append(l)
                    ubounds.append(u)
                    fixed.append(f)
        else:
            self.nmbr_beta = 0

        # inelastic band parameters
        if values_inelastic is not None:
            self.nmbr_inelastic = len(values_inelastic)
            if len(values_inelastic[0]) != 5:
                raise ValueError('Each inelastic band need exactly 5 parameters!')
            for i in range(self.nmbr_inelastic):
                names.append("IE_{}_M".format(i))
                names.append("IE_{}_S".format(i))
                names.append("IE_{}_E".format(i))
                names.append("IE_{}_p0".format(i))
                names.append("IE_{}_p1".format(i))
                for v, l, u, f in zip(values_inelastic[i], lbounds_inelastic[i], ubounds_inelastic[i],
                                      fixed_inelastic[i]):
                    values.append(v)
                    lbounds.append(l)
                    ubounds.append(u)
                    fixed.append(f)
        else:
            self.nmbr_inelastic = 0

        # turn into numpy arrays

        self.names = names
        self.values = np.array(values)
        self.lbounds = np.array(lbounds)
        self.ubounds = np.array(ubounds)
        self.fixed = np.array(fixed)

        print('Bandfit Instance created.')

    def load_files(self,
                   path_xy,
                   path_ncal,
                   path_cuteff,
                   region_of_interest,
                   ):
        """
        TODO

        :param path_xy:
        :type path_xy:
        :param path_ncal:
        :type path_ncal:
        :param path_cuteff:
        :type path_cuteff:
        :param region_of_interest:
        :type region_of_interest:
        :return:
        :rtype:
        """

        def out_of_bound_index(array,
                               roi):  # find indices in array, where values are outside acceptance regions (ranges)
            index = []
            for i in range(array.shape[0]):
                if array[i, 0] < roi[0]:
                    index.append(i)
                elif array[i, 0] > roi[1]:
                    index.append(i)
                elif array[i, 1] < roi[2]:
                    index.append(i)
                elif array[i, 1] > roi[3]:
                    index.append(i)
            return index

        bck = np.loadtxt(path_xy, skiprows=4)
        bck = np.delete(bck, out_of_bound_index(bck, region_of_interest), axis=0)
        self.bck = bck[np.argsort(bck[:, 0])]

        ncal = np.loadtxt(path_ncal, skiprows=4)
        ncal = np.delete(ncal, out_of_bound_index(ncal, region_of_interest), axis=0)
        self.ncal = ncal[np.argsort(ncal[:, 0])]

        self.xy = np.append(bck, ncal, axis=0)

        self.xy = self.xy[np.argsort(self.xy[:, 0])]
        self.xy[:, 1] = self.xy[:, 0] * self.xy[:, 1]

        self.cuteff = np.loadtxt(path_cuteff, skiprows=4)
        self.cutf = scipy.interpolate.Akima1DInterpolator(self.cuteff[:, 0], self.cuteff[:, 1])

        self.cateffarr = self.cutf(self.xy[:, 0])
        self.region_of_interest = region_of_interest

        print('Files loaded.')

    def minimize(self,
                 method='Nelder-Mead',
                 verb=True,
                 maxiter=None,
                 workers=-1,
                 optstart=False
                 ):
        """
        TODO

        :param method:
        :type method:
        :return:
        :rtype:
        """

        valuesred = reduceparvalues(self.values, self.fixed)
        lboundsred = reduceparvalues(self.lbounds, self.fixed)
        uboundsred = reduceparvalues(self.ubounds, self.fixed)

        if method == 'Nelder-Mead':
            if maxiter is None:
                maxiter = 2e4

            x0 = valuesred

            if optstart:
                print('Getting optimal start values.')
                info = {}
                if verb:
                    print('{}   {}'.format('Nfeval', 'f(X)'))
                    info['Nfeval'] = 0
                minresult = scipy.optimize.differential_evolution(wrappernoint,
                                                                  scipy.optimize.Bounds(lboundsred, uboundsred),
                                                                  args=(
                                                                      self.values, self.fixed, self.lbounds,
                                                                      self.ubounds,
                                                                      self.xy, self.cateffarr, self.region_of_interest,
                                                                      self.nmbr_nuclei, self.nmbr_gamma, self.nmbr_beta,
                                                                      self.nmbr_inelastic, info),
                                                                  strategy='randtobest1bin',
                                                                  workers=workers,
                                                                  maxiter=maxiter,
                                                                  popsize=8,
                                                                  tol=0.001,
                                                                  updating='deferred')

                x0 = minresult.x

            print('Start optimization.')
            info = {}
            if verb:
                print('{}   {}'.format('Nfeval', 'f(X)'))
                info['Nfeval'] = 0
            minresult = scipy.optimize.minimize(wrappernoint,
                                                x0=x0,
                                                args=(self.values, self.fixed, self.lbounds, self.ubounds,
                                                      self.xy, self.cateffarr, self.region_of_interest,
                                                      self.nmbr_nuclei, self.nmbr_gamma, self.nmbr_beta,
                                                      self.nmbr_inelastic, info),
                                                method='Nelder-Mead',
                                                options={  # 'maxiter': 1e100,
                                                    'maxiter': maxiter,
                                                    # 'maxfev': 1e100,
                                                    'maxfev': maxiter,
                                                    'xatol': 1e-10,
                                                    'fatol': 1e-10,
                                                    'adaptive': True})
        elif method == 'differential_evolution':
            if maxiter is None:
                maxiter = 1000
            print('Start optimization.')
            info = {}
            if verb:
                print('{}   {}'.format('Nfeval', 'f(X)'))
                info['Nfeval'] = 0
            minresult = scipy.optimize.differential_evolution(wrappernoint,
                                                              scipy.optimize.Bounds(lboundsred, uboundsred),
                                                              args=(self.values, self.fixed, self.lbounds, self.ubounds,
                                                                    self.xy, self.cateffarr, self.region_of_interest,
                                                                    self.nmbr_nuclei, self.nmbr_gamma, self.nmbr_beta,
                                                                    self.nmbr_inelastic, info),
                                                              strategy='randtobest1bin',
                                                              workers=workers,
                                                              maxiter=maxiter,
                                                              popsize=8,
                                                              tol=0.001,
                                                              updating='deferred')
        else:
            raise NotImplementedError('Method not implemented.')

        print('Likelihood optimization complete.')
        print(minresult, "\n")

        self.minresult = minresult.x

    def plot_bck(self,
                 binwidth=0.05,
                 lowErange=[0., 0.5],
                 lowEbinw=0.01,
                 plot_bands=True,
                 grid_step=0.001,
                 all_in_one=False,
                 upper_acceptance: float = 0.5,
                 lower_acceptance: float = 0.005,
                 ):
        """
        TODO

        plot background & neutron calibration data, and show signal survival probability

        :param binwidth:
        :type binwidth:
        :param lowErange:
        :type lowErange:
        :param lowEbinw:
        :type lowEbinw:
        :return:
        :rtype:
        """

        if plot_bands:
            try:
                band_means = []
                band_sigmas = []
                for i in range(self.nmbr_nuclei):
                    grid = np.arange(self.region_of_interest[0], self.region_of_interest[1], grid_step)
                    band_means.append(self._get_band_mean(nucleus=i,
                                                          energy=grid) / grid)
                    band_sigmas.append(self._get_band_sigma(nucleus=i,
                                                            energy=grid) / grid)
            except AttributeError:
                raise AttributeError('Before you can plot bands, you need to call minimize!')

            # get limits acceptance region
            lower_limit, upper_limit = self._get_acceptance_region(grid, upper_acceptance, lower_acceptance) / grid

        def plt_hist_high(axis, data):
            axis.hist(data[:, 0],
                      bins=np.arange(self.region_of_interest[0], self.region_of_interest[1], binwidth),
                      zorder=15)
            axis.set_yscale('log', nonpositive='clip')
            axis.set_xlabel("Energy / keV")
            axis.set_ylabel("counts / " + str(binwidth) + " keV")
            make_grid(axis)

        def plt_hist_low(axis, data):
            axis.hist(data[:, 0], bins=np.arange(lowErange[0], lowErange[1], lowEbinw), zorder=15)
            axis.set_yscale('log', nonpositive='clip')
            axis.set_xlabel("Energy / keV")
            axis.set_ylabel("counts / " + str(lowEbinw) + " keV")
            make_grid(axis)

        def plt_scatter_high(axis, data):
            axis.scatter(data[:, 0], data[:, 1], s=5, zorder=15)
            if plot_bands:
                for i in range(self.nmbr_nuclei):
                    axis.plot(grid, band_means[i] - band_sigmas[i], linewidth=1, color='C' + str(i + 1),
                              linestyle='dotted', zorder=20)
                    axis.plot(grid, band_means[i], color='C' + str(i + 1), linewidth=1, zorder=20)
                    axis.plot(grid, band_means[i] + band_sigmas[i], linewidth=1, color='C' + str(i + 1),
                              linestyle='dotted', zorder=20)
                axis.fill_between(grid, y1=lower_limit, y2=upper_limit, color='yellow', zorder=0)
            axis.set_xlabel("Energy / keV")
            axis.set_ylabel("Light Yield")
            axis.set_ylim(-10, 10)
            make_grid(axis)

        def plt_scatter_low(axis, data):
            axis.scatter(data[:, 0], data[:, 1], s=5, zorder=15)
            if plot_bands:
                for i in range(self.nmbr_nuclei):
                    axis.plot(grid, band_means[i] - band_sigmas[i], linewidth=1, color='C' + str(i + 1),
                              linestyle='dotted', zorder=20)
                    axis.plot(grid, band_means[i], color='C' + str(i + 1), linewidth=1, zorder=20)
                    axis.plot(grid, band_means[i] + band_sigmas[i], linewidth=1, color='C' + str(i + 1),
                              linestyle='dotted', zorder=20)
                axis.fill_between(grid, y1=lower_limit, y2=upper_limit, color='yellow', zorder=0)
            axis.set_xlabel("Energy / keV")
            axis.set_ylabel("Light Yield")
            axis.set_ylim(-10, 10)
            axis.set_xlim(lowErange[0], lowErange[1])
            make_grid(axis)

        if all_in_one:

            plt.close()
            use_cait_style()
            fig, ax = plt.subplots(2, 2, figsize=(13, 8))
            plt_hist_high(ax[0, 0], data=self.bck)
            plt_hist_low(ax[0, 1], data=self.bck)
            plt_scatter_high(ax[1, 0], data=self.bck)
            plt_scatter_low(ax[1, 1], data=self.bck)
            fig.suptitle("Background Data")
            plt.show()

        else:
            for f_handle in [plt_hist_high, plt_hist_low, plt_scatter_high, plt_scatter_low]:
                plt.close()
                use_cait_style()
                fig, ax = plt.subplots(1, 1)
                f_handle(ax, data=self.bck)
                ax.set_title("Background Data")
                plt.show()

    def plot_ncal(self,
                  binwidth=0.05,
                  lowErange=[0., 0.5],
                  lowEbinw=0.01,
                  plot_bands=True,
                  grid_step=0.001,
                  all_in_one=False,
                  upper_acceptance: float = 0.5,
                  lower_acceptance: float = 0.005,
                  ):
        """
        TODO

        :param binwidth:
        :type binwidth:
        :param lowErange:
        :type lowErange:
        :param lowEbinw:
        :type lowEbinw:
        :return:
        :rtype:
        """

        if plot_bands:
            try:
                band_means = []
                band_sigmas = []
                grid = np.arange(self.region_of_interest[0], self.region_of_interest[1], grid_step)
                for i in range(self.nmbr_nuclei):
                    band_means.append(self._get_band_mean(nucleus=i,
                                                          energy=grid) / grid)
                    band_sigmas.append(self._get_band_sigma(nucleus=i,
                                                            energy=grid) / grid)
            except AttributeError:
                raise AttributeError('Before you can plot bands, you need to call minimize!')

            # get limits acceptance region
            lower_limit, upper_limit = self._get_acceptance_region(grid, upper_acceptance, lower_acceptance) / grid

        def plt_hist_high(axis, data):
            axis.hist(data[:, 0],
                      bins=np.arange(self.region_of_interest[0], self.region_of_interest[1], binwidth),
                      zorder=15)
            axis.set_yscale('log', nonpositive='clip')
            axis.set_xlabel("Energy / keV")
            axis.set_ylabel("counts / " + str(binwidth) + " keV")
            make_grid(axis)

        def plt_hist_low(axis, data):
            axis.hist(data[:, 0], bins=np.arange(lowErange[0], lowErange[1], lowEbinw), zorder=15)
            axis.set_yscale('log', nonpositive='clip')
            axis.set_xlabel("Energy / keV")
            axis.set_ylabel("counts / " + str(lowEbinw) + " keV")
            make_grid(axis)

        def plt_scatter_high(axis, data):
            axis.scatter(data[:, 0], data[:, 1], s=5, zorder=15)
            if plot_bands:
                for i in range(self.nmbr_nuclei):
                    axis.plot(grid, band_means[i] - band_sigmas[i], linewidth=1, color='C' + str(i + 1),
                              linestyle='dotted', zorder=20)
                    axis.plot(grid, band_means[i], color='C' + str(i + 1), linewidth=1, zorder=20)
                    axis.plot(grid, band_means[i] + band_sigmas[i], linewidth=1, color='C' + str(i + 1),
                              linestyle='dotted', zorder=20)
                axis.fill_between(grid, y1=lower_limit, y2=upper_limit, color='yellow', zorder=0)
            axis.set_xlabel("Energy / keV")
            axis.set_ylabel("Light Yield")
            axis.set_ylim(-10, 10)
            make_grid(axis)

        def plt_scatter_low(axis, data):
            axis.scatter(data[:, 0], data[:, 1], s=5, zorder=15)
            if plot_bands:
                for i in range(self.nmbr_nuclei):
                    axis.plot(grid, band_means[i] - band_sigmas[i], linewidth=1, color='C' + str(i + 1),
                              linestyle='dotted', zorder=20)
                    axis.plot(grid, band_means[i], color='C' + str(i + 1), linewidth=1, zorder=20)
                    axis.plot(grid, band_means[i] + band_sigmas[i], linewidth=1, color='C' + str(i + 1),
                              linestyle='dotted', zorder=20)
                axis.fill_between(grid, y1=lower_limit, y2=upper_limit, color='yellow', zorder=0)
            axis.set_xlabel("Energy / keV")
            axis.set_ylabel("Light Yield")
            axis.set_ylim(-10, 10)
            axis.set_xlim(lowErange[0], lowErange[1])
            make_grid(axis)

        if all_in_one:

            plt.close()
            use_cait_style()
            fig, ax = plt.subplots(2, 2, figsize=(13, 8))
            plt_hist_high(ax[0, 0], data=self.ncal)
            plt_hist_low(ax[0, 1], data=self.ncal)
            plt_scatter_high(ax[1, 0], data=self.ncal)
            plt_scatter_low(ax[1, 1], data=self.ncal)
            fig.suptitle("Neutron Calibration Data")
            plt.show()

        else:
            for f_handle in [plt_hist_high, plt_hist_low, plt_scatter_high, plt_scatter_low]:
                plt.close()
                use_cait_style()
                fig, ax = plt.subplots(1, 1)
                f_handle(ax, data=self.ncal)
                ax.set_title("Neutron Calibration Data")
                plt.show()

    def plot_survival_prob(self):
        """
        TODO

        :return:
        :rtype:
        """

        plt.close()
        use_cait_style()
        xnew = np.geomspace(self.region_of_interest[0], self.region_of_interest[1], num=100, endpoint=True)
        plt.plot(self.cuteff[:, 0], self.cuteff[:, 1], 'o', xnew, self.cutf(xnew), '-', zorder=15)
        make_grid()
        plt.xlim(self.region_of_interest[0:2])
        plt.xlabel("Energy / keV")
        plt.ylabel("Survival Probability")
        plt.xscale('log', nonpositive='clip')
        plt.title("Signal Survival Probability")
        plt.show()

    def nuclear_efficiency(self,
                           energy_array: float,
                           nucleus: int,
                           upper_acceptance: float = 0.5,
                           lower_acceptance: float = 0.005,
                           step_size: float = 10e-5,
                           include_cut_efficiency: bool = True,
                           ):
        # TODO

        # get mean of band
        mean_nuclear_band = self._get_band_mean(nucleus, energy_array)  # mean of nuclear recoil band

        # get sigma of band
        sigma_nuclear_band = self._get_band_sigma(nucleus, energy_array)

        # get limits acceptance region
        lower_limit, upper_limit = self._get_acceptance_region(energy_array, upper_acceptance, lower_acceptance)

        # do integral over gauss with mean and sigma, from min to max light
        nuclear_cut_eff = np.zeros(len(energy_array))
        print('Calculating integral over light for nucleus {}.'.format(nucleus))
        for i in trange(len(energy_array)):
            grid = np.arange(lower_limit[i], upper_limit[i], step=step_size)
            nuclear_cut_eff[i] = np.trapz(y=norm(loc=mean_nuclear_band[i],
                                                 scale=sigma_nuclear_band[i]).pdf(grid),
                                          x=grid)

        # multiply with energy dependent cut efficiency
        if include_cut_efficiency:
            nuclear_cut_eff = self.cutf(energy_array) * nuclear_cut_eff

        # return
        return nuclear_cut_eff

    def get_accepted_events(self,
                            upper_acceptance: float = 0.5,
                            lower_acceptance: float = 0.005,
                            ):
        # TODO
        try:

            accepted_events = []

            print('Calculating accepted events.')
            for (e, l) in zip(self.bck[:, 0], self.bck[:, 1]):
                ll, ul = self._get_acceptance_region(energy=e,
                                                     upper_acceptance=upper_acceptance,
                                                     lower_acceptance=lower_acceptance,
                                                     )
                if l > ll and l < ul:
                    accepted_events.append(e)

        except:
            raise AttributeError('You need to call minimize, before you can get accepted events!')

        return np.array(accepted_events), np.array(self.bck[:, 0])

    # private

    def _get_band_mean(self, nucleus, energy):
        p = expandparvalues(self.values, self.minresult, self.fixed)
        # get mean of band
        peb = p[0:4]
        eps = p[19]
        pnb = p[
              22 + nucleus * 5:22 + nucleus * 5 + 3]  # neutron band parameters; 0 = QF; 1 = es_f; 1 = es_lb
        mnn = meanenn(energy, peb)
        mean_nuclear_band = meann(energy, pnb, eps, mnn)  # mean of nuclear recoil band
        return mean_nuclear_band

    def _get_band_slope(self, nucleus, energy):
        p = expandparvalues(self.values, self.minresult, self.fixed)
        # get mean of band
        peb = p[0:4]
        eps = p[19]
        pnb = p[
              22 + nucleus * 5:22 + nucleus * 5 + 3]  # neutron band parameters; 0 = QF; 1 = es_f; 1 = es_lb
        slope_nuclear_band = slopen(energy, pnb, peb, eps)
        return slope_nuclear_band

    def _get_band_sigma(self, nucleus, energy):
        p = expandparvalues(self.values, self.minresult, self.fixed)
        # get mean of band
        peb = p[0:4]
        plr = p[4:7]
        ppr = p[10:12]
        eps = p[19]
        pnb = p[22 + nucleus * 5:22 + nucleus * 5 + 3]  # neutron band parameters; 0 = QF; 1 = es_f; 1 = es_lb
        thr = p[21]
        mnn = meanenn(energy, peb)
        mean_nuclear_band = meann(energy, pnb, eps, mnn)  # mean of nuclear recoil band
        slope_nuclear_band = slopen(energy, pnb, peb, eps)

        # get sigma of band
        sigma_nuclear_band = np.sqrt(bressq(energy, mean_nuclear_band, ppr, plr, thr, slope_nuclear_band))
        return sigma_nuclear_band

    def _get_acceptance_region(self,
                               energy,
                               upper_acceptance: float = 0.5,
                               lower_acceptance: float = 0.005,
                               ):
        # TODO
        p = expandparvalues(self.values, self.minresult, self.fixed)
        peb = p[0:4]
        plr = p[4:7]
        ppr = p[10:12]
        eps = p[19]
        thr = p[21]
        mnn = meanenn(energy, peb)

        # get upper limit acceptance region
        pnb_temp = p[22 + 0 * 5:22 + 0 * 5 + 3]
        meanlight_temp = meann(energy, pnb_temp, eps, mnn)
        slope_nuclear_band_temp = slopen(energy, pnb_temp, peb, eps)
        sigma_temp = np.sqrt(bressq(energy, meanlight_temp, ppr, plr, thr, slope_nuclear_band_temp))

        upper_limit = meanlight_temp + ((erfinv(2.0 * upper_acceptance - 1.0) * np.sqrt(2)) * sigma_temp)

        # get lower limit acceptance region
        pnb_temp = p[22 + (self.nmbr_nuclei - 1) * 5:22 + (self.nmbr_nuclei - 1) * 5 + 3]
        meanlight_temp = meann(energy, pnb_temp, eps, mnn)
        slope_nuclear_band_temp = slopen(energy, pnb_temp, peb, eps)
        sigma_temp = np.sqrt(bressq(energy, meanlight_temp, ppr, plr, thr, slope_nuclear_band_temp))

        lower_limit = meanlight_temp + ((erfinv(2.0 * lower_acceptance - 1.0) * np.sqrt(2)) * sigma_temp)

        return lower_limit, upper_limit
