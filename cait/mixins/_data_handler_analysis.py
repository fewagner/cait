# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
from collections import Counter
from scipy.stats import norm
from ..cuts import rate_cut, testpulse_stability, controlpulse_stability
from ..calibration import energy_calibration, light_yield_correction, energy_calibration_linear, energy_calibration_tree



# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class AnalysisMixin(object):
    """
    A Mixin to the DataHandler class with methods for the typical raw data analysis.
    """

    def calc_resolution(self,
                        cpe_factors,
                        ph_intervals,
                        fit_gauss=True,
                        of_filter=False,
                        sev_fit=False,
                        use_tp=False,
                        verb=False):
        """
        This function calculates the resolution of a detector from the testpulses.
        the ph intervals need to include only one low-energetic peak in the spectrum
        Calculation can take a gauss fit or take the sample std deviation
        Ph can be determined as peak of event, of peak, sev fit height
        Needed attributes: self.path_h5, self.nmbr_channels
        Optional needed attributes:

        :param cpe_factors: list of length nmbr_channels, the cpe factors to calculate the resolution
        :param ph_intervals: list that contains nmbr_channels lists, the upper and lower bounds for
            the resolution calculation
        :param fit_gauss:
        :param of_filter:
        :param sev_fit:
        :param use_tp:
        :param verb: bool, verbal feedback
        :return: 1D array of length nmbr_channels, the calculated resolutions
        """

        print('Calculating resolution.')

        h5 = h5py.File(self.path_h5, 'r+')
        mus = []
        resolutions = []
        ph_intervals = np.array(ph_intervals)
        if ph_intervals.shape != (self.nmbr_channels, 2):
            raise KeyError('The shape of the ph intervals must be (self.nmbr_channels, 2)!')

        if of_filter and sev_fit:
            raise KeyError('Choose maximal one of of_filter or sev_fit!')

        for c in range(self.nmbr_channels):
            if use_tp:
                tphs = np.array(h5['testpulses']['mainpar'])[c, :, 0]
                tpas = np.array(h5['testpulses']['testpulseamplitude'])
                naming = 'Testpulses'
            # choose the correct quantity for resolution calculation
            elif of_filter:
                tphs = np.array(h5['events']['of_ph'])[c]
                tpas = np.array(h5['events']['true_ph'])[c]
                naming = 'Optimum Filter'
            elif sev_fit:
                tphs = np.array(h5['events']['sev_fit_par'])[c, :, 0]
                tpas = np.array(h5['events']['true_ph'])[c]
                naming = 'Standard Event Fit'
            else:
                tphs = np.array(h5['events']['mainpar'])[c, :, 0]
                tpas = np.array(h5['events']['true_ph'])[c]
                naming = 'Pulse Heights'

            tpas = tpas[tphs > ph_intervals[c][0]]
            tphs = tphs[tphs > ph_intervals[c][0]]
            tpas = tpas[tphs < ph_intervals[c][1]]
            tphs = tphs[tphs < ph_intervals[c][1]]

            true_ph = Counter(tpas).most_common(1)[0][0]
            tphs = tphs[tpas == true_ph]

            if fit_gauss:
                mu, sigma = norm.fit(tphs)
            else:
                mu = np.mean(tphs)
                sigma = np.std(tphs)

            mus.append(cpe_factors[c] * mu)
            resolutions.append(cpe_factors[c] * mu / true_ph * sigma)

            print('Resolution channel {}: {} eV (mean {} keV, calculated with {})'.format(c, resolutions[c] * 1000,
                                                                                          mus[c], naming))
        h5.close()
        return np.array(resolutions), np.array(mus)

    def calc_rate_cut(self, interval=10, significance=3, min=0, max=60):
        # TODO

        h5 = h5py.File(self.path_h5, 'r+')
        hours = np.array(h5['events']['hours'])

        flag = rate_cut(hours * 60, interval, significance, min, max)

        h5['events'].require_dataset(name='rate_cut',
                                     shape=(flag.shape),
                                     dtype=bool)
        h5['events']['rate_cut'][...] = flag
        h5.close()

    def calc_controlpulse_stability(self, channel, significance=3, max_gap=0.5):
        # TODO

        f = h5py.File(self.path_h5, 'r+')

        cphs = f['controlpulses']['pulse_height'][channel]
        hours_cp = f['controlpulses']['hours']

        hours_ev = f['events']['hours']
        #cphs, hours_cp, hours_ev, significance=3, max_gap=1
        flag_ev, flag_cp = controlpulse_stability(cphs, hours_cp, hours_ev,
                                         significance=significance, max_gap=max_gap)

        f['events'].require_dataset(name='controlpuls_stability',
                                    shape=(self.nmbr_channels, len(flag_ev)),
                                    dtype=bool)
        f['events']['controlpuls_stability'][channel, ...] = flag_ev

        f['controlpulses'].require_dataset(name='controlpuls_stability',
                                        shape=(self.nmbr_channels, len(flag_cp)),
                                        dtype=bool)
        f['controlpulses']['controlpuls_stability'][channel, ...] = flag_cp
        f.close()

    def calc_testpulse_stability(self, channel, significance=3, noise_level=0.005, max_gap=0.5):
        # TODO

        f = h5py.File(self.path_h5, 'r+')

        tpas = f['testpulses']['testpulseamplitude']
        tphs = f['testpulses']['mainpar'][channel, :, 0]  # 0 is the mainpar index for pulseheight
        hours_tp = f['testpulses']['hours']
        hours_ev = f['events']['hours']

        flag_ev, flag_tp = testpulse_stability(tpas, tphs, hours_tp, hours_ev,
                                         significance=significance, noise_level=noise_level, max_gap=max_gap)

        f['events'].require_dataset(name='testpulse_stability',
                                    shape=(self.nmbr_channels, len(flag_ev)),
                                    dtype=bool)
        f['events']['testpulse_stability'][channel, ...] = flag_ev

        f['testpulses'].require_dataset(name='testpulse_stability',
                                        shape=(self.nmbr_channels, len(flag_tp)),
                                        dtype=bool)
        f['testpulses']['testpulse_stability'][channel, ...] = flag_tp
        f.close()

    def calc_calibration(self,
                         starts_saturation,  #
                         cpe_factor,
                         max_dist=1,  # in hours
                         smoothing_factor=0.95,
                         exclude_tpas=[],
                         plot=False,
                         only_stable=False,
                         cut_flag=None,
                         linear_with_uncertainty=False,
                         tree=False
                         ):

        if tree and linear_with_uncertainty:
            raise KeyError("Choose either linear_with_uncertainty or tree!")

        # TODO
        f = h5py.File(self.path_h5, 'r+')

        evhs = np.array(f['events']['mainpar'][:, :, 0])
        ev_hours = np.array(f['events']['hours'])
        tpas = np.array(f['testpulses']['testpulseamplitude'])
        tphs = np.array(f['testpulses']['mainpar'][:, :, 0])
        tp_hours = np.array(f['testpulses']['hours'])

        if only_stable:
            stable = np.array(f['testpulses']['testpulse_stability'], dtype=bool)
        else:
            stable = np.ones([self.nmbr_channels, len(tpas)], dtype=bool)
        if cut_flag is not None:
            stable = np.logical_and(stable, cut_flag)

        f['events'].require_dataset(name='recoil_energy',
                                    shape=(self.nmbr_channels, len(ev_hours)),
                                    dtype=float)
        if linear_with_uncertainty or tree:
            f['events'].require_dataset(name='recoil_energy_sigma',
                                        shape=(self.nmbr_channels, len(ev_hours)),
                                        dtype=float)

        for channel in range(self.nmbr_channels):
            print('Energy Calibration for channel ', channel)
            if linear_with_uncertainty:
                f['events']['recoil_energy'][channel, ...], \
                f['events']['recoil_energy_sigma'][channel, ...] = energy_calibration_linear(evhs=evhs[channel],
                                                                                             ev_hours=ev_hours,
                                                                                             tphs=tphs[channel, stable[
                                                                                                 channel]],
                                                                                             tpas=tpas[stable[channel]],
                                                                                             tp_hours=tp_hours[
                                                                                                 stable[channel]],
                                                                                             start_saturation=
                                                                                             starts_saturation[
                                                                                                 channel],
                                                                                             max_dist=max_dist,
                                                                                             cpe_factor=cpe_factor,
                                                                                             exclude_tpas=exclude_tpas,
                                                                                             plot=plot,
                                                                                             )
            elif tree:
                f['events']['recoil_energy'][channel, ...], \
                f['events']['recoil_energy_sigma'][channel, ...] = energy_calibration_tree(evhs=evhs[channel],
                                                                                             ev_hours=ev_hours,
                                                                                             tphs=tphs[channel, stable[
                                                                                                 channel]],
                                                                                             tpas=tpas[stable[channel]],
                                                                                             tp_hours=tp_hours[
                                                                                                 stable[channel]],
                                                                                             start_saturation=
                                                                                             starts_saturation[
                                                                                                 channel],
                                                                                             cpe_factor=cpe_factor,
                                                                                             exclude_tpas=exclude_tpas,
                                                                                             plot=plot,
                                                                                             )
            else:
                f['events']['recoil_energy'][channel, ...] = energy_calibration(evhs=evhs[channel],
                                                                                ev_hours=ev_hours,
                                                                                tphs=tphs[channel, stable[channel]],
                                                                                tpas=tpas[stable[channel]],
                                                                                tp_hours=tp_hours[stable[channel]],
                                                                                start_saturation=starts_saturation[channel],
                                                                                max_dist=max_dist,
                                                                                cpe_factor=cpe_factor,
                                                                                smoothing_factor=smoothing_factor,
                                                                                exclude_tpas=exclude_tpas,
                                                                                plot=plot,
                                                                                )
        f.close()


    def calc_light_correction(self,
                              scintillation_efficiency,
                              channels_to_calibrate=[0],
                              light_channel=1):
        # TODO

        f = h5py.File(self.path_h5, 'r+')
        recoil_energy = np.array(f['events']['recoil_energy'])

        f['events'].require_dataset(name='corrected_energy',
                                    shape=(len(channels_to_calibrate), len(recoil_energy[0])),
                                    dtype=float)

        for channel in channels_to_calibrate:
            f['events']['recoil_energy'][channel, ...] = light_yield_correction(phonon_energy=recoil_energy[channel],
                                                                                light_energy=recoil_energy[light_channel],
                                                                                scintillation_efficiency=scintillation_efficiency)

        f.close()