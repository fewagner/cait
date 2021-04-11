# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
from collections import Counter
from scipy.stats import norm
from ..cuts import rate_cut, testpulse_stability, controlpulse_stability
from ..calibration import light_yield_correction
from tqdm.auto import tqdm
from ..calibration import PulserModel


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class AnalysisMixin(object):
    """
    A Mixin to the DataHandler class with methods for the typical raw data analysis.
    """

    def calc_resolution(self,
                        ph_intervals: list,
                        pec_factors: list = None,
                        fit_gauss: bool = True,
                        of_filter: bool = False,
                        sev_fit: bool = False,
                        use_tp: bool = False,
                        ):
        """
        This function calculates the resolution of a detector from the testpulses or from simulated events.

        The pulse height intervals need to include only one low-energetic peak in the spectrum.
        Calculation can take a gauss fit of the peak or take the sample std deviation.
        The pulse height can be determined with the raw pulse height, the pulse height after filtering with the
        optimum filter and as estimation with an standard event fit. To use the raw pulse height on simulated events,
        set the bool arguments of_filter, sev_fit and use_tp all to false.

        :param ph_intervals: The upper and lower bounds of the peak in the pulse height spectrum to calculate
            the resolution.
        :type ph_intervals: list of float 2-tuples
        :param pec_factors: The PEC factor in keV for each channel to calculate the resolution in energy. This is the
            linearization of
            the energy calibration, calculated by dividing the energy of the calibration peak by its height in Volt.
            Multiplication of a volt pulse height with this factor gives a rough estimation of the recoil energy. The
            exact energy calibration uses an energy-dependent PEC factor. If this argument is None, the output is in mV
            instead.
        :type pec_factors: list of floats or None
        :param fit_gauss: If this argument is true, the width of the peak is estimated with a gauss fit rather than with
            the sample standard deviation.
        :type fit_gauss: bool
        :param of_filter: If this argument is true, the filtered pulse height is taken for the energy estimation.
            This option is prioritized against sev_fit.
        :type of_filter: bool
        :param sev_fit: If this argument is true, the standard event fit is taken for the pulse height estimation.
        :type sev_fit: bool
        :param use_tp: If this argument is true, the raw pulse height of a test pulse is taken for the resolution
            estimation. This option is prioritized against of_filter and sev_fit.
        :type use_tp: bool
        :return: The calculated resolutions and the mean values of the peaks.
        :rtype: tuple of two numpy arrays

        >>> resolutions, mus = dh.calc_resolution(cpe_factors=[6, 12], ph_intervals=[(0.25,0.35), (0.25,0.35)], use_tp=True)
        Calculating resolution.
        Resolution channel 0: 13.197 eV (mean 1.747 keV, calculated with Testpulses)
        Resolution channel 1: 25.183 eV (mean 3.619 keV, calculated with Testpulses)
        """

        print('Calculating resolution.')

        with h5py.File(self.path_h5, 'r+') as h5:
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

                if pec_factors is not None:
                    mus.append(pec_factors[c] * mu)
                    resolutions.append(pec_factors[c] * mu / true_ph * sigma)

                    print('Resolution channel {}: {:.3} eV (mean {:.3} keV, calculated with {})'.format(c, resolutions[
                        c] * 1000,
                                                                                                        mus[c], naming))
                else:
                    mus.append(mu)
                    resolutions.append(mu / true_ph * sigma)

                    print('Resolution channel {}: {:.3} mV (mean {:.3} V, calculated with {})'.format(c, resolutions[
                        c] * 1000,
                                                                                                      mus[c], naming))
        return np.array(resolutions), np.array(mus)

    def calc_rate_cut(self, interval: float = 10, significance: float = 3, min: float = 0, max: float = 60):
        """
        Calculate a rate cut on the events.

        The rate cut assignes a bool value to each event, that tells if the event is in a region with normal or
        anomalous rate. For this, we measure the event rate in every interval (typically 10 minutes) and exclude
        intervals from the analysis with a rate that is not within a certain number of standard deviations of the
        average rate per interval. We exclude intervals with rate that exceed a certain maximal value or subceed a
        certain minimal value from the calculation of the average rate. This renders the calculation robust against
        intervals in which the TES is superconducting (no events) or triggers only in the noise, e.g. due to
        warming up of the cryostat.

        :param interval: The interval length in minutes that is compared.
        :type interval: float
        :param significance: Rates that are by more than this factor times the standard deviation of the
            rates away from the average rate are excluded.
        :type significance: float
        :param min: Rates that are lower than this value are excluded from the calculation of the average rate.
        :type min: float
        :param max: Rates that are higher than this value are excluded from the calculation of the average rate.
        :type max: float
        """

        with h5py.File(self.path_h5, 'r+') as h5:
            hours = np.array(h5['events']['hours'])
            hours_cp = np.array(h5['controlpulses']['hours'])
            hours_tp = np.array(h5['testpulses']['hours'])

            flag_ev, flag_cp, flag_tp = rate_cut(hours * 60, hours_cp, hours_tp,
                                                 interval, significance, min, max)

            h5['events'].require_dataset(name='rate_cut',
                                         shape=(flag_ev.shape),
                                         dtype=bool)
            h5['events']['rate_cut'][...] = flag_ev
            h5['controlpulses'].require_dataset(name='rate_cut',
                                                shape=(flag_cp.shape),
                                                dtype=bool)
            h5['controlpulses']['rate_cut'][...] = flag_cp
            h5['testpulses'].require_dataset(name='rate_cut',
                                             shape=(flag_tp.shape),
                                             dtype=bool)
            h5['testpulses']['rate_cut'][...] = flag_tp

    def calc_controlpulse_stability(self, channel: int, significance: float = 3, max_gap: float = 0.5, lb: float = 0,
                                    ub: float = 100):
        """
        Do a stability cut on the control pulses.

        In the stability cut we assign a boolean value to each event, if it is within a stable region or not. Stable
        regions are defined as the interval between two stable control pulses. A control pulse is stable, if its height
        is within a certain number of standard deviations of the average control pulse height. Single outlying control
        pulses are ignored. Control pulse heights higher than a certain maximal value or lower than a certain minimal
        value are also ignored. If for a duration of more than a certain interval no control pulses appear in the data,
        the region is automatically counted as unstable.

        :param channel: The number of the channel on that we calculate the cut in the HDF5 file.
        :type channel: int
        :param significance: Pulse heights further than this factor times the pulse height standard deviation away from
            the mean pulse height are counted as unstable.
        :type significance: float
        :param max_gap: Intervals longer than this value (in minutes) without control pulses are automatically counted
            as unstable.
        :type max_gap: float
        :param lb: Pulse heights lower than this value are ignored.
        :type lb: float
        :param ub: Pulse heights higher than this value are ignored.
        :type ub: float
        """

        with h5py.File(self.path_h5, 'r+') as f:
            cphs = f['controlpulses']['pulse_height'][channel]
            hours_cp = f['controlpulses']['hours']

            hours_ev = f['events']['hours']
            # cphs, hours_cp, hours_ev, significance=3, max_gap=1
            flag_ev, flag_cp = controlpulse_stability(cphs, hours_cp, hours_ev,
                                                      significance=significance, max_gap=max_gap,
                                                      lb=lb, ub=ub)

            f['events'].require_dataset(name='controlpuls_stability',
                                        shape=(self.nmbr_channels, len(flag_ev)),
                                        dtype=bool)
            f['events']['controlpuls_stability'][channel, ...] = flag_ev

            f['controlpulses'].require_dataset(name='controlpuls_stability',
                                               shape=(self.nmbr_channels, len(flag_cp)),
                                               dtype=bool)
            f['controlpulses']['controlpuls_stability'][channel, ...] = flag_cp

    def calc_testpulse_stability(self, channel: int, significance: float = 3, noise_level: float = 0.005,
                                 max_gap: float = 0.5, ub: float = None, lb: float = None):
        """
        Do a stability cut on the test pulses.

        The stability for the test pulses is similar to the stability cut for the control pulses, with the difference
        that we calculate average pulse height values for each individual test pulse amplitude value for the declaration
        of unstable testpulses. This cut is especially needed for the energy calibration, where single outlying test
        pulses can disturb the fit curve between TPA values and pulse heights.

        In the stability cut we assign a boolean value to each event, if it is within a stable region or not. Stable
        regions are defined as the interval between two stable test pulses. A test pulse is stable, if its height
        is within a certain number of standard deviations of the average test pulse height with its TPA.
        Single outlying test
        pulses are ignored. Test pulse heights higher than a certain maximal value or lower than a certain minimal
        value are also ignored. If for a duration of more than a certain interval no control pulses appear in the data,
        the region is automatically counted as unstable.

        :param channel: The number of the channel on that we calculate the cut in the HDF5 file.
        :type channel: int
        :param significance: Pulse heights further than this factor times the pulse height standard deviation away from
            the mean pulse height are counted as unstable.
        :type significance: float
        :param noise_level: Test pulses lower than this value are generally ignored, as they are probably triggered
            noise instead of actual pulses.
        :type noise_level: float
        :param max_gap: Intervals longer than this value (in minutes) without test pulses are automatically counted
            as unstable.
        :type max_gap: float
        :param lb: Pulse heights lower than this value are ignored.
        :type lb: float
        :param ub: Pulse heights higher than this value are ignored.
        :type ub: float
        """

        with h5py.File(self.path_h5, 'r+') as f:
            tpas = f['testpulses']['testpulseamplitude']
            tphs = f['testpulses']['mainpar'][channel, :, 0]  # 0 is the mainpar index for pulseheight
            hours_tp = f['testpulses']['hours']
            hours_ev = f['events']['hours']

            flag_ev, flag_tp = testpulse_stability(tpas, tphs, hours_tp, hours_ev,
                                                   significance=significance, noise_level=noise_level, max_gap=max_gap,
                                                   ub=ub, lb=lb)

            f['events'].require_dataset(name='testpulse_stability',
                                        shape=(self.nmbr_channels, len(flag_ev)),
                                        dtype=bool)
            f['events']['testpulse_stability'][channel, ...] = flag_ev

            f['testpulses'].require_dataset(name='testpulse_stability',
                                            shape=(self.nmbr_channels, len(flag_tp)),
                                            dtype=bool)
            f['testpulses']['testpulse_stability'][channel, ...] = flag_tp

    def calc_calibration(self,
                         starts_saturation: list,  #
                         cpe_factor: list,
                         max_dist: float = 1,  # in hours
                         exclude_tpas: list = [],
                         plot: bool = False,
                         only_stable: bool = False,
                         cut_flag: list = None,
                         interpolation_method: bool = 'linear',
                         poly_order: int = 5,
                         only_channels: list = None,
                         method: str = 'ph',
                         name_appendix_ev: str = '',
                         name_appendix_tp: str = '',
                         return_pulser_models: bool = False,
                         pulser_models: object = None,
                         name_appendix_energy: str = '',
                         ):
        """
        Calculates the calibrated energies of all events with uncertainties.

        The energy calibration is a two step process.

        In the first step, we need an time continuous estimation of the pulse height for each injected TPA. This is
        either done with a spline fit, a linear regression or a gradient boosted regression tree. In the later two
        methods, an uncertainty estimation is included.

        In the second step, we fit for the time stamp of every event a higher order polyonmial to the TPA/pulse height
        relation, for which we value estimations at discrete TPA points. From this we can estimate a TPA value
        corresponding to the pulse height of the triggered event. The TPA is in a linear relation with the recoil energy,
        which is determined by the CPE factor. In the fitting process with a polynomial, we also include the uncertainties
        in the estimated pulse heights of the test pulses with an orthogonal distance relation, linear error propagation
        and the calculation of a prediction interval.

        :param starts_saturation: The pulse heights (V) at which the saturation of the pulses starts, for each channel.
        :type starts_saturation: list of floats
        :param cpe_factor: The CPE factors for all channels.
        :type cpe_factor: list of floats
        :param max_dist: If two testpulses are more than this interval (in hours) apart, a new spline or linear regression
            model is started for the consecutive region. If the regression tree is used for the time continuous pulse
            height estimation, this argument is not used.
        :type max_dist: float
        :param exclude_tpas: Testpulses with these TPA values are excluded from the energy calibration. This is useful,
            if there are only very few pulses with a certain TPA value.
        :type exclude_tpas: list of floats
        :param plot: If set, the continuous pulse height estimation and the TPA/PH polynomial fit are plotted.
        :type plot: bool
        :param only_stable: If set, only stable test pulses are included in the energy calibration.
        :type only_stable: bool
        :param cut_flag: If provided, this is list of bool values, that determines which test pulses are to be included
            in the energy calibration.
        :type cut_flag: list of bools
        :param interpolation_method: If 'linear', we take linear regressions for the continuous pulse height estimation
            and include an uncertainty estimation. If 'tree, we take a gradient boosted regression tree for the continuous pulse height estimation
            and include an uncertainty estimation.
        :type interpolation_method: str
        :param poly_order: The order of the polynomial that we fit to describe the TPA/PH relation. This should be
            between 3 and 5.
        :type poly_order: int
        :param only_channels: If set, the calibration is done only on the channels that are handed here.
        :type only_channels: list of ints or None
        :param method: Either 'ph' (main parameter pulse height), 'of' (optimum filter), 'sef' (standard event fit)
            or 'true_ph' (in case of simulated events - here you probably want to hand pulser models as well).
            Test pulse heights and event heights are then estimated with this method for the calibration.
        :type method: string
        :param name_appendix_ev: This is appended to the event pulse height estimation method, e.g. '_down16'.
        :type name_appendix_ev: string
        :param name_appendix_tp: This is appended to the test pulse height estimation method, e.g. '_down16'.
        :type name_appendix_tp: string
        :param return_pulser_models: If set to true, a list of the used PulserModels is returned.
        :type return_pulser_models: bool
        :param pulser_models: Here a list of PulserModels that shall be used can be passed. This is useful is case the
            Calibration is done on one file with test pulses, but you want to predict the TPA equivalent values of another
            data set, e.g. the resolution data set, with the same pulser models.
        :type pulser_models: list of instances of PulserModel


        >>> dh.calc_calibration(starts_saturation=[1.5, 0.8],
        ...                     cpe_factor=[1, 1],
        ...                     exclude_tpas=[0.01],
        ...                     plot=True,
        ...                     method='tree',
        ...                     poly_order=3,
        ...                     )
        Energy Calibration for Channel  0
        Unique TPAs:  [ 0.02        0.1         0.2         0.40000001  0.60000002  0.80000001
          1.          2.          3.          4.          5.          6.
          7.          8.          9.         10.        ]
        Plot Regression Polynomial at 20.8 hours.
        Calculating Recoil Energies: 0.0 %
        Calculating Recoil Energies: 65.3 %
        Energy Calibration for Channel  1
        Unique TPAs:  [ 0.02        0.1         0.2         0.40000001  0.60000002  0.80000001
          1.          2.          3.          4.          5.          6.
          7.          8.          9.         10.        ]
        Plot Regression Polynomial at 20.8 hours.
        Calculating Recoil Energies: 0.0 %
        Calculating Recoil Energies: 65.3 %
        Finished.

        .. image:: ../pics/Ch0_tphs.png

        .. image:: ../pics/Ch0_cal.png

        .. image:: ../pics/Ch1_tphs.png

        .. image:: ../pics/Ch1_cal.png
        """

        with h5py.File(self.path_h5, 'r+') as f:

            if cpe_factor is None:
                cpe_factor = [None for i in range(self.nmbr_channels)]

            if method == 'ph':
                evhs = np.array(f['events']['mainpar' + name_appendix_ev][:, :, 0])
                tphs = np.array(f['testpulses']['mainpar' + name_appendix_tp][:, :, 0])
            elif method == 'of':
                evhs = np.array(f['events']['of_ph' + name_appendix_ev])
                tphs = np.array(f['testpulses']['of_ph' + name_appendix_tp])
            elif method == 'sef':
                evhs = np.array(f['events']['sev_fit_par' + name_appendix_ev][:, :, 0])
                tphs = np.array(f['testpulses']['sev_fit_par' + name_appendix_tp][:, :, 0])
            elif method == 'true_ph':
                evhs = np.array(f['events']['true_ph'])
                if 'true_ph' in f['testpulses']:
                    tphs = np.array(f['testpulses']['true_ph'])
                else:
                    tphs = np.array(f['testpulses']['mainpar' + name_appendix_tp][:, :, 0])
            else:
                raise KeyError('Pulse Height Estimation method not implemented, try ph, of or sef.')

            tpas = np.array(f['testpulses']['testpulseamplitude'])
            ev_hours = np.array(f['events']['hours'])
            tp_hours = np.array(f['testpulses']['hours'])

            if only_stable:
                stable = np.array(f['testpulses']['testpulse_stability'], dtype=bool)
            else:
                stable = np.ones([self.nmbr_channels, len(tpas)], dtype=bool)
            if cut_flag is not None:
                stable = np.logical_and(stable, cut_flag)

            f['events'].require_dataset(name='recoil_energy' + name_appendix_energy,
                                        shape=(self.nmbr_channels, len(ev_hours)),
                                        dtype=float)
            f['events'].require_dataset(name='tpa_equivalent' + name_appendix_energy,
                                        shape=(self.nmbr_channels, len(ev_hours)),
                                        dtype=float)
            f['events'].require_dataset(name='recoil_energy_sigma' + name_appendix_energy,
                                        shape=(self.nmbr_channels, len(ev_hours)),
                                        dtype=float)
            f['events'].require_dataset(name='tpa_equivalent_sigma' + name_appendix_energy,
                                        shape=(self.nmbr_channels, len(ev_hours)),
                                        dtype=float)

            if only_channels is None:
                only_channels = list(range(self.nmbr_channels))

            if pulser_models is None:
                pulser_models = [None for i in range(self.nmbr_channels)]

            for channel in only_channels:
                print('Energy Calibration for Channel ', channel)

                if pulser_models[channel] is None:
                    pulser_models[channel] = PulserModel(start_saturation=starts_saturation[channel],
                                                         max_dist=max_dist)

                    pulser_models[channel].fit(tphs=tphs[channel, stable[channel]],
                                               tpas=tpas[stable[channel]],
                                               tp_hours=tp_hours[stable[channel]],
                                               exclude_tpas=exclude_tpas,
                                               interpolation_method=interpolation_method,
                                               )

                if plot:
                    pulser_models[channel].plot(interpolation_method=interpolation_method,
                                                poly_order=poly_order)

                f['events']['recoil_energy' + name_appendix_energy][channel, ...], \
                f['events']['recoil_energy_sigma' + name_appendix_energy][channel, ...], \
                f['events']['tpa_equivalent' + name_appendix_energy][channel, ...], \
                f['events']['tpa_equivalent_sigma' + name_appendix_energy][channel, ...] = pulser_models[channel].predict(evhs=evhs[channel],
                                                                                                   ev_hours=ev_hours,
                                                                                                   poly_order=poly_order,
                                                                                                   cpe_factor=cpe_factor[channel],
                                                                                                   interpolation_method=interpolation_method,
                                                                                                   )
        print('Finished.')
        if return_pulser_models:
            return pulser_models


    def calc_light_correction(self,
                              scintillation_efficiency: float,
                              channels_to_calibrate: list = [0],
                              light_channel: int = 1):
        """
        Calculate the correction of the energy estimation that comes from the scintillation of light.

        When a recoil happens inside an absorber crystal, both a phonon and a light signal emerge, the recoil energy is
        therefore split apart. If we only uns the phonon channel to estimate the recoil enery, we have an error depending
        on the share of energy that went into scintillation light. It is therefor necessary to correct the energy of the
        phonon channel with a light-energy-dependent factor.

        :param scintillation_efficiency: The share of the recoil energy that is turned into scintillation light.
        :type scintillation_efficiency: float >0, <1
        :param channels_to_calibrate: All channels that scintillate light and are used as energy estimators (typically the phonon channel).
        :type channels_to_calibrate: list
        :param light_channel: The number of the channel that is the light channel, typically 1, but e.g. 2 for Gode modules.
        :type light_channel: int
        """

        with h5py.File(self.path_h5, 'r+') as f:
            recoil_energy = np.array(f['events']['recoil_energy'])

            f['events'].require_dataset(name='corrected_energy',
                                        shape=(len(channels_to_calibrate), len(recoil_energy[0])),
                                        dtype=float)

            for channel in channels_to_calibrate:
                f['events']['corrected_energy'][channel, ...] = light_yield_correction(
                    phonon_energy=recoil_energy[channel],
                    light_energy=recoil_energy[light_channel],
                    scintillation_efficiency=scintillation_efficiency)

    def exposure(self,
                 detector_mass=None,  # in kg
                 max_dist=0.1,  # in hours
                 tp_exclusion_interval=1,  # in seconds
                 return_values=False,
                 ):
        """
        Calculate the exposure in the data set.

        This method calculated the live time of the detector by excluding all test pulses, instable intervals and
        non-measurement times. If a detector mass is handed, the exposure is calculated as well.

        :param detector_mass: The mass of the detector in kg.
        :type detector_mass: float
        :param max_dist: The maximal distance between two test pulses in hours, such that the interval in between is still counted
            as measurement time.
        :type max_dist: float
        :param tp_exclusion_interval: The time in seconds that has to be excluded for every test pulse. Typically this
        is 1.5*length of record window, i.e. ~ a second for a window of length 16384 samples and 25 kHz sample frequency.
        :type tp_exclusion_interval: float
        :param return_values: If this is set to True, a tuple of (exposure, live_time) is returned.
        :type return_values: bool
        """

        with h5py.File(self.path_h5, 'r+') as f:

            hours_tp = f['testpulses']['hours']
            hours_cp = f['controlpulses']['hours']
            rate_cp = f['controlpulses']['rate_cut']
            stability_cp = f['controlpulses']['controlpuls_stability']

            good_intervals = []
            dead_time = 0
            buffer_cond = False

            print('\n Calc Live Time.')
            # calc live time from stability and rate flags
            for i in tqdm(range(1, len(hours_cp))):
                if hours_cp[i] - hours_cp[i - 1] < max_dist:  # check if gap too large

                    cond = rate_cp[i]
                    for c in range(self.nmbr_channels):
                        cond = np.logical_and(cond, stability_cp[c, i])

                    if cond:
                        good_intervals.append((hours_cp[i - 1], hours_cp[i]))
                        dead_time += tp_exclusion_interval / 3600
                        if buffer_cond:
                            good_intervals.append((hours_cp[i - 2], hours_cp[i - 1]))
                            dead_time += tp_exclusion_interval / 3600
                            buffer_cond = False
                    elif not buffer_cond:
                        buffer_cond = True
                    else:
                        buffer_cond = False
                else:
                    buffer_cond = False

            print('\n Exclude Test Pulses.')
            # exclude the record windows of testpulses
            tp_flag = np.ones(len(hours_tp), dtype=bool)
            for (l, u) in tqdm(good_intervals):
                tp_flag[np.logical_and(hours_tp > l, hours_tp < u)] = False

            dead_time += np.sum(tp_flag) * tp_exclusion_interval / 3600

            good_hours = np.sum([u - l for (l, u) in good_intervals]) - dead_time

        print('Good Measurement Time: {:.3f}/{:.3f} hours, {:.3f}%.'.format(good_hours,
                                                                            good_intervals[-1][1],
                                                                            100 * good_hours / good_intervals[-1][1]))
        print('Dead Time: {:.3f} hours'.format(dead_time))
        if detector_mass is not None:
            exposure = good_hours / 24 * detector_mass
            print('Exposure: {:.6f} kg * days.'.format(exposure))

        if return_values and detector_mass is not None:
            return (exposure, good_hours)
        elif return_values:
            return (None, good_hours)