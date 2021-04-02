# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool
from ..fit._pm_fit import fit_pulse_shape
from ..fit._templates import baseline_template_cubic, sev_fit_template
from scipy.optimize import curve_fit
from ..fit._bl_fit import get_rms
from ..fit._noise import noise_trigger_template, get_noise_parameters_binned
from ..fit._saturation import logistic_curve
from ..styles import use_cait_style, make_grid
from tqdm.auto import tqdm


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class FitMixin(object):
    """
    A Mixin Class to the DataHandler Class with methods for the calculation of fits.
    """

    # -----------------------------------------------------------
    # FEATURE CALCULATION
    # -----------------------------------------------------------

    # Recalculate Fit
    def calc_parametric_fit(self, path_h5=None, type='events', processes=4):
        """
        Calculate the Parameteric Fit for the Events in an HDF5 File.

        :param path_h5: string, optional, the full path to the hdf5 file, e.g. "data/bck_001.h5"
        :type
        :param type: string, either events or testpulses
        :type
        :param processes: int, the number of processes to use for the calculation
        :type
        """

        if type not in ['events', 'testpulses']:
            raise NameError('Type must be events or testpulses.')

        if not path_h5:
            path_h5 = self.path_h5

        with h5py.File(path_h5, 'r+') as h5f:
            events = h5f[type]['event']

            # take away offset
            idx = [i for i in range(len(events[0]))]
            events = events - np.mean(events[:, :, :int(self.record_length / 8)], axis=2, keepdims=True)

            print('CALCULATE FIT.')

            # get start values from SEV fit if exists
            try:
                if type == 'events':
                    sev_fitpar = h5f['stdevent']['fitpar']
                    p_fit_pm = partial(fit_pulse_shape, x0=sev_fitpar[0])
                    l_fit_pm = partial(fit_pulse_shape, x0=sev_fitpar[1])
                else:
                    raise NameError('This is only to break the loop, bc type is not events.')
            except NameError:
                p_fit_pm = fit_pulse_shape
                l_fit_pm = fit_pulse_shape

            with Pool(processes) as p:
                p_fitpar_event = np.array(
                    p.map(p_fit_pm, events[0, idx, :]))
                l_fitpar_event = np.array(
                    p.map(l_fit_pm, events[1, idx, :]))

            fitpar_event = np.array([p_fitpar_event, l_fitpar_event])

            h5f[type].require_dataset('fitpar',
                                      shape=(self.nmbr_channels, len(events[0]), 6),
                                      dtype='f')

            h5f[type]['fitpar'][:, idx, :] = fitpar_event

    # apply sev fit
    def apply_sev_fit(self, type='events', only_channels=None, sample_length=0.04, down=1, order_bl_polynomial=3,
                      t0_bounds=(-20, 20), truncation_level=None, interval_restriction_factor=None,
                      verb=False, processes=4, name_appendix='', group_name_appendix=''):
        """
        Calculates the SEV fit for all events of type (events or tp) and stores in hdf5 file
        The stored parameters are (pulse_height, onset_in_ms, bl_offset[, bl_linear_coeffiient, quadratic, cubic])

        :param type: Name of the group in the HDF5 set, either events or testpulses.
        :type type: string
        :param only_channels: Only these channels are fitted, the others are left as is or filled with zeros.
        :type only_channels: list of ints
        :param order_bl_polynomial: Either 0,1,2 or 3 - the order of the polynomial assumed for baseline.
        :type order_bl_polynomial: int
        :param sample_length: The length of a sample in milliseconds.
        :type sample_length: float
        :param down: The downsample factor for the fit, has to be a power of 2.
        :type down: int
        :param t0_bounds: The lower and upper bounds in milliseconds for the onset position.
        :type t0_bounds: 2-tuple of ints
        :param truncation_level: The pulse height Volt value at that the detector saturation starts.
        :type truncation_level: list of nmbr_channel floats
        :param interval_restriction_factor: Indices not inside this interval are ignored from the fit.
        :type interval_restriction_factor: 2-tuple of ints
        :param verb: Verbal feedback about the progress.
        :type verb: bool
        :param processes: The number of workers for the fit.
        :type processes: int
        :param name_appendix: This gets appendend to the dataset name in the HDF5 set.
        :type name_appendix: string
        :param group_name_appendix: This is appendend to the group name of the stdevent in the HDF5 set.
        :type group_name_appendix: string
        """

        print('Calculating SEV Fit.')

        if order_bl_polynomial not in [3]:
            raise KeyError('Order Polynomial must be 3! (Other Versions Depricated.)')

        if truncation_level is None:
            truncation_level = [None for i in range(self.nmbr_channels)]

        # open the dataset
        with h5py.File(self.path_h5, 'r+') as f:
            events = f[type]['event']
            sev_par = np.array(f['stdevent' + group_name_appendix]['fitpar'])
            t = (np.arange(0, self.record_length, dtype=float) - self.record_length / 4) * sample_length

            # apply fit for all channels, save parameters
            par = np.zeros([self.nmbr_channels, len(events[0]), int(order_bl_polynomial + 3)])
            for c in range(self.nmbr_channels):
                if only_channels is None or c in only_channels:
                    if verb:
                        print('Fitting channel {}.'.format(c))
                    # create instance of fit model
                    fit_model = sev_fit_template(pm_par=sev_par[c], t=t, down=down, t0_bounds=t0_bounds,
                                                 truncation_level=truncation_level[c],
                                                 interval_restriction_factor=interval_restriction_factor)

                    # fit all
                    with Pool(processes) as p:
                        par[c, ...] = list(tqdm(p.imap(fit_model.fit_cubic, events[c]), total=len(events[c])))

            # write sev fit results to file
            set_fitpar = f[type].require_dataset(name='sev_fit_par{}'.format(name_appendix),
                                                 shape=par.shape,
                                                 dtype='float')
            set_fitpar.attrs.create(name='pulse_height', data=0)
            set_fitpar.attrs.create(name='onset', data=1)
            set_fitpar.attrs.create(name='constant_coefficient', data=2)
            set_fitpar.attrs.create(name='linear_coefficient', data=3)
            set_fitpar.attrs.create(name='quadratic_coefficient', data=4)
            set_fitpar.attrs.create(name='cubic_coefficient', data=5)
            set_fitrms = f[type].require_dataset(name='sev_fit_rms{}'.format(name_appendix),
                                                 shape=(self.nmbr_channels, len(events[0])),
                                                 dtype='float')
            for c in range(self.nmbr_channels):
                if only_channels is None or c in only_channels:
                    set_fitpar[c, ...] = par[c]
            for c in range(self.nmbr_channels):
                if only_channels is None or c in only_channels:
                    fit_model = sev_fit_template(pm_par=sev_par[c], t=t)
                    for i in range(len(events[0])):
                        # if order_bl_polynomial == 0:
                        #     f['events']['sev_fit_rms_bl{}'.format(order_bl_polynomial)][c, i] = np.mean((events[c, i] - fit_model.sef(*par[c, i])) ** 2)
                        # elif order_bl_polynomial == 1:
                        #     f['events']['sev_fit_rms_bl{}'.format(order_bl_polynomial)][c, i] = np.mean((events[c, i] - fit_model.sel(*par[c, i])) ** 2)
                        # elif order_bl_polynomial == 2:
                        #     f['events']['sev_fit_rms_bl{}'.format(order_bl_polynomial)][c, i] = np.mean((events[c, i] - fit_model.seq(*par[c, i])) ** 2)
                        # elif order_bl_polynomial == 3:
                        set_fitrms[c, i] = np.mean((events[c, i][
                                                    fit_model.low:fit_model.up] - fit_model.wrap_sec(
                            *par[c, i])[fit_model.low:fit_model.up]) ** 2)
                        # else:
                        #     raise KeyError('Order Polynomial must be 0,1,2,3!')

            print('Done.')

    def calc_bl_coefficients(self, type='noise', down=1):
        """
        Calcualted the fit coefficients with a cubic polynomial on the noise baselines.

        :param type: The group name in the HDF5 set, should be noise.
        :type type: string
        :param down: The baselines are downsampled by this factor before the fit.
        :type down: int
        """

        print('Calculating Baseline Coefficients.')

        # open file
        with h5py.File(self.path_h5, 'r+') as h5f:
            events = h5f[type]
            nmbr_bl = len(events['event'][0])
            events.require_dataset('fit_coefficients',
                                   shape=(self.nmbr_channels, nmbr_bl, 4),
                                   dtype='f')
            events.require_dataset('fit_rms',
                                   shape=(self.nmbr_channels, nmbr_bl),
                                   dtype='f')
            bl_temp = baseline_template_cubic

            t = np.linspace(0, self.record_length - 1, self.record_length)
            if down > 1:
                t = np.mean(t.reshape(int(len(t) / down), down), axis=1)

            for c in range(self.nmbr_channels):
                for i in tqdm(range(nmbr_bl)):
                    # fit template to every bl
                    ev = events['event'][c, i]
                    if down > 1:
                        ev = np.mean(ev.reshape(int(len(ev) / down), down), axis=1)
                    coeff, _ = curve_fit(bl_temp, t, ev)
                    rms = get_rms(bl_temp(t, *coeff), ev)

                    # save fit coefficients in hdf5
                    events['fit_coefficients'][c, i, ...] = coeff
                    events['fit_rms'][c, i] = rms

            print('Fit Coeff and Rms calculated.')

    def calc_saturation(self,
                        channel=0,
                        only_idx=None):
        """
        Fit a logistics curve to the testpulse amplitudes vs their pulse heights.

        :param channel: The channel for that we calculate the saturation.
        :type channel: int
        :param only_idx: Only these indices are used in the fit of the saturation.
        :type only_idx: list of ints
        """

        with h5py.File(self.path_h5, 'r+') as h5f:
            if only_idx is None:
                only_idx = list(range(len(h5f['testpulses']['testpulseamplitude'])))

            par, _ = curve_fit(logistic_curve,
                               xdata=h5f['testpulses']['testpulseamplitude'][only_idx],
                               ydata=h5f['testpulses']['mainpar'][channel, only_idx, 0],
                               bounds=(0, [np.inf, np.inf, ]))

            sat = h5f.require_group('saturation')
            sat.require_dataset(name='fitpar',
                                shape=(self.nmbr_channels, len(par)),
                                dtype=np.float)
            sat['fitpar'][channel, ...] = par

        print('Saturation saved.')

    def estimate_trigger_threshold(self,
                                   channel,
                                   detector_mass,
                                   allowed_noise_triggers=1,
                                   method='of',
                                   bins=100,
                                   yran=(1, 10e4),
                                   xran=None,
                                   xran_hist=None,
                                   ul=30, # in mV
                                   ll=0,  # in mV
                                   cut_flag=None,
                                   plot=True,
                                   title=None,
                                   sample_length=4e-5,  # in seconds
                                   record_length=16384,  # in samples
                                   interval_restriction=0.75,
                                   ):
        # TODO

        print('Estimating Trigger Threshold.')

        with h5py.File(self.path_h5, 'r+') as h5f:
            if method == 'of':
                phs = h5f['noise']['of_ph'][channel]
            elif method == 'ph':
                phs = h5f['noise']['mainpar'][channel, :, 0]
            else:
                raise NotImplementedError('This method is not implemented.')
            phs = np.array(phs) * 1000  # to mV
        if cut_flag is not None:
            phs = phs[cut_flag]

        # make histogram of events heights
        counts_hist, bins_hist = np.histogram(phs, bins=bins, range=(ll, ul), density=True)

        nmbr_baselines = len(phs)
        print('Nmbr baseline: ', nmbr_baselines)
        d, sigma = get_noise_parameters_binned(counts=counts_hist,
                                               bins=bins_hist,
                                               )
        # d, sigma = get_noise_parameters(x_max=phs, baseline_resolution=baseline_resolution)
        print('Fitted Noise Trigger Template Parameters: d {},  sigma {:.3} mV'.format(d, sigma))

        # calc the exposure in kg days
        trigger_window = record_length * sample_length * detector_mass / 3600 / 24 * interval_restriction

        # get the noise trigger rate
        num = 1000
        h = (ul - ll)/num
        x_grid = np.linspace(start=ll, stop=ul, num=num)
        ph_distribution = noise_trigger_template(x_max=x_grid, d=d, sigma=sigma)
        noise_trigger_rate = np.array([h*np.sum(ph_distribution[i:]) for i in range(len(ph_distribution))])
        ph_distribution /= trigger_window
        noise_trigger_rate /= trigger_window

        # calc the threshold
        threshold = x_grid[noise_trigger_rate < allowed_noise_triggers][0]
        print('Threshold for {} Noise Trigger per kg day: {:.3} mV'.format(allowed_noise_triggers, threshold))


        if plot:

            # plot the counts
            plt.close()
            use_cait_style()
            xdata = bins[:-1] + (bins[1] - bins[0]) / 2
            plt.hist(xdata, bins_hist, weights=counts_hist / trigger_window,
                     zorder=8, alpha=0.8, label='Counts')
            plt.plot(x_grid, ph_distribution, linewidth=2, zorder=12, color='black', label='Fit Model')
            make_grid()
            if title is not None:
                plt.title(title)
            if xran_hist is not None:
               plt.xlim(xran_hist)
            plt.legend()
            plt.xlabel('Pulse Height (mV)')
            plt.ylabel('Counts (1 / kg days mV)')
            plt.show()

            # plot the threshold
            plt.close()
            use_cait_style()
            plt.plot(x_grid, noise_trigger_rate, linewidth=2, zorder=16, color='black', label='Noise Trigger Rate')
            plt.vlines(x=threshold, ymin=yran[0], ymax=allowed_noise_triggers, color='tab:red',
                        linewidth=2, zorder=20, label='{} / kg days'.format(allowed_noise_triggers))
            plt.hlines(y=allowed_noise_triggers, xmin=0, xmax=threshold, color='tab:red',
                        linewidth=2, zorder=20)
            make_grid()
            plt.ylim(yran)
            if xran is not None:
               plt.xlim(xran)
            plt.yscale('log')
            if title is not None:
                plt.title(title)
            plt.legend()
            plt.xlabel('Threshold (mV)')
            plt.ylabel('Noise Trigger Rate (1 / kg days)')
            plt.show()
