import warnings
from functools import partial
from multiprocessing import Pool
from deprecation import deprecated
from typing import Union, List

import numpy as np
import h5py
from tqdm.auto import tqdm

from ..fit._pm_fit import fit_pulse_shape
from ..fit._templates import baseline_template_cubic, sev_fit_template
from scipy.optimize import curve_fit
from ..fit._bl_fit import get_rms
from ..fit._noise import get_noise_parameters_binned, get_noise_parameters_unbinned, \
    plot_noise_trigger_model, calc_threshold
from ..fit._saturation import logistic_curve_zero, A_zero
from ..fit._numerical_fit import array_fit, arr_fit_rms
from ..styles._print_styles import txt_fmt

import cait.versatile as vai

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

        This methods was described in "(1995) F. Pröbst et. al., Model for cryogenic particle detectors with superconducting phase
        transition thermometers."

        :param path_h5: Optional, the full path to the hdf5 file, e.g. "data/bck_001.h5".
        :type path_h5: string
        :param type: Either events or testpulses.
        :type type: string
        :param processes: The number of processes to use for the calculation.
        :type processes: int
        """

        if type not in ['events', 'testpulses']:
            raise NameError('Type must be events or testpulses.')

        if not path_h5:
            path_h5 = self.path_h5

        with h5py.File(path_h5, 'r+') as h5f:
            events = h5f[type]['event']

            # take away offset
            events = events - np.mean(events[:, :, :int(self.record_length / 8)], axis=2, keepdims=True)

            print('CALCULATE FIT.')

            # get start values from SEV fit if exists
            if type == 'events' and 'stdevent' in h5f:
                sev_fitpar = h5f['stdevent']['fitpar']

                fit_pm = [partial(fit_pulse_shape, x0=fp) for fp in sev_fitpar]
            else:
                fit_pm = fit_pulse_shape

            fitpar_event = np.empty((self.nmbr_channels, events.shape[1], 6))

            with Pool(processes) as p:
                for c in range(self.nmbr_channels):
                    print('Fitting Channel: ', c)
                    # fitpar_event[c] = list(tqdm(p.map(fit_pm[c], events[c]), total=events.shape[1]))
                    fitpar_event[c] = list(tqdm(p.imap(fit_pm[c], events[c]), total=events.shape[1]))

            fitpar_event = np.array([fitpar_event])

            h5f[type].require_dataset('fitpar',
                                      shape=(self.nmbr_channels, events.shape[1], 6),
                                      dtype='float')

            h5f[type]['fitpar'][:, :, :] = fitpar_event

    # apply sev fit
    @deprecated(deprecated_in='1.1.0', details="Use DataHandler.apply_template_fit() instead.")
    def apply_sev_fit(self, type='events', only_channels=None, sample_length=None, down=1, order_bl_polynomial=3,
                      t0_bounds=(-20, 20), truncation_level=None, interval_restriction_factor=None,
                      verb=False, processes=4, name_appendix='', group_name_appendix='', first_channel_dominant=False,
                      use_saturation=False):
        """
        Calculates the SEV fit for all events of type (events or tp) and stores in HDF5 file.
        The stored parameters are (pulse_height, onset_in_ms, bl_offset, bl_linear_coeffiient, quadratic, cubic).

        Attention! Since v1.1 it is recommended to use apply_array_fit instead, which provides a more stable fit implementation.

        This method was described in "F. Reindl, Exploring Light Dark Matter With CRESST-II Low-Threshold Detector",
        available via http://mediatum.ub.tum.de/?id=1294132 (accessed on the 9.7.2021).

        :param type: Name of the group in the HDF5 set, either events or testpulses.
        :type type: string
        :param only_channels: Only these channels are fitted, the others are left as is or filled with zeros.
        :type only_channels: list of ints
        :param order_bl_polynomial: Either 0,1,2 or 3 - the order of the polynomial assumed for baseline.
        :type order_bl_polynomial: int
        :param sample_length: The length of a sample in milliseconds. If None, this is calculated from the sample frequency.
        :type sample_length: float
        :param down: The downsample factor for the fit, has to be a power of 2.
        :type down: int
        :param t0_bounds: The lower and upper bounds in milliseconds for the onset position.
        :type t0_bounds: 2-tuple of ints
        :param truncation_level: The pulse height Volt value at that the detector saturation starts.
        :type truncation_level: list of nmbr_channel floats
        :param interval_restriction_factor: Value between 0 and 1, the inverval of the event is restricted
            around 1/4 by this factor.
        :type interval_restriction_factor: 2-tuple of ints
        :param verb: Verbal feedback about the progress.
        :type verb: bool
        :param processes: The number of workers for the fit.
        :type processes: int
        :param name_appendix: This gets appendend to the dataset name in the HDF5 set.
        :type name_appendix: string
        :param group_name_appendix: This is appendend to the group name of the stdevent in the HDF5 set.
        :type group_name_appendix: string
        :param first_channel_dominant: Take the peak position from the first channel and evaluate the others at the
            same position.
        :type first_channel_dominant: bool
        """

        if sample_length is None:
            sample_length = 1000 / self.sample_frequency

        print('Calculating SEV Fit.')

        if order_bl_polynomial not in [3]:
            raise KeyError('Order Polynomial must be 3! (Other Versions Depricated.)')

        if truncation_level is None:
            truncation_level = [None for i in range(self.nmbr_channels)]

        # open the dataset
        with h5py.File(self.path_h5, 'r+') as f:
            assert not use_saturation or 'saturation' in f, 'For using the saturation you need to calculate ' \
                                                            'the saturation curve first!'

            events = f[type]['event']
            sev_par = np.array(f['stdevent' + group_name_appendix]['fitpar'])
            t = (np.arange(0, self.record_length, dtype=float) - self.record_length / 4) * sample_length

            # get start values for t0
            if 'mainpar' in f[type]:
                t0_start = (np.array(f[type]['mainpar'][:, :, 1]) - self.record_length / 4) / self.sample_frequency
            else:
                t0_start = [-3 for i in range(events.shape[1])]
                warnings.warn('No main parameters calculated. With main parameters, the fit will work much better!')

            # apply fit for all channels, save parameters
            par = np.zeros([self.nmbr_channels, events.shape[1], int(order_bl_polynomial + 3)])
            for c in range(self.nmbr_channels):
                if only_channels is None or c in only_channels:
                    if verb:
                        print('Fitting channel {}.'.format(c))

                    if use_saturation:
                        saturation_pars = f['saturation']['fitpar'][c]
                    else:
                        saturation_pars = None

                        # create instance of fit model
                    fit_model = sev_fit_template(pm_par=sev_par[c], t=t, down=down, t0_bounds=t0_bounds,
                                                 truncation_level=truncation_level[c],
                                                 interval_restriction_factor=interval_restriction_factor,
                                                 saturation_pars=saturation_pars)

                    # fit all
                    with Pool(processes) as p:
                        if first_channel_dominant and c != 0:
                            par[c, ...] = list(
                                tqdm(p.imap(fit_model.fit_cubic, zip(events[c], par[0, :, 1], t0_start[c])),
                                     total=events.shape[1]))
                        else:
                            par[c, ...] = list(tqdm(p.imap(fit_model.fit_cubic,
                                                           zip(events[c], [None for i in range(events.shape[1])],
                                                               t0_start[c])),
                                                    total=events.shape[1]))

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
                                                    fit_model.low:fit_model.up] - fit_model._wrap_sec(
                            *par[c, i])[fit_model.low:fit_model.up]) ** 2)
                        # else:
                        #     raise KeyError('Order Polynomial must be 0,1,2,3!')

            print('Done.')

    # apply array fit
    @deprecated(deprecated_in='1.3.0', details="Use DataHandler.apply_template_fit() instead.")
    def apply_array_fit(self, type='events', only_channels=None, sample_length=None,
                        max_shift=20,
                        truncation_level=None,
                        processes=4, name_appendix='', group_name_appendix='',
                        first_channel_dominant=False,
                        use_this_array=None, blcomp=4, no_bl_when_sat=True,
                        ):
        """
        Calculates the array fit for all events of type (events or tp) and stores in HDF5 file.

        The stored parameters are (pulse_height, onset_in_ms, bl_offset, bl_linear_coeffiient, quadratic, cubic). This
        method is different to apply_sev_fit, because we can use an arbitrary numerical array as standard event fit
        component, not only one described by the parameters of the pulse shape model. Per default, the numerical sev
        is used as sev component. Furthermore, the fit is split into a linear and nonlinear regression part, which
        provides significant speedup. Only the t0 parameter is subject to a nonlinear optimization problem, while the
        others are calculated by matrix inversion.

        :param type: Name of the group in the HDF5 set, either events or testpulses.
        :type type: string
        :param only_channels: Only these channels are fitted, the others are left as is or filled with zeros.
        :type only_channels: list of ints
        :param sample_length: The length of a sample in milliseconds. If None, this is calculated from the sample frequency.
        :type sample_length: float
        :param max_shift: The maximal shift in ms allowed for the t0 value.
        :type max_shift: float
        :param truncation_level: The pulse height Volt value at that the detector saturation starts.
        :type truncation_level: list of nmbr_channel floats
        :param processes: The number of workers for the fit.
        :type processes: int
        :param name_appendix: This gets appendend to the dataset name in the HDF5 set.
        :type name_appendix: string
        :param group_name_appendix: This is appendend to the group name of the stdevent in the HDF5 set.
        :type group_name_appendix: string
        :param first_channel_dominant: Take the peak position from the first channel and evaluate the others at the
            same position.
        :type first_channel_dominant: bool
        :param use_this_array: List of the standardevents/array that are used for the fit. Shape: (nmbr_channels,
            record_length).
        :type use_this_array: 2D numpy array
        :param blcomp: Either 1,2,3 or 4 - number of the baseline components, i.e. order + 1 of the polynomial assumed for baseline.
        :type blcomp: int
        """

        if sample_length is None:
            sample_length = 1000 / self.sample_frequency

        print('Calculating Array Fit.')

        if truncation_level is None:
            truncation_level = [None for i in range(self.nmbr_channels)]

        # open the dataset
        with h5py.File(self.path_h5, 'r+') as f:

            if use_this_array is None:
                sevs = f['stdevent{}'.format(group_name_appendix)]['event']
            else:
                sevs = use_this_array
            events = f[type]['event'][:, :, :]
            record_length = events.shape[2]
            t = (np.arange(0, record_length, dtype='f') - record_length / 4) * sample_length
            bs = int(max_shift / sample_length)
            par = np.zeros((events.shape[0], events.shape[1], 6))
            with Pool(processes) as p:
                for c in range(events.shape[0]):
                    if only_channels is None or c in only_channels:
                        print('Fitting Channel: ', c)
                        if c == 0 or not first_channel_dominant:
                            fh = partial(array_fit, sev=sevs[c], t=t, blcomp=blcomp, trunclv=truncation_level[c],
                                         bs=bs, no_bl_when_sat=no_bl_when_sat)
                            par[c] = list(tqdm(p.imap(fh, zip(events[c], [None for i in range(events.shape[1])])),
                                               total=events.shape[1]))
                        else:
                            fh = partial(array_fit, sev=sevs[c], t=t, blcomp=blcomp, trunclv=truncation_level[c],
                                         bs=bs, no_bl_when_sat=no_bl_when_sat)
                            par[c] = list(tqdm(p.imap(fh, zip(events[c], par[0, :, 1])),
                                               total=events.shape[1]))

            # write sev fit results to file
            set_fitpar = f[type].require_dataset(name='arr_fit_par{}'.format(name_appendix),
                                                 shape=par.shape,
                                                 dtype='float')
            set_fitpar.attrs.create(name='pulse_height', data=0)
            set_fitpar.attrs.create(name='onset', data=1)
            set_fitpar.attrs.create(name='constant_coefficient', data=2)
            set_fitpar.attrs.create(name='linear_coefficient', data=3)
            set_fitpar.attrs.create(name='quadratic_coefficient', data=4)
            set_fitpar.attrs.create(name='cubic_coefficient', data=5)

            set_fitrms = f[type].require_dataset(name='arr_fit_rms{}'.format(name_appendix),
                                                 shape=(self.nmbr_channels, events.shape[1]),
                                                 dtype=float)

            bound_samples = int(max_shift / sample_length)
            A = np.empty((record_length - 2*bound_samples, 1 + blcomp), dtype='f')
            for i in range(blcomp):
                A[:, i + 1] = t[bound_samples:-bound_samples] ** i

            for c in range(self.nmbr_channels):
                if only_channels is None or c in only_channels:
                    set_fitpar[c, ...] = par[c]
                    for i in range(events.shape[1]):
                        set_fitrms[c, i] = arr_fit_rms(par[c, i, :2 + blcomp], A, events[c, i, bound_samples:-bound_samples],
                                                       sevs[c], bound_samples)

            print('Done.')

    def apply_template_fit(self,
                           group: str,
                           sev: np.ndarray,
                           bl_poly_order: Union[int, List[int]] = 3,
                           truncation_limit: Union[float, List[float]] = None,
                           correlated: bool = False,
                           fit_onset: Union[bool, List[bool]] = True,
                           max_shift: int = 50,
                           only_channels: Union[int, List[int]] = None,
                           event_flag: np.ndarray = None,
                           **kwargs
                           ):
        """
        Perform a (correlated) template fit for for events of a specified group, i.e. fit a numeric SEV to data with possibility to also specify a polynomial baseline model (for each channel individually) and a truncation limit (for each channel individually).
        The 'correlated' in this context means that you can choose which of the channel's onset should be fitted (possibly multiple, see below).
        See https://edoc.ub.uni-muenchen.de/23762/ and https://mediatum.ub.tum.de/?id=1294132 for details.

        :param group: The DataHandler group with the events that should be fitted. The fit parameters will be saved to this group, too.
        :type group: str
        :param sev: The template (SEV) to use in the fit (with as many channels as you want to fit).
        :type sev: np.ndarray
        :param bl_poly_order: List of the baseline models to use in the fit (one entry for each channel that you want to fit). Has to be a non-zero integer or None. If 0, a constant offset is fitted, if 1, a linear baseline is assumed, etc. If None, the baseline is assumed to be constantly 0 (here, it's the users responsibility to remove the baseline accordingly). If only None or an integer is provided, this value is used for all fitted channels, defaults to 3, i.e. fitting a cubic baseline for all channels.
        :type bl_poly_order: Union[int, List[int]], optional
        :param truncation_limit: List with as many entries as there are channels to fit. For each entry that is not None, a truncated fit is performed: all samples between the first and the last sample above 'truncation_limit' are ignored in the fit. To determine these samples, the baseline of the event is removed by fitting a linear polynomial to the beginning of the record window. If only None or a float is provided, this value is used for all fitted channels. Defaults to None, i.e. not performing a truncated fit in any channel.
        :type truncation_limit: Union[float, List[float]], optional
        :param correlated: If True, a correlated fit is performed, i.e. the SEV is shifted for all channels simultaneously. Depending on 'fit_onset' (see below), different behavior can be achieved. If False, each channel is fitted independently of the others, defaults to False. 
        :type correlated: bool, optional
        :param fit_onset: List with as many entries as there are channels to fit. For each entry that is True, the onset value of the respective channel is fitted. If ``correlated=False``, the onsets are fitted independently. If ``correlated=True``, all channels for which ``fit_onset=True`` participate in the onset fit (common minimization): If only one of the entries is True, this channel is the 'dominant' one, i.e. its onset is fitted and all other channels are moved (passively) in unison (and only their pulse height + baseline is fitted). If multiple are True, their chi-squared for the fit is combined, i.e. the template is still moved in unison, but the minimizer considers all channels. If only one boolean is provided, it is used for all fitted channels. Defaults to True, i.e. fit onset in all channels
        :type fit_onset: Union[bool, List[bool]], optional
        :param max_shift: The maximum shift value (in samples) to search for a minimum. The onset fit will search the minimum for shifts in ``(-max_shift, +max_shift)``.
        :type max_shift: int, optional

        :param only_channels: If you only want to fit some of the channels in 'group', you can specify the channel index/indices here. Note that the size of 'sev', 'bl_poly_order', 'truncation_limit', and 'fit_onset' have to match the number of channels to be fitted, i.e. if you fit two channels, the 'sev' also has to have two channels. Defaults to None, i.e. fitting all channels in 'group'.
        :type only_channels: Union[int, List[int]], optional
        :param event_flag: A boolean flag. If you don't want to fit all events in 'group', you can specify a flag for which to fit here. Events that were not fit, receive an RMS value of -404 in the output dataset. Has to have the same length as there are events in 'group' and applies to all channels. Defaults to None, i.e. fit all events.
        :type event_flag: np.ndarray, optional
        :param kwargs: Additional keyword arguments passed to :class:`cait.versatile.TemplateFit` or :class:`cait.versatile.TemplateFitCorrelated` (depending on the 'correlated' keyword).
        :type kwargs: any, optional
        """
        
        if not self.exists(group):
            raise KeyError(f"Group '{group}' is not available in this DataHandler.")
        
        _ds_to_be_written = ['templatefit_pars', 'templatefit_rms', 'templatefit_shift']
        if any([self.exists(group, ds) for ds in _ds_to_be_written]):
            raise KeyError(f"One or more of the datasets {_ds_to_be_written} are already present in the '{group}' group, and would be overwritten by this function call. If you intend to do so, please manually delete the respective datasets first by calling 'dh.drop({group}, 'dataset')'.")
        
        events = self.get_event_iterator(group)

        if only_channels is None: only_channels = slice(None, None, None)
        if event_flag is None: event_flag = np.ones(len(events), dtype=bool)
        
        n_channels = events.n_channels
        n_channels_used = events[only_channels].n_channels
        channels_used = np.atleast_1d(np.arange(n_channels)[only_channels])
        events_used = events[:, event_flag]

        if correlated and n_channels_used<2:
            raise ValueError("The 'correlated' fit is only available for >=2 channels.")

        # Make sure all (relevant) inputs are lists
        if bl_poly_order is None or isinstance(bl_poly_order, int):
            bl_poly_order = [bl_poly_order]*n_channels_used

        if truncation_limit is None or isinstance(truncation_limit, float):
            truncation_limit = [truncation_limit]*n_channels_used

        if fit_onset is None or isinstance(fit_onset, bool):
            fit_onset = [fit_onset]*n_channels_used

        # Make sure all lists have the same length as the number of used channels.
        if len(bl_poly_order) != n_channels_used:
            raise ValueError(f"Length of 'bl_poly_order' has to match the number of fitted channels. Got {len(bl_poly_order)} and {n_channels_used}.")
        if len(truncation_limit) != n_channels_used:
            raise ValueError(f"Length of 'truncation_limit' has to match the number of fitted channels. Got {len(truncation_limit)} and {n_channels_used}.")
        if len(fit_onset) != n_channels_used:
            raise ValueError(f"Length of 'fit_onset' has to match the number of fitted channels. Got {len(fit_onset)} and {n_channels_used}.")

        sev = np.atleast_2d(np.array(sev))

        if sev.shape[0] != n_channels_used:
            raise ValueError(f"'sev' must have as many channels as you want to fit. Got {sev.shape[0]} and {n_channels_used}.")
        
        # Construct the output array (to be filled later)
        non_none_orders = [x for x in bl_poly_order if x is not None]
        max_order = np.max(non_none_orders) if len(non_none_orders)>0 else -1
        output_shape = (n_channels, len(events), 2+max_order)

        # For unused channels, the RMS is set to -404 and the 
        # fit parameters are all 0
        output_pars = np.zeros(output_shape)
        output_shift = np.zeros((n_channels, len(events)), dtype=np.int16)
        output_rms = -404*np.ones((n_channels, len(events)))

        if correlated:
            tf = vai.TemplateFitCorrelated(
                    sev=sev,
                    bl_poly_order=bl_poly_order,
                    truncation_limit=truncation_limit,
                    fit_onset=fit_onset,
                    max_shift=max_shift, 
                    **kwargs)
            fitpar, opt_shift, rms = vai.apply(tf, events_used)

            output_pars[channels_used, event_flag, :] = np.transpose(fitpar, [1,0,2])
            output_shift[channels_used, event_flag] = opt_shift.T
            output_rms[channels_used, event_flag] = rms.T

        else:
            for i in range(n_channels_used):
                ch = channels_used[i]
                tf = vai.TemplateFit(
                        sev=sev[i],
                        bl_poly_order=bl_poly_order[i],
                        truncation_limit=truncation_limit[i],
                        fit_onset=fit_onset[i],
                        max_shift=max_shift, 
                        **kwargs)

                fitpar, opt_shift, rms = vai.apply(tf, events_used[i], pb_prefix=f"Channel {ch}")

                output_pars[ch, event_flag, :] = np.transpose(fitpar, [1,0,2])
                output_shift[ch, event_flag] = opt_shift
                output_rms[ch, event_flag] = rms

        self.set(group, 
                 templatefit_pars=output_pars, 
                 templatefit_rms=output_rms, 
                 dtype=np.float32)
        self.set(group, templatefit_shift=output_shift, dtype=np.int16)

        if np.any(output_rms == -404):
            print(f"{txt_fmt('One or more RMS value(s) was/were set to -404.', style='bold')} The corresponding fit values should not be trusted! If you provided an 'event_flag', the events which were excluded from the fit received an RMS value of -404. Likewise, if you chose only to fit a subset of channels using the 'only_channels' argument, channels which were not fitted had their RMS values set to -404. Finally, if the fit failed for any of the fitted events, the respective RMS was also set to -404. {txt_fmt('This means that you should ALWAYS perform a cut of the form RMS>0.', style='bold')}")

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
                                   dtype=float)
            events.require_dataset('fit_rms',
                                   shape=(self.nmbr_channels, nmbr_bl),
                                   dtype=float)
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

                    coeff = np.polyfit(x=t, y=ev, deg=3)
                    coeff = np.flip(coeff)

                    # coeff, _ = curve_fit(bl_temp, t, ev)

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

        This method was used to describe the detector saturation in "M. Stahlberg, Probing low-mass dark matter with
        CRESST-III : data analysis and first results",
        available via https://doi.org/10.34726/hss.2021.45935 (accessed on the 9.7.2021).

        :param channel: The channel for that we calculate the saturation.
        :type channel: int
        :param only_idx: Only these indices are used in the fit of the saturation.
        :type only_idx: list of ints
        """

        with h5py.File(self.path_h5, 'r+') as h5f:
            if only_idx is None:
                only_idx = np.arange(h5f['testpulses']['testpulseamplitude'].shape[0])

            tphs = h5f['testpulses']['mainpar'][channel, only_idx, 0]
            tpas = h5f['testpulses']['testpulseamplitude']
            if len(tpas.shape) > 1:
                tpas = tpas[channel, only_idx]
            else:
                tpas = tpas[only_idx]

            par, _ = curve_fit(logistic_curve_zero,
                               xdata=tpas,
                               ydata=tphs,
                               # (A) K C Q B nu - A is not fitted
                               bounds=([0, 0, 0, 0, 0],
                                       [np.inf, np.inf, np.inf, np.inf, np.inf]))

            A = A_zero(*par)

            print('Saturation calculated: A {} K {} C {} Q {} B {} nu {}'.format(A, *par))

            sat = h5f.require_group('saturation')
            sat.require_dataset(name='fitpar',
                                shape=(self.nmbr_channels, len(par) + 1),
                                dtype=np.float)
            sat['fitpar'][channel, 0] = A
            sat['fitpar'][channel, 1:] = par

            sat['fitpar'].attrs.create(name='A', data=0)
            sat['fitpar'].attrs.create(name='K', data=1)
            sat['fitpar'].attrs.create(name='C', data=2)
            sat['fitpar'].attrs.create(name='Q', data=3)
            sat['fitpar'].attrs.create(name='B', data=4)
            sat['fitpar'].attrs.create(name='nu', data=5)

    def estimate_trigger_threshold(self,
                                   channel,
                                   detector_mass,
                                   allowed_noise_triggers=1,
                                   sigma_x0=2,
                                   method='of',
                                   bins=200,
                                   yran=None,
                                   xran=None,
                                   xran_hist=None,
                                   ul=30,  # in mV
                                   ll=0,  # in mV
                                   cut_flag=None,
                                   plot=True,
                                   title=None,
                                   sample_length=None,  # in seconds
                                   record_length=None,  # in samples
                                   interval_restriction=0.75,
                                   binned_fit=False,
                                   model='gauss',
                                   ylog=False,
                                   save_path=None,
                                   ):
        """
        Estimate the trigger threshold to obtain a given number of noise triggers per exposure.

        The method assumes a Gaussian sample distribution of the noise, following "A method to define the energy
        threshold depending on noise level for rare event searches" (arXiv:1711.11459). There are multiple
        extensions implemented, that descibe additional Gaussian mixture or non-Gaussian components. A more extensive
        description can be found in the corresponding tutorial.

        :param channel: The number of the channel for that we estimate the noise trigger threshold.
        :type channel: int
        :param detector_mass: The mass of the detector in kg.
        :type detector_mass: float
        :param allowed_noise_triggers: The number of noise triggers that are allowed per kg day exposure.
        :type allowed_noise_triggers: float
        :param sigma_x0: A start value for the baseline resolution. Is only used for the unbinned fit.
        :type sigma_x0: float
        :param method: Either 'of' for estimating the noise triggers after optimal filtering or 'ph' for taking the
            maximum value of the raw data.
        :type method: string
        :param bins: The number of bins for the histogram plots.
        :type bins: int
        :param yran: The range of the y axis on both plots.
        :type yran: tuple of two floats
        :param xran: The range of the x axis on the noise trigger estimation plot.
        :type xran: tuple of two floats
        :param xran_hist: The range of the x axis on the histogram plot.
        :type xran_hist: tuple of two floats
        :param ul: The upper limit of the interval that is used to search a threshold, in mV.
        :type ul: float
        :param ll: The lower limit of the interval that is used to search a threshold, in mV.
        :type ll: float
        :param cut_flag: A list of boolean values that determine which events are excluded from the calculation.
        :type cut_flag: list of bool
        :param plot: If True, a plot of the fit and the noise trigger estimation are shown.
        :type plot: bool
        :param title: A title for both plots.
        :type title: string
        :param sample_length: The length of a sample in seconds. If None, it is calculated from the sample frequency.
        :type sample_length: float
        :param record_length: The number of samples within a record window.
        :type record_length: int
        :param interval_restriction:
        :type interval_restriction:
        :param binned_fit: Not recommended. If chosen, the model is fit with least squared to the histogram. Otherwise
            an unbinned likelihood fit is performed.
        :type binned_fit: bool
        :param model: Determine which model is fit to the noise.
            - 'gauss': Model of purely Gaussian noise.
            - 'pollution_exponential': Model of Gaussian noise with one exponentially distributed sample on each baseline.
            - 'fraction_exponential': Mixture model of Gaussian and exponentially distributed noise.
            - 'pollution_gauss': Model of Gaussian noise and one sample in each baseline that follows another, also Gaussian distribution.
            - 'fraction_gauss':  Mixture model of two Gaussian noise components.
        :type model: string
        :param ylog: If set, the y axis is plotted logarithmically on the histogram plot.
        :type ylog: bool
        :param save_path: A path to save the plots.
        :type save_path: string
        """

        if sample_length is None:
            sample_length = 1 / self.sample_frequency

        if record_length is None:
            record_length = self.record_length

        print('Estimating Trigger Threshold.')
        print('Using model: {}'.format(model))

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
        if binned_fit:
            if model != 'gauss':
                raise NotImplementedError('For the binned fit, only the Gauss model is implemented!')
            pars = get_noise_parameters_binned(counts=counts_hist,
                                               bins=bins_hist,
                                               )
        else:
            pars = get_noise_parameters_unbinned(events=phs,
                                                 model=model,
                                                 sigma_x0=sigma_x0,
                                                 )

        x_grid, \
        trigger_window, \
        ph_distribution, \
        polluted_ph_distribution, \
        noise_trigger_rate, \
        polluted_trigger_rate, \
        threshold, \
        nmbr_pollution_triggers = calc_threshold(record_length, sample_length, detector_mass, interval_restriction, ul,
                                                 ll, model,
                                                 pars, allowed_noise_triggers)

        if plot:
            plot_noise_trigger_model(bins_hist=bins_hist,
                                     counts_hist=counts_hist,
                                     x_grid=x_grid,
                                     trigger_window=trigger_window,
                                     ph_distribution=ph_distribution,
                                     model=model,
                                     polluted_ph_distribution=polluted_ph_distribution,
                                     title=title,
                                     xran_hist=xran_hist,
                                     noise_trigger_rate=noise_trigger_rate,
                                     polluted_trigger_rate=polluted_trigger_rate,
                                     threshold=threshold,
                                     yran=yran,
                                     allowed_noise_triggers=allowed_noise_triggers,
                                     nmbr_pollution_triggers=nmbr_pollution_triggers,
                                     xran=xran,
                                     ylog=ylog,
                                     only_histogram=False,
                                     save_path=save_path,
                                     )
