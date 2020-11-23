# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
from functools import partial
from multiprocessing import Pool
from ..fit._pm_fit import fit_pulse_shape
from ..fit._templates import baseline_template_cubic, sev_fit_template
from scipy.optimize import curve_fit
from ..fit._bl_fit import get_rms
from ..fit._saturation import logistic_curve

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
    def recalc_fit(self, path_h5=None, type='events', processes=4):
        """
        Calculate the Parameteric Fit for the Events in an HDF5 File.
        :param path_h5: string, optional, the full path to the hdf5 file, e.g. "data/bck_001.h5"

        :param type: string, either events or testpulses
        :param processes: int, the number of processes to use for the calculation
        :return: -
        """

        if type not in ['events', 'testpulses']:
            raise NameError('Type must be events or testpulses.')

        if not path_h5:
            path_h5 = self.path_h5

        h5f = h5py.File(path_h5, 'r+')
        events = h5f[type]['event']

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
                p.map(p_fit_pm, events[0, :, :]))
            l_fitpar_event = np.array(
                p.map(l_fit_pm, events[1, :, :]))

        fitpar_event = np.array([p_fitpar_event, l_fitpar_event])

        events.require_dataset('fitpar',
                               shape=fitpar_event.shape,
                               dtype='f')

        events['fitpar'][...] = fitpar_event


    # apply sev fit
    def apply_sev_fit(self, type, order_bl_polynomial, sample_length=0.04, verb=False):
        """
        Calculates the SEV fit for all events of type (events or tp) and stores in hdf5 file
        The stored parameters are (pulse_height, onset_in_ms, bl_offset[, bl_linear_coeffiient, quadratic, cubic])

        :param type: string, either events or testpulses
        :param order_bl_polynomial: int, either 0,1,2 or 3 - the order of the polynomial assumed for baseline
        :param sample_length: float, 1000/sample_frequency of the time series of the events
        :return: -
        """

        print('Calculating SEV Fit.')

        if order_bl_polynomial not in [0, 1, 2, 3]:
            raise KeyError('Order Polynomial must be in 0,1,2,3!')

        # open the dataset
        f = h5py.File(self.path_h5, 'r+')
        events = f[type]['event']
        sev_par = np.array(f['stdevent']['fitpar'])
        t = (np.arange(0, self.record_length, dtype=float) - self.record_length / 4) * sample_length

        # apply fit for all channels, save parameters
        par = np.zeros([self.nmbr_channels, len(events[0]), int(order_bl_polynomial + 3)])
        for c in range(self.nmbr_channels):
            if verb:
                print('Fitting channel {}.'.format(c))
            # create instance of fit model
            fit_model = sev_fit_template(pm_par=sev_par[c], t=t)

            # fit all
            for i in range(len(events[0])):
                par[c, i] = fit_model.fit(events[c, i], order_polynomial=order_bl_polynomial)

        # write sev fit results to file
        f['events'].require_dataset(name='sev_fit_par',
                                    shape=par.shape,
                                    dtype='float')
        f['events'].require_dataset(name='sev_fit_rms',
                                    shape=(self.nmbr_channels, len(events[0])),
                                    dtype='float')
        f['events']['sev_fit_par'][...] = par
        for c in range(self.nmbr_channels):
            fit_model = sev_fit_template(pm_par=sev_par[c], t=t)
            for i in range(len(events[0])):
                if order_bl_polynomial == 0:
                    f['events']['sev_fit_rms'][c, i] = np.sum((events[c, i] - fit_model.sef(*par[c, i])) ** 2)
                elif order_bl_polynomial == 1:
                    f['events']['sev_fit_rms'][c, i] = np.sum((events[c, i] - fit_model.sel(*par[c, i])) ** 2)
                elif order_bl_polynomial == 2:
                    f['events']['sev_fit_rms'][c, i] = np.sum((events[c, i] - fit_model.seq(*par[c, i])) ** 2)
                elif order_bl_polynomial == 3:
                    f['events']['sev_fit_rms'][c, i] = np.sum((events[c, i] - fit_model.sec(*par[c, i])) ** 2)
                else:
                    raise KeyError('Order Polynomial must be 0,1,2,3!')


    def calc_bl_coefficients(self, verb=False):
        """
        Calcualted the fit coefficients with a cubic polynomial on the noise baselines.

        :param verb: bool, if True the code provides verbal feedback about the progress
        :return: -
        """

        print('Calculating Baseline Coefficients.')

        # open file
        h5f = h5py.File(self.path_h5, 'r+')
        baselines = h5f['noise']
        nmbr_bl = len(baselines['event'][0])
        baselines.require_dataset('fit_coefficients',
                                  shape=(self.nmbr_channels, nmbr_bl, 4),
                                  dtype='f')
        baselines.require_dataset('fit_rms',
                                  shape=(self.nmbr_channels, nmbr_bl),
                                  dtype='f')
        bl_temp = baseline_template_cubic

        t = np.linspace(0, self.record_length - 1, self.record_length)

        for c in range(self.nmbr_channels):
            for i in range(nmbr_bl):
                if verb and i % 100 == 0:
                    print('Calculating Baseline: ', i)
                # fit template to every bl
                coeff, _ = curve_fit(bl_temp, t, baselines['event'][c, i])
                rms = get_rms(bl_temp(t, *coeff), baselines['event'][c, i])

                # save fit coefficients in hdf5
                baselines['fit_coefficients'][c, i, ...] = coeff
                baselines['fit_rms'][c, i] = rms

        print('Fit Coeff and Rms calculated.')
        h5f.close()

    def calc_saturation(self,
                        channel=0,
                        only_idx=None):
        """
        Fit a logistics curve to the testpulse amplitudes vs their pulse heights

        :param channel: the channel for that we calcualte the saturation
        :type channel: int
        :param only_idx: only these indices are used in the fit of the saturation
        :type only_idx: list of ints
        :return: -
        :rtype: -
        """

        h5f = h5py.File(self.path_h5, 'r+')

        if only_idx is None:
            only_idx = list(range(len(h5f['testpulses']['testpulseamplitude'])))

        par, _ = curve_fit(logistic_curve,
                           xdata=h5f['testpulses']['testpulseamplitude'][only_idx],
                           ydata=h5f['testpulses']['mainpar'][channel, only_idx, 0],
                           bounds=(0,[np.inf, np.inf, ]))

        sat = h5f.require_group('saturation')
        sat.require_dataset(name='fitpar',
                            shape=(self.nmbr_channels, len(par)),
                            dtype=np.float)
        sat['fitpar'][channel, ...] = par

        h5f.close()
        print('Saturation saved.')