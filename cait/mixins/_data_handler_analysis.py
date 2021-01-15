# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
from collections import Counter
from scipy.stats import norm
from ..cuts import rate_cut, stability_cut

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

            print('Resolution channel {}: {} eV (mean {} keV, calculated with {})'.format(c, resolutions[c]*1000, mus[c], naming))
        return np.array(resolutions), np.array(mus)

    def calc_rate_cut(self, interval=10, significance=3, min=0, max=60):
        # TODO

        h5 = h5py.File(self.path_h5, 'r+')
        hours = h5['events']['hours']

        flag = rate_cut(hours*60, interval, significance, min, max)

        h5['events'].require_dataset(name='rate_cut',
                                     shape=(flag.shape),
                                     dtype=bool)
        h5['events']['rate_cut'] = flag

    def calc_stability_cut(self, channel, significance=3, noise_level=0.005):
        # TODO

        f = h5py.File(self.path_h5, 'r+')

        tpas = f['testpulses']['testpulseamplitude']
        tphs = f['testpulses']['mainpar'][channel, :, 0]  # 0 is the mainpar index for pulseheight
        hours_tp = f['testpulses']['hours']
        hours_ev = f['events']['hours']

        flag = stability_cut(tpas, tphs, hours_tp, hours_ev,
                             significance=significance, noise_level=noise_level)

        f['events'].require_dataset(name='stability_cut',
                                     shape=(flag.shape),
                                     dtype=bool)
        f['events']['stability_cut'] = flag