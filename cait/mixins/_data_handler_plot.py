# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
import matplotlib.pyplot as plt
from ..fit._templates import pulse_template

# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class PlotMixin(object):

    # Plot the SEV
    def show_SEV(self,
                 type='stdevent',
                 block=True,
                 sample_length=0.04):

        f = h5py.File(self.path_h5, 'r')
        sev = f[type]['event']
        sev_fitpar = f[type]['fitpar']

        t = (np.arange(0, self.record_length, dtype=float) - self.record_length / 4) * sample_length

        # plot
        plt.close()

        for i, ch in enumerate(self.channel_names):
            plt.subplot(2, 1, i + 1)
            plt.plot(t, sev[i], color=self.colors[i])
            plt.plot(t, pulse_template(t, *sev_fitpar[i]), color='orange')
            plt.title(ch + ' ' + type)

        plt.show(block=block)

        f.close()

    # Plot the NPS
    def show_NPS(self, block=True):
        f = h5py.File(self.path_h5, 'r')

        # plot
        plt.close()

        for i, ch in enumerate(self.channel_names):
            plt.subplot(2, 1, i + 1)
            plt.loglog(f['noise']['nps'][i], color=self.colors[i])
            plt.title(ch + ' NPS')

        plt.show(block=block)

        f.close()

    # Plot the OF
    def show_OF(self, block=True):
        f = h5py.File(self.path_h5, 'r')

        of = np.array(f['optimumfilter']['optimumfilter_real']) + \
             1j * np.array(f['optimumfilter']['optimumfilter_imag'])
        of = np.abs(of) ** 2

        # plot
        plt.close()

        for i, ch in enumerate(self.channel_names):
            plt.subplot(2, 1, i + 1)
            plt.loglog(of[i], color=self.colors[i])
            plt.title(ch + ' OF')

        plt.show(block=block)

        f.close()

    # plot histogram of some value
    def show_values(self,
                    group,
                    key,
                    idx0,
                    idx1=None,
                    idx2=None,
                    block=False,
                    bins=100,
                    range=None):
        """
        Shows a histogram of some values from the HDF5 file

        :param group: string, The group index that is used in the hdf5 file,
            typically either events, testpulses or noise
        :param key: string, the key index of the hdf5 file, typically mainpar, fit_rms, ...
        :param idx0: int, the first index of the array
        :param idx0: int or None, the second index of the array
        :param idx0: int or None, the third index of the array
        :param block: bool, if the plot blocks the code when executed in cmd
        :param bins: int, the number of bins for the histogram
        :param range: 2D tuple of floats, the interval that is shown in the histogram
        :return:
        """

        hf5 = h5py.File(self.path_h5, 'r+')

        if idx1 is None:
            if idx2 is None:
                vals = hf5[group][key][idx0]
            else:
                vals = hf5[group][key][idx0, :, idx2]
        else:
            if idx2 is None:
                vals = hf5[group][key][idx0, idx1]
            else:
                vals = hf5[group][key][idx0, idx1, idx2]

        plt.close()

        plt.hist(vals,
                 bins=bins,
                 range=range)
        plt.title('{} {} {},{},{}'.format(group, key, str(idx0), str(idx1), str(idx2)))
        plt.show(block=block)

    # show histogram of main parameter
    def show_hist(self,
                  which_mp='pulse_height',
                  which_channel=0,
                  type='events',
                  which_labels=None,
                  bins=100,
                  block=False,
                  range=None):
        # pulse_height
        # t_zero
        # t_rise
        # t_max
        # t_decaystart
        # t_half
        # t_end
        # offset
        # linear_drift
        # quadratic_drift

        f_h5 = h5py.File(self.path_h5, 'r')
        nmbr_mp = f_h5[type]['mainpar'].attrs[which_mp]
        par = f_h5[type]['mainpar'][which_channel, :, nmbr_mp]
        if which_labels is not None:
            pars = []
            for lab in which_labels:
                pars.append(par[f_h5[type]['labels'][which_channel] == lab])

        # choose which mp to plot
        plt.close()
        if which_labels is not None:
            for p, l in zip(pars, which_labels):
                plt.hist(p,
                         bins=bins,
                         color=self.colors[which_channel],
                         label='Label ' + str(l), alpha=0.8,
                         range=range)
        else:
            plt.hist(par,
                     bins=bins,
                     color=self.colors[which_channel],
                     range=range)
        plt.title(type + ' ' + which_mp + ' Channel ' + str(which_channel))
        plt.show(block=block)

        print('Histogram for {} created.'.format(which_mp))
        f_h5.close()

    # show light yield plot
    def show_LY(self):
        # choose which labels to plot
        # choose which channels (e.g. for Gode modules)
        raise NotImplementedError('Not Implemented.')
