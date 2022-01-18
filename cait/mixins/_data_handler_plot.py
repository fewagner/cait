# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
import matplotlib.pyplot as plt
from ..fit._templates import pulse_template
from ..fit._saturation import logistic_curve
from ..styles._plt_styles import use_cait_style, make_grid
import warnings


# functions

def _str_empty(value):
    """
    Return an empty string if the argument is None, otherwise return it as string.
    """
    if value is None:
        return ''
    else:
        return str(value)


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class PlotMixin(object):
    """
    Mixin Class for the DataHandler to make essential plots for the analysis
    """

    # Plot the SEV
    def show_sev(self,
                 type='stdevent',
                 channel=None,
                 title=None,
                 show_fit=True,
                 block=True,
                 sample_length=None,
                 show=True,
                 save_path=None,
                 name_appendix='',
                 dpi=150):
        """
        Plot the standardevent of all channels.

        :param type: Either stdevent for events or stdevent_tp for testpulses.
        :type type: string
        :param channel: If chosen, only this channel is plotted.
        :type channel: int
        :param title: A title for the plot.
        :type title: string
        :param show_fit: If True then also plot the parametric fit.
        :type show_fit: bool
        :param block: If False the matplotlib generated figure window does not block
            the futher code execution.
        :type block: bool
        :param sample_length: The length of a sample in milliseconds. If None, it is calcualted from the sample frequency.
        :type sample_length: float
        :param show: If set, the plots are shown.
        :type show: bool
        :param save_path: If set, the plots are save to this directory.
        :type save_path: string
        :param name_appendix: A string that is appended to the group name standardevent. Typically this is _tp in case
            we want to plot a test pulse standardevent.
        :type name_appendix: string
        :param dpi: The dots per inch of the plot.
        :type dpi: int
        """

        if sample_length is None:
            sample_length = 1 / self.sample_frequency * 1000

        with h5py.File(self.path_h5, 'r') as f:
            sev = f[type + name_appendix]['event']
            sev_fitpar = f[type + name_appendix]['fitpar']

            t = (np.arange(0, self.record_length, dtype=float) - self.record_length / 4) * sample_length

            # plot
            use_cait_style(dpi=dpi)
            plt.close()

            if channel is None:

                for i, ch in enumerate(self.channel_names):
                    plt.subplot(self.nmbr_channels, 1, i + 1)
                    if not show_fit:
                        plt.plot(t, sev[i], color=self.colors[i], zorder=10, linewidth=3, label='Standardevent')
                    else:
                        plt.plot(t, sev[i], color=self.colors[i], zorder=10, linewidth=3, alpha=0.5,
                                 label='Standardevent')
                        plt.plot(t, pulse_template(t, *sev_fitpar[i]), color='black', alpha=0.7, zorder=10, linewidth=2,
                                 label='Parametric Fit')
                    make_grid()
                    plt.xlabel('Time (ms)')
                    plt.ylabel('Amplitude (V)')
                    plt.legend()
                    if title is None:
                        plt.title('Channel ' + str(ch) + ' ' + type + name_appendix)
                    else:
                        plt.title(title)

            else:

                if not show_fit:
                    plt.plot(t, sev[channel], color=self.colors[channel], zorder=10, linewidth=3, label='Standardevent')
                else:
                    plt.plot(t, sev[channel], color=self.colors[channel], zorder=10, linewidth=3, alpha=0.5,
                             label='Standardevent')
                    plt.plot(t, pulse_template(t, *sev_fitpar[channel]), color='black', alpha=0.7, zorder=10,
                             linewidth=2,
                             label='Parametric Fit')
                make_grid()
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (V)')
                plt.legend()
                if title is None:
                    plt.title('Channel ' + str(channel) + ' ' + type + name_appendix)
                else:
                    plt.title(title)

            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show(block=block)

    def show_exceptional_sev(self,
                             naming,
                             title=None,
                             show_fit=True,
                             block=True,
                             sample_length=None,
                             show=True,
                             save_path=None,
                             dpi=150):
        """
        Plot an exceptional standardevent.

        :param naming: The naming of the event, must match the group in the h5 data set,
            e.g. "carrier" --> group name "stdevent_carrier"
        :type naming: string
        :param title: A title for the plot.
        :type title: string
        :param show_fit: If True then also plot the parametric fit.
        :type show_fit: bool
        :param block: If False the matplotlib generated figure window does not block
            the futher code execution
        :type block: bool
        :param sample_length: The length of a sample in milliseconds. If None, it is calcualted from the sample frequency.
        :type sample_length: float
        :param show: If set, the plots are shown.
        :type show: bool
        :param save_path: If set, the plots are save to this directory.
        :type save_path: string
        :param dpi: The dots per inch of the plot.
        :type dpi: int
        """

        if sample_length is None:
            sample_length = 1 / self.sample_frequency * 1000

        with h5py.File(self.path_h5, 'r') as f:
            sev = np.array(f['stdevent_{}'.format(naming)]['event'])
            sev_fitpar = np.array(f['stdevent_{}'.format(naming)]['fitpar'])

            t = (np.arange(0, self.record_length, dtype=float) - self.record_length / 4) * sample_length

            # plot
            use_cait_style(dpi=dpi)
            plt.close()

            if not show_fit:
                plt.plot(t, sev, zorder=10, linewidth=3, label='Standardevent')
            else:
                plt.plot(t, sev, color='red', zorder=10, linewidth=3, alpha=0.5, label='Standardevent')
                plt.plot(t, pulse_template(t, *sev_fitpar), linewidth=2, color='black', alpha=0.7, zorder=10,
                         label='Parametric Fit')
            make_grid()
            plt.legend()
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude (V)')
            if title is None:
                plt.title('stdevent_{}'.format(naming))
            else:
                plt.title(title)

            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show(block=block)

    # Plot the NPS
    def show_nps(self,
                 channel=None,
                 title=None,
                 block=True,
                 show=True,
                 save_path=None,
                 xran=None,
                 yran=None,
                 dpi=150):
        """
        Plot the Noise Power Spectrum.

        :param channel: If chosen, only this channel is plotted.
        :type channel: int
        :param title: A title for the plot.
        :type title: string
        :param block: If False the matplotlib generated figure window does not block
            the futher code execution.
        :type block: bool
        :param show: If set, the plots are shown.
        :type show: bool
        :param save_path: If set, the plots are save to this directory.
        :type save_path: string
        :param xran: The range of the x axis.
        :type xran: tuple of two floats
        :param yran: The range of the y axis.
        :type yran: tuple of two floats
        :param dpi: The dots per inch of the plot.
        :type dpi: int
        """
        with h5py.File(self.path_h5, 'r') as f:

            # plot
            use_cait_style(dpi=dpi)
            plt.close()

            if channel is None:

                for i, ch in enumerate(self.channel_names):
                    plt.subplot(self.nmbr_channels, 1, i + 1)
                    plt.loglog(np.array(f['noise']['freq']), np.array(f['noise']['nps'][i]), color=self.colors[i],
                               zorder=10, linewidth=3)
                    make_grid()
                    if title is None:
                        plt.title('Channel ' + str(ch) + ' NPS')
                    else:
                        plt.title(title)
                    plt.ylabel('Amplitude (a.u.)')
                plt.xlabel('Frequency (Hz)')

            else:
                plt.loglog(np.array(f['noise']['freq']), np.array(f['noise']['nps'][channel]),
                           color=self.colors[channel], zorder=10,
                           linewidth=3)
                make_grid()
                if title is None:
                    plt.title('Channel ' + str(channel) + ' NPS')
                else:
                    plt.title(title)
                plt.ylabel('Amplitude (a.u.)')
                plt.xlabel('Frequency (Hz)')

            if xran is not None:
                plt.xlim(xran)
            if yran is not None:
                plt.ylim(yran)

            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show(block=block)

    # Plot the OF
    def show_of(self,
                channel=None,
                title=None,
                block=True,
                show=True,
                group_name_appendix='',
                save_path=None,
                down=None,
                xran=None,
                yran=None,
                dpi=150):
        """
        Plot the Optimum Filter.

        :param channel: If chosen, only this channel is plotted.
        :type channel: int
        :param title: A title for the plot.
        :type title: string
        :param block: If False the matplotlib generated figure window does not block
            the futher code execution.
        :type block: bool
        :param show: If set, the plots are shown.
        :type show: bool
        :param group_name_appendix: A string that is appended to the group name optimumfilter. Typically this is _tp in case
            we want to plot a test pulse standardevent.
        :type group_name_appendix: string
        :param save_path: If set, the plots are save to this directory.
        :type save_path: string
        :param down: The downsample factor of the optimum filter. This is appended to the name of the data sets in the Hdf5 set.
        :type down: int
        :param xran: The range of the x axis.
        :type xran: tuple of two floats
        :param yran: The range of the y axis.
        :type yran: tuple of two floats
        :param dpi: The dots per inch of the plot.
        :type dpi: int
        """
        with h5py.File(self.path_h5, 'r') as f:

            if down is None:
                of = np.array(f['optimumfilter' + group_name_appendix]['optimumfilter_real']) + \
                     1j * np.array(f['optimumfilter' + group_name_appendix]['optimumfilter_imag'])
            else:
                of = np.array(f['optimumfilter' + group_name_appendix]['optimumfilter_real_down{}'.format(down)]) + \
                     1j * np.array(f['optimumfilter' + group_name_appendix]['optimumfilter_imag_down{}'.format(down)])
            of = np.abs(of) ** 2

            freq = f['noise']['freq']

            if down is not None:
                first = np.array([freq[0]])
                freq = np.mean(freq[1:].reshape(int(len(freq) / down), down), axis=1)
                freq = np.concatenate((first, freq), axis=0)

            # plot
            use_cait_style(dpi=dpi)
            plt.close()

            if channel is None:

                for i, ch in enumerate(self.channel_names):
                    plt.subplot(self.nmbr_channels, 1, i + 1)
                    plt.loglog(np.array(freq), np.array(of[i]), color=self.colors[i], zorder=10, linewidth=3)
                    make_grid()
                    plt.ylabel('Amplitude (a.u.)')
                    if title is None:
                        plt.title('Channel ' + str(ch) + ' OF')
                    else:
                        plt.title(title)
                plt.xlabel('Frequency (Hz)')
            else:
                plt.loglog(np.array(freq), np.array(of[channel]), color=self.colors[channel], zorder=10, linewidth=3)
                make_grid()
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Amplitude (a.u.)')
                if title is None:
                    plt.title('Channel ' + str(channel) + ' OF')
                else:
                    plt.title(title)

            if xran is not None:
                plt.xlim(xran)
            if yran is not None:
                plt.ylim(yran)

            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show(block=block)

    def show_efficiency(self,
                        channel=0,
                        cut_flag=None,
                        which_quantity='true_ph',
                        bins=100,
                        title=None,
                        xlabel=None,
                        ylabel=None,
                        show=True,
                        plot=True,
                        xran=None,
                        yran=None,
                        block=True,
                        save_path=None,
                        dpi=150,
                        range=None,
                        xscale='linear',
                        ):
        """
        Calculate the cut efficiency for a given cut flag and plot it.

        :param channel: The cut efficiency is calculated and plotted for this channel.
        :type channel: int
        :param cut_flag: The cut values that are used for the cut efficiency calculation.
        :type cut_flag: list of bools
        :param which_quantity: Either 'true_ph', 'ph', 'of', 'sef' or 'recoil_energy'. The method that is used for the
            pulse height estimation.
        :type which_quantity: string
        :param bins: The number of bins in which we calculate the efficiency.
        :type bins: int
        :param title: A title for the plot.
        :type title: string
        :param xlabel: A label for the x axis.
        :type xlabel: string
        :param ylabel: A label for the y axis.
        :type ylabel: string
        :param show: If set, the plots are shown.
        :type show: bool
        :param plot: Do a plot of the cut efficiency. Otherwise, only the values are returned.
        :type plot: bool
        :param xran: The range of the x axis.
        :type xran: tuple of two floats
        :param yran: The range of the y axis.
        :type yran: tuple of two floats
        :param block: If False the matplotlib generated figure window does not block
            the futher code execution.
        :type block: bool
        :param save_path: If set, the plots are save to this directory.
        :type save_path: string
        :param dpi: The dots per inch of the plot.
        :type dpi: int
        :param range: The range in which the histogram is calculated. This should be maximally as large as the interval
            in which the pulses are simulated.
        :type range: tuple of two floats
        :param xscale: Either 'linear' or 'log'. The binning of the x axis.
        :type xscale: string
        :return: The efficiency within the bins, the number of counts within the bins, the bin edges.
        :rtype: tuple of (array of length bins, array of length bins, array of length bins+1)
        """

        if type(range) == tuple:
            range = list(range)

        with h5py.File(self.path_h5, 'r+') as hf5:
            if which_quantity == 'true_ph':
                vals = hf5['events']['true_ph'][channel]
            elif which_quantity == 'ph':
                vals = hf5['events']['mainpar'][channel, :, 0]  # zero is the pulse height
            elif which_quantity == 'of':
                vals = hf5['events']['of_ph'][channel]
            elif which_quantity == 'sef':
                vals = hf5['events']['sev_fit_par'][channel, :, 0]  # zero is the fitted height
            elif which_quantity == 'recoil_energy':
                vals = hf5['events']['recoil_energy'][channel]
            else:
                try:
                    vals = hf5['events'][which_quantity][channel]
                except:
                    raise KeyError(f'A dataset with name {which_quantity} is not in the HDF5 set.')

        if range is None:
            range = [np.min(vals), np.max(vals)]

        if xscale == 'linear':
            bins = np.linspace(range[0], range[1], bins + 1)
        elif xscale == 'log':
            if range is not None and range[0] == 0:
                print('Changing lower end of range from 0 to 1e-3!')
                range[0] = 1e-3
            bins = np.logspace(start=np.log10(range[0]), stop=np.log10(range[1]), num=bins + 1)
        else:
            raise ValueError('The argument of xscale must be either linear or log!')

        all, bins = np.histogram(vals, bins=bins)
        surviving, _ = np.histogram(vals[cut_flag], bins=bins)

        if np.any(all == 0):
            empties = bins[:-1] + np.diff(bins) / 2
            raise ValueError(
                f'Attention, the bins {empties[all == 0]} in your uncut efficiency events is zero! Reduce number of bins, '
                'use log scale or hand more, or binned events.')
        efficiency = surviving / all

        if plot:
            use_cait_style(dpi=dpi)
            plt.close()
            plt.hist(bins[:-1] + np.diff(bins) / 2,
                     # this gives the mean values of all bins, which do each appear once
                     bins=bins,
                     weights=efficiency,  # by this we weight the bins counts (all 1) to the actual efficiency
                     zorder=10)
            make_grid()
            if xlabel is None:
                plt.xlabel('True Pulse Height')
            else:
                plt.xlabel(xlabel)
            if ylabel is None:
                plt.ylabel('Survival Probability')
            else:
                plt.ylabel(ylabel)
            if title is not None:
                plt.title(title)
            plt.xlim(xran)
            plt.ylim(yran)
            plt.xscale(xscale)
            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show(block=block)

        return efficiency, all, bins

    # plot histogram of some value
    def show_values(self,
                    group,
                    key,
                    title=None,
                    xlabel=None,
                    ylabel=None,
                    cut_flag=None,
                    idx0=None,
                    idx1=None,
                    idx2=None,
                    block=True,
                    bins=100,
                    range=None,
                    show=True,
                    xran=None,
                    yran=None,
                    save_path=None,
                    dpi=150,
                    scale='linear'):
        """
        Shows a histogram of some values from the HDF5 file

        :param group: The group index that is used in the hdf5 file,
            typically either events, testpulses or noise.
        :type group: string
        :param key: The key index of the hdf5 file, typically mainpar, fit_rms, ...; There are a few exceptional
            properties that are calculated from the main parameters and can be plotted: 'pulse_height', 'onset',
            'rise_time', 'decay_time', 'slope'.
        :type key: string
        :param title: A title for the plot.
        :type title: string
        :param xlabel: A label for the x axis.
        :type xlabel: string
        :param ylabel: A label for the y axis.
        :type ylabel: string
        :param cut_flag: A booled array that decides which values are included in the histogram.
        :type cut_flag: list of bools
        :param idx0: The first index of the array.
        :type idx0: int
        :param idx1: The second index of the array.
        :type idx1: int or None
        :param idx2: The third index of the array.
        :type idx2: int or None
        :param block: If the plot blocks the code when executed in cmd.
        :type block: bool
        :param bins: The number of bins for the histogram.
        :type bins: int
        :param range: The interval that is shown in the histogram.
        :type range: 2D tuple of floats
        :param show: If set, the plots are shown.
        :type show: bool
        :param xran: The range of the x axis.
        :type xran: tuple of two floats
        :param yran: The range of the y axis.
        :type yran: tuple of two floats
        :param save_path: If set, the plots are save to this directory.
        :type save_path: string
        :param dpi: The dots per inch of the plot.
        :type dpi: int
        :param scale: Put 'linear' for non-log plot and 'log' for log plot.
        :type scale: string
        """

        with h5py.File(self.path_h5, 'r+') as hf5:

            if key in ['pulse_height', 'onset', 'rise_time', 'decay_time', 'slope']:
                data = self.get(group, key)
            else:
                data = hf5[group][key]
            if idx0 is None and idx1 is None and idx2 is None:
                vals = data
            elif idx0 is None and idx1 is None and idx2 is not None:
                vals = data[:, :, idx2]
            elif idx0 is None and idx1 is not None and idx2 is None:
                vals = data[:, idx1]
            elif idx0 is None and idx1 is not None and idx2 is not None:
                vals = data[:, idx1, idx2]
            elif idx0 is not None and idx1 is None and idx2 is None:
                vals = data[idx0]
            elif idx0 is not None and idx1 is None and idx2 is not None:
                vals = data[idx0, :, idx2]
            elif idx0 is not None and idx1 is not None and idx2 is None:
                vals = data[idx0, idx1]
            elif idx0 is not None and idx1 is not None and idx2 is not None:
                vals = data[idx0, idx1, idx2]

            if cut_flag is not None:
                vals = vals[cut_flag]

            use_cait_style(dpi=dpi)
            plt.close()
            plt.hist(vals,
                     bins=bins,
                     range=range, zorder=10)
            make_grid()
            if xlabel is None:
                plt.xlabel('{} {} [{},{},{}]'.format(group, key, _str_empty(idx0), _str_empty(idx1), _str_empty(idx2)))
            else:
                plt.xlabel(xlabel)
            if ylabel is None:
                plt.ylabel('Counts')
            else:
                plt.ylabel(ylabel)
            if title is not None:
                plt.title(title)
            plt.xlim(xran)
            plt.yscale(scale)
            plt.ylim(yran)
            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show(block=block)

    # show scatter plot of some value
    def show_scatter(self,
                     groups,
                     keys,
                     title=None,
                     xlabel=None,
                     ylabel=None,
                     cut_flag=None,
                     idx0s=[None, None],
                     idx1s=[None, None],
                     idx2s=[None, None],
                     block=True,
                     marker='.',
                     xran=None,
                     yran=None,
                     show=True,
                     save_path=None,
                     dpi=150,
                     rasterized=False,
                     ):
        """
        Shows a scatter plot of some values from the HDF5 file

        :param groups: The group index that is used in the hdf5 file,
            typically either events, testpulses or noise; first list element applies to data on x,
            second to data on y axis.
        :type groups: list of string
        :param keys: The key index of the hdf5 file, typically mainpar, fit_rms, ...;
            first list element applies to data on x, second to data on y axis.
        :type keys: list of string
        :param title: A title for the plot.
        :type title: string
        :param xlabel: A label for the x axis.
        :type xlabel: string
        :param ylabel: A label for the y axis.
        :type ylabel: string
        :param cut_flag: A booled array that decides which values are included in the histogram.
        :type cut_flag: list of bools
        :param idx0s: The first index of the array; first list element applies to data on x,
            second to data on y axis.
        :type idx0s: list of int
        :param idx1s: The second index of the array; first list element applies to data on x,
            second to data on y axis.
        :type idx1s: list of int or None
        :param idx2s: The third index of the array; first list element applies to data on x,
            second to data on y axis.
        :type idx2s: list of int or None
        :param block: If the plot blocks the code when executed in cmd.
        :type block: bool
        :param marker: The marker type from pyplots scatter plot.
        :type marker: string
        :param xran: The range of the x axis.
        :type xran: tuple of two floats
        :param yran: The range of the y axis.
        :type yran: tuple of two floats
        :param show: If set, the plots are shown.
        :type show: bool
        :param save_path: If set, the plots are save to this directory.
        :type save_path: string
        :param dpi: The dots per inch of the plot.
        :type dpi: int
        :param rasterized: If activated, the scatter plot is done rasterized.
        :type rasterized: bool
        """

        with h5py.File(self.path_h5, 'r+') as hf5:
            vals = []

            for i in [0, 1]:
                if idx0s[i] is None and idx1s[i] is None and idx2s[i] is None:
                    vals.append(hf5[groups[i]][keys[i]])
                elif idx0s[i] is None and idx1s[i] is None and idx2s[i] is not None:
                    vals.append(hf5[groups[i]][keys[i]][:, :, idx2s[i]])
                elif idx0s[i] is None and idx1s[i] is not None and idx2s[i] is None:
                    vals.append(hf5[groups[i]][keys[i]][:, idx1s[i]])
                elif idx0s[i] is None and idx1s[i] is not None and idx2s[i] is not None:
                    vals.append(hf5[groups[i]][keys[i]][:, idx1s[i], idx2s[i]])
                elif idx0s[i] is not None and idx1s[i] is None and idx2s[i] is None:
                    vals.append(hf5[groups[i]][keys[i]][idx0s[i]])
                elif idx0s[i] is not None and idx1s[i] is None and idx2s[i] is not None:
                    vals.append(hf5[groups[i]][keys[i]][idx0s[i], :, idx2s[i]])
                elif idx0s[i] is not None and idx1s[i] is not None and idx2s[i] is None:
                    vals.append(hf5[groups[i]][keys[i]][idx0s[i], idx1s[i]])
                elif idx0s[i] is not None and idx1s[i] is not None and idx2s[i] is not None:
                    vals.append(hf5[groups[i]][keys[i]][idx0s[i], idx1s[i], idx2s[i]])

                if cut_flag is not None:
                    vals[i] = vals[i][cut_flag]

            use_cait_style(dpi=dpi)
            plt.close()
            plt.scatter(vals[0],
                        vals[1],
                        marker=marker, zorder=10, rasterized=rasterized)
            make_grid()
            if xlabel is None:
                plt.xlabel('{} {} [{},{},{}]'.format(groups[0], keys[0], _str_empty(idx0s[0]), _str_empty(idx1s[0]),
                                                     _str_empty(idx2s[0])))
            else:
                plt.xlabel(xlabel)
            if ylabel is None:
                plt.ylabel('{} {} [{},{},{}]'.format(groups[1], keys[1], _str_empty(idx0s[1]), _str_empty(idx1s[1]),
                                                     _str_empty(idx2s[1])))
            else:
                plt.ylabel(ylabel)
            if title is not None:
                plt.title(title)
            plt.xlim(xran)
            plt.ylim(yran)
            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show(block=block)

    # show histogram of main parameter
    def show_hist(self,
                  title=None,
                  which_mp='pulse_height',
                  only_idx=None,
                  which_channel=0,
                  type='events',
                  which_labels=None,
                  which_predictions=None,
                  pred_model=None,
                  bins=100,
                  block=True,
                  ran=None,
                  show=True,
                  save_path=None,
                  dpi=150):
        """
        Show a histogram of main parameter values

        Attention! This method is depricated! Use show_values instead!
        """

        warnings.warn("Attention! This function is depricated! Use show_values instead!", warnings.DeprecationWarning)

        with h5py.File(self.path_h5, 'r') as f_h5:
            nmbr_mp = f_h5[type]['mainpar'].attrs[which_mp]
            par = f_h5[type]['mainpar'][which_channel, :, nmbr_mp]
            nmbr_events = len(par)

            if only_idx is None:
                only_idx = [i for i in range(nmbr_events)]
            par = par[only_idx]

            if which_labels is not None:
                pars = []
                for lab in which_labels:
                    pars.append(par[f_h5[type]['labels'][which_channel, only_idx] == lab])
            elif which_predictions is not None:
                pars = []
                for pred in which_predictions:
                    pars.append(par[f_h5[type]['{}_predictions'.format(pred_model)][which_channel, only_idx] == pred])

            # choose which mp to plot
            use_cait_style(dpi=dpi)
            plt.close()
            if which_labels is not None:
                for p, l in zip(pars, which_labels):
                    plt.hist(p,
                             bins=bins,
                             # color=self.colors[which_channel],
                             label='Label ' + str(l), alpha=0.8,
                             range=ran, zorder=10)
            elif which_predictions is not None:
                for p, l in zip(pars, which_predictions):
                    plt.hist(p,
                             bins=bins,
                             # color=self.colors[which_channel],
                             label='Prediction ' + str(l), alpha=0.8,
                             range=ran, zorder=10)
            else:
                plt.hist(par,
                         bins=bins,
                         # color=self.colors[which_channel],
                         range=ran, zorder=10)
            make_grid()
            plt.ylabel('Counts')
            plt.xlabel(type + ' ' + which_mp + ' Channel ' + str(which_channel))
            if title is not None:
                plt.title(title)
            plt.legend()

            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show(block=block)

            print('Histogram for {} created.'.format(which_mp))

    # show light yield plot
    def show_ly(self,
                title=None,
                xlabel=None,
                ylabel=None,
                which_method='ph',
                x_channel=0,
                y_channel=1,
                xlim=None,
                ylim=None,
                only_idx=None,
                type='events',
                which_labels=None,
                good_y_classes=None,
                which_predictions=None,
                pred_model=None,
                block=True,
                marker='.',
                alpha=0.8,
                s=10,
                show=True,
                save_path=None,
                dpi=150,
                name_appendix=''):
        """
        Make a Light Yield Plot out of specific Labels or Predictions.

        The Light Yield Parameter was described in "CRESST Collaboration, First results from the CRESST-III low-mass dark matter program"
        (10.1103/PhysRevD.100.102002).

        :param title: A title for the plot.
        :type title: string
        :param xlabel: A label for the x axis.
        :type xlabel: string
        :param ylabel: A label for the y axis.
        :type ylabel: string
        :param which_method: Either ph, sef or of. The pulse height estimation method that is used for the plot.
        :type which_method: string
        :param x_channel: The number of the channel that PHs are on the x axis.
        :type x_channel: int
        :param y_channel: The number of the channel that PHs are on the y axis.
        :type y_channel: int
        :param xlim: The range of the x axis.
        :type xlim: tuple of two floats
        :param ylim: The range of the y axis.
        :type ylim: tuple of two floats
        :param only_idx: If set only these indices are used.
        :type only_idx: list of ints or None
        :param type: Either events or testpulses.
        :type type: string
        :param which_labels: The labels that are used in the plot.
        :type which_labels: list of ints
        :param good_y_classes: If set events with y class other than
            in that list are not used in the plot.
        :type good_y_classes: list of ints or None
        :param which_predictions: The predictions that are used in the plot.
        :type which_predictions: list of ints or None
        :param pred_model: The naming of the model from that the predictions are.
        :type pred_model: string
        :param block: If the plot blocks the code when executed in cmd.
        :type block: bool
        :param marker: The marker type from pyplots scatter plot.
        :type marker: string
        :param alpha: The transparency factor of the scatter objects. Between 0 and 1.
        :type alpha: float
        :param s: The size parameter of the scatter objects.
        :type s: int
        :param show: If set, the plots are shown.
        :type show: bool
        :param save_path: If set, the plots are save to this directory.
        :type save_path: string
        :param dpi: The dots per inch of the plot.
        :type dpi: int
        :param name_appendix: A string that is appended to the dataset of the pulse height estimation method. This is
            typically _downX if pulse height estimations were calculated with downsampling.
        :type name_appendix: string
        """

        with h5py.File(self.path_h5, 'r') as f_h5:
            if which_method == 'ph':
                x_par = f_h5[type]['mainpar' + name_appendix][x_channel, :, 0]
                y_par = f_h5[type]['mainpar' + name_appendix][y_channel, :, 0]
            elif which_method == 'sef':
                x_par = f_h5[type]['sev_fit_par' + name_appendix][x_channel, :, 0]
                y_par = f_h5[type]['sev_fit_par' + name_appendix][y_channel, :, 0]
            elif which_method == 'of':
                x_par = f_h5[type]['of_ph' + name_appendix][x_channel]
                y_par = f_h5[type]['of_ph' + name_appendix][y_channel]
            else:
                raise NotImplementedError('This method is not implemented.')

            nmbr_events = len(x_par)

            if only_idx is None:
                only_idx = [i for i in range(nmbr_events)]
            x_par = x_par[only_idx]
            y_par = y_par[only_idx]

            if which_labels is not None:
                x_pars = []
                y_pars = []
                for lab in which_labels:
                    if good_y_classes is None:
                        condition = f_h5[type]['labels'][x_channel, only_idx] == lab
                    else:
                        condition = [e in good_y_classes for e in f_h5[type]['labels'][y_channel, only_idx]]
                        condition = np.logical_and(f_h5[type]['labels'][x_channel, only_idx] == lab,
                                                   condition)
                    x_pars.append(x_par[condition])
                    y_pars.append(y_par[condition])
            elif which_predictions is not None:
                x_pars = []
                y_pars = []
                for pred in which_predictions:
                    if good_y_classes is None:
                        condition = f_h5[type]['{}_predictions'.format(pred_model)][x_channel, only_idx] == pred
                    else:
                        condition = [e in good_y_classes for e in
                                     f_h5[type]['{}_predictions'.format(pred_model)][y_channel, only_idx]]
                        condition = np.logical_and(
                            f_h5[type]['{}_predictions'.format(pred_model)][x_channel, only_idx] == pred,
                            condition)
                    x_pars.append(x_par[condition])
                    y_pars.append(y_par[condition])

            # choose which mp to plot
            use_cait_style(dpi=dpi)
            plt.close()
            if which_labels is not None:
                for xp, yp, l in zip(x_pars, y_pars, which_labels):
                    plt.scatter(xp,
                                yp / xp,
                                marker=marker,
                                label='Label ' + str(l),
                                alpha=alpha,
                                s=s, zorder=10)
            elif which_predictions is not None:
                for xp, yp, l in zip(x_pars, y_pars, which_predictions):
                    plt.scatter(xp,
                                yp / xp,
                                marker=marker,
                                label='Prediction ' + str(l),
                                alpha=alpha,
                                s=s, zorder=10)
            else:
                plt.scatter(x_par,
                            y_par / x_par,
                            marker=marker,
                            alpha=alpha,
                            s=s, zorder=10)
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
            make_grid()
            if xlabel is None:
                plt.xlabel('Amplitude Ch ' + str(x_channel) + ' (V)')
            else:
                plt.xlabel(xlabel)
            if ylabel is None:
                plt.ylabel('Amplitude Ch ' + str(y_channel) + ' / Amplitude Ch ' + str(x_channel))
            else:
                plt.ylabel(ylabel)
            plt.title(title)
            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show(block=block)

            print('LY Plot created.')

    def show_saturation(self,
                        show_fit=True,
                        channel=0,
                        marker='.',
                        s=1,
                        alpha=1,
                        only_idx=None,
                        title=None,
                        block=True,
                        show=True,
                        save_path=None,
                        dpi=150,
                        method: str = 'ph',
                        name_appendix_tp: str = '',
                        ):
        """
        Plot the testpulse amplitudes vs their pulse heights and the fitted logistic curve.

        This method was used to describe the detector saturation in "M. Stahlberg, Probing low-mass dark matter with
        CRESST-III : data analysis and first results",
        available via https://doi.org/10.34726/hss.2021.45935 (accessed on the 9.7.2021).

        :param show_fit: If true show the fitted logistics curve.
        :type show_fit: bool
        :param channel: The channel for that we want to plot the saturation curve.
        :param channel: int
        :param marker: The marker type from pyplots scatter plot.
        :type marker: string
        :param s: The size parameter of the scatter objects.
        :type s: int
        :param alpha: The transparency factor of the scatter objects. Between 0 and 1.
        :type alpha: float
        :param only_idx: Only these indices are used in the fit of the saturation.
        :type only_idx: list of ints
        :param title: A title for the plot.
        :type title: string
        :param block: If the plot blocks the code when executed in cmd.
        :type block: bool
        :param show: If set, the plots are shown.
        :type show: bool
        :param save_path: If set, the plots are save to this directory.
        :type save_path: string
        :param dpi: The dots per inch of the plot.
        :type dpi: int
        :param method: Either 'ph' (main parameter pulse height), 'of' (optimum filter) or 'sef' (standard event fit).
            Test pulse heights and event heights are then estimated with this method.
        :type method: string
        :param name_appendix_tp: This is appended to the test pulse height estimation method, e.g. '_down16'.
        :type name_appendix_tp: string
        """

        with h5py.File(self.path_h5, 'r') as f_h5:

            if only_idx is None:
                only_idx = list(range(len(f_h5['testpulses']['testpulseamplitude'])))

            tpa = f_h5['testpulses']['testpulseamplitude']
            if len(tpa.shape) > 1:
                tpa = tpa[channel]
            if method == 'ph':
                ph = f_h5['testpulses']['mainpar' + name_appendix_tp][channel, only_idx, 0]
            elif method == 'of':
                ph = f_h5['testpulses']['of_ph' + name_appendix_tp][channel, only_idx]
            elif method == 'sef':
                ph = f_h5['testpulses']['sev_fit_par' + name_appendix_tp][channel, only_idx, 0]
            elif method == 'arrf':
                ph = f_h5['testpulses']['arr_fit_par' + name_appendix_tp][channel, only_idx, 0]
            else:
                raise KeyError('Pulse Height Estimation method not implemented, try ph, of or sef.')

            x = np.linspace(0, np.max(tpa))

            use_cait_style(dpi=dpi)
            plt.close()
            plt.scatter(tpa, ph,
                        marker=marker, s=s, label='TPA vs PH', zorder=10, alpha=alpha)
            if show_fit:
                fitpar = f_h5['saturation']['fitpar'][channel]
                plt.plot(x, logistic_curve(x, *fitpar), color='red', alpha=0.5,
                         label='Fitted Log Curve', zorder=10)
            make_grid()
            plt.xlabel('Test Pulse Amplitude (V)')
            plt.ylabel('Pulse Height (V)')
            if title is None:
                plt.title('Saturation Ch {}'.format(channel))
            else:
                plt.title(title)
            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show(block=block)
