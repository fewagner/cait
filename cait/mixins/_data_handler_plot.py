# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
import matplotlib.pyplot as plt
from ..fit._templates import pulse_template
from ..fit._saturation import logistic_curve
from ..styles._plt_styles import use_cait_style, make_grid


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

def _str_empty(value):
    if value is None:
        return ''
    else:
        return str(value)


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
                 sample_length=0.04,
                 show=True,
                 save_path=None,
                 dpi=150):
        """
        Plot the standardevent of all channels
        :param type: string, either stdevent for events or stdevent_tp for testpulses
        :param show_fit: bool, if True then also plot the parametric fit
        :param block: bool, if False the matplotlib generated figure window does not block
            the futher code execution
        :param sample_length: float, the length of a sample milliseconds
        :return: -
        """

        with h5py.File(self.path_h5, 'r') as f:
            sev = f[type]['event']
            sev_fitpar = f[type]['fitpar']

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
                        plt.title('Channel ' + str(ch) + ' ' + type)
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
                    plt.title('Channel ' + str(channel) + ' ' + type)
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
                             sample_length=0.04,
                             show=True,
                             save_path=None,
                             dpi=150):
        """
        Plot an exceptional standardevent of one channel

        :param naming: string, the naming of the event, must match the group in the h5 data set,
            e.g. "carrier" --> group name "stdevent_carrier"
        :param show_fit: bool, if True then also plot the parametric fit
        :param block: bool, if False the matplotlib generated figure window does not block
            the futher code execution
        :param sample_length: float, the length of a sample milliseconds
        :return: -
        """

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
                plt.plot(t, pulse_template(t, *sev_fitpar), linewidth=2, color='black', alpha=0.7, zorder=10, label='Parametric Fit')
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
                 dpi=150):
        """
        Plot the Noise Power Spectrum of all channels
        :param block: bool, if False the matplotlib generated figure window does not block
            the futher code execution
        :return: -
        """
        with h5py.File(self.path_h5, 'r') as f:

            # plot
            use_cait_style(dpi=dpi)
            plt.close()

            if channel is None:

                for i, ch in enumerate(self.channel_names):
                    plt.subplot(self.nmbr_channels, 1, i + 1)
                    plt.loglog(f['noise']['freq'], f['noise']['nps'][i], color=self.colors[i], zorder=10, linewidth=3)
                    make_grid()
                    if title is None:
                        plt.title('Channel ' + str(ch) + ' NPS')
                    else:
                        plt.title(title)
                    plt.ylabel('Amplitude (a.u.)')
                plt.xlabel('Frequency (Hz)')

            else:
                plt.loglog(f['noise']['freq'], f['noise']['nps'][channel], color=self.colors[channel], zorder=10,
                           linewidth=3)
                make_grid()
                if title is None:
                    plt.title('Channel ' + str(channel) + ' NPS')
                else:
                    plt.title(title)
                plt.ylabel('Amplitude (a.u.)')
                plt.xlabel('Frequency (Hz)')

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
                save_path=None,
                down=None,
                dpi=150):
        """
        Plot the Optimum Filter of all channels
        :param block: bool, if False the matplotlib generated figure window does not block
            the futher code execution
        :return: -
        """
        with h5py.File(self.path_h5, 'r') as f:

            if down is None:
                of = np.array(f['optimumfilter']['optimumfilter_real']) + \
                     1j * np.array(f['optimumfilter']['optimumfilter_imag'])
            else:
                of = np.array(f['optimumfilter']['optimumfilter_real_down{}'.format(down)]) + \
                     1j * np.array(f['optimumfilter']['optimumfilter_imag_down{}'.format(down)])
            of = np.abs(of) ** 2

            freq = f['noise']['freq']

            if down is not None:
                first = np.array([freq[0]])
                freq = np.mean(freq[1:].reshape(int(len(freq)/down), down), axis=1)
                freq = np.concatenate((first, freq), axis=0)

            # plot
            use_cait_style(dpi=dpi)
            plt.close()

            if channel is None:

                for i, ch in enumerate(self.channel_names):
                    plt.subplot(self.nmbr_channels, 1, i + 1)
                    plt.loglog(freq, of[i], color=self.colors[i], zorder=10, linewidth=3)
                    make_grid()
                    plt.ylabel('Amplitude (a.u.)')
                    if title is None:
                        plt.title('Channel ' + str(ch) + ' OF')
                    else:
                        plt.title(title)
                plt.xlabel('Frequency (Hz)')
            else:
                plt.loglog(freq, of[channel], color=self.colors[channel], zorder=10, linewidth=3)
                make_grid()
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Amplitude (a.u.)')
                if title is None:
                    plt.title('Channel ' + str(channel) + ' OF')
                else:
                    plt.title(title)

            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show(block=block)

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
                    block=False,
                    bins=100,
                    range=None,
                    show=True,
                    xran=None,
                    yran=None,
                    save_path=None,
                    dpi=150):
        """
        Shows a histogram of some values from the HDF5 file

        :param group: string, The group index that is used in the hdf5 file,
            typically either events, testpulses or noise
        :param key: string, the key index of the hdf5 file, typically mainpar, fit_rms, ...
        :param idx0: int, the first index of the array
        :param idx1: int or None, the second index of the array
        :param idx2: int or None, the third index of the array
        :param block: bool, if the plot blocks the code when executed in cmd
        :param bins: int, the number of bins for the histogram
        :param range: 2D tuple of floats, the interval that is shown in the histogram
        :return: -
        """

        with h5py.File(self.path_h5, 'r+') as hf5:

            if idx0 is None and idx1 is None and idx2 is None:
                vals = hf5[group][key]
            elif idx0 is None and idx1 is None and idx2 is not None:
                vals = hf5[group][key][:, :, idx2]
            elif idx0 is None and idx1 is not None and idx2 is None:
                vals = hf5[group][key][:, idx1]
            elif idx0 is None and idx1 is not None and idx2 is not None:
                vals = hf5[group][key][:, idx1, idx2]
            elif idx0 is not None and idx1 is None and idx2 is None:
                vals = hf5[group][key][idx0]
            elif idx0 is not None and idx1 is None and idx2 is not None:
                vals = hf5[group][key][idx0, :, idx2]
            elif idx0 is not None and idx1 is not None and idx2 is None:
                vals = hf5[group][key][idx0, idx1]
            elif idx0 is not None and idx1 is not None and idx2 is not None:
                vals = hf5[group][key][idx0, idx1, idx2]

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
                     block=False,
                     marker='.',
                     xran=None,
                     yran=None,
                     show=True,
                     save_path=None,
                     dpi=150):
        """
        Shows a scatter plot of some values from the HDF5 file

        :param groups: list of string, The group index that is used in the hdf5 file,
            typically either events, testpulses or noise; first list element applies to data on x,
            second to data on y axis
        :param keys: list of string, the key index of the hdf5 file, typically mainpar, fit_rms, ...;
            first list element applies to data on x, second to data on y axis
        :param idxs: list of int, the first index of the array; first list element applies to data on x,
            second to data on y axis
        :param idx0s: list of int or None, the second index of the array; first list element applies to data on x,
            second to data on y axis
        :param idx0s: list of int or None, the third index of the array; first list element applies to data on x,
            second to data on y axis
        :param block: bool, if the plot blocks the code when executed in cmd
        :param marker: string, the marker type from pyplots scatter plot
        :return: -
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
                        marker=marker, zorder=10)
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
                  block=False,
                  ran=None,
                  show=True,
                  save_path=None,
                  dpi=150):
        """
        Show a histogram of main parameter values

        :param which_mp: string, possible are: ['pulse_height', 't_zero', 't_rise', 't_max', 't_decaystart', 't_half', 't_end',
            'offset', 'linear_drift', 'quadratic_drift']
        :param only_idx: list of ints or None, if set only these indices are used
        :param which_channel: int, the number of the channel from that we want the histogram
        :param type: string, either events or testpulses
        :param which_labels: list or None, if set only events with these labels are included; needs a labels file
            to be included in the hdf5 set
        :param which_predictions: list or None, if set only events with these predictions are included;
            needs a predictions file to be included in the hdf5 set
        :param pred_model: string, the naming of the model that made the predictions; must match the naming of the
            predictions in the HDF5 file
        :param bins: int, the number of bins in the histogram
        :param block: bool, if False the matplotlib generated figure window does not block
            the futher code execution
        :param range: 2-tuple or None, if set the range of the histogram
        :return: -
        """

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
                block=False,
                marker='.',
                alpha=0.8,
                s=10,
                show=True,
                save_path=None,
                dpi=150):
        """
        Make a Light Yield Plot out of specific Labels or Predictions

        :param x_channel: int, the number of the channel that PHs are on the x axis
        :param y_channel: int, the number of the channel that PHs are on the y axis
        :param only_idx: list of ints or None, if set only these indices are used
        :param type: string, either events or testpulses
        :param which_labels: list of ints, the labels that are used in the plot
        :param good_y_classes: list of ints or None, if set events with y class other than
            in that list are not used in the plot
        :param which_predictions: list of ints or None, the predictions that are used in the plot
        :param pred_model: string, the naming of the model from that the predictions are
        :param block: bool, if True the plots are non blocking in the cmd
        """

        with h5py.File(self.path_h5, 'r') as f_h5:
            if which_method == 'ph':
                x_par = f_h5[type]['mainpar'][x_channel, :, 0]
                y_par = f_h5[type]['mainpar'][y_channel, :, 0]
            elif which_method == 'sef':
                x_par = f_h5[type]['sev_fit_par'][x_channel, :, 0]
                y_par = f_h5[type]['sev_fit_par'][y_channel, :, 0]
            elif which_method == 'of':
                x_par = f_h5[type]['of_ph'][x_channel]
                y_par = f_h5[type]['of_ph'][y_channel]

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

    def show_saturation(self, show_fit=True, channel=0, marker='.', s=1, alpha=1,
                        only_idx=None, title=None,
                        block=False,
                        show=True,
                        save_path=None,
                        dpi=150):
        """
        Plot the testpulse amplitudes vs their pulse heights and the fitted logistic curve

        :param show_fit: bool, if true show the fitted logistics curve
        :param channel: int, the channel for that we want to plot the saturation curve
        :param only_idx: only these indices are used in the fit of the saturation
        :type only_idx: list of ints
        :param s: float, radius of the markers in the scatter plot
        :param marker: string, the matplotlib marker in the scatter plot
        :return: -
        :rtype: -
        """

        with h5py.File(self.path_h5, 'r') as f_h5:

            if only_idx is None:
                only_idx = list(range(len(f_h5['testpulses']['testpulseamplitude'])))

            tpa = f_h5['testpulses']['testpulseamplitude']
            ph = f_h5['testpulses']['mainpar'][channel, only_idx, 0]

            x = np.linspace(0, np.max(tpa))

            use_cait_style(dpi=dpi)
            plt.close()
            plt.scatter(tpa, ph,
                        marker=marker, s=s, label='TPA vs PH', zorder=10, alpha=alpha)
            if show_fit:
                fitpar = f_h5['saturation']['fitpar'][channel]
                plt.plot(x, logistic_curve(x, *fitpar),
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
