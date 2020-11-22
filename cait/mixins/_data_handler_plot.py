# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
import matplotlib.pyplot as plt
from ..fit._templates import pulse_template
from ..fit._saturation import logistic_curve


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class PlotMixin(object):
    """
    Mixin Class for the DataHandler to make essential plots for the analysis
    """

    # Plot the SEV
    def show_SEV(self,
                 type='stdevent',
                 show_fit=True,
                 block=True,
                 sample_length=0.04):
        """
        Plot the standardevent of all channels
        :param type: string, either stdevent for events or stdevent_tp for testpulses
        :param show_fit: bool, if True then also plot the parametric fit
        :param block: bool, if False the matplotlib generated figure window does not block
            the futher code execution
        :param sample_length: float, the length of a sample milliseconds
        :return: -
        """

        f = h5py.File(self.path_h5, 'r')
        sev = f[type]['event']
        sev_fitpar = f[type]['fitpar']

        t = (np.arange(0, self.record_length, dtype=float) - self.record_length / 4) * sample_length

        # plot
        plt.close()

        for i, ch in enumerate(self.channel_names):
            plt.subplot(2, 1, i + 1)
            plt.plot(t, sev[i], color=self.colors[i])
            if show_fit:
                plt.plot(t, pulse_template(t, *sev_fitpar[i]), color='orange')
            plt.title(ch + ' ' + type)

        plt.show(block=block)

        f.close()

    def show_exceptional_SEV(self,
                             naming,
                             show_fit=True,
                             block=True,
                             sample_length=0.04):
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

        f = h5py.File(self.path_h5, 'r')
        sev = np.array(f['stdevent_{}'.format(naming)]['event'])
        sev_fitpar = np.array(f['stdevent_{}'.format(naming)]['fitpar'])

        t = (np.arange(0, self.record_length, dtype=float) - self.record_length / 4) * sample_length

        # plot
        plt.close()

        plt.plot(t, sev)
        if show_fit:
            plt.plot(t, pulse_template(t, *sev_fitpar))
        plt.title('stdevent_{}'.format(naming))

        plt.show(block=block)

        f.close()

    # Plot the NPS
    def show_NPS(self, block=True):
        """
        Plot the Noise Power Spectrum of all channels
        :param block: bool, if False the matplotlib generated figure window does not block
            the futher code execution
        :return: -
        """
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
        """
        Plot the Optimum Filter of all channels
        :param block: bool, if False the matplotlib generated figure window does not block
            the futher code execution
        :return: -
        """
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
                    idx0=None,
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
        :return: -
        """

        hf5 = h5py.File(self.path_h5, 'r+')

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

        plt.close()

        plt.hist(vals,
                 bins=bins,
                 range=range)
        plt.title('{} {} {},{},{}'.format(group, key, str(idx0), str(idx1), str(idx2)))
        plt.show(block=block)

    # show scatter plot of some value
    def show_scatter(self,
                     groups,
                     keys,
                     idx0s=[None, None],
                     idx1s=[None, None],
                     idx2s=[None, None],
                     block=False,
                     marker='.'):
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

        hf5 = h5py.File(self.path_h5, 'r+')
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

        plt.close()

        plt.scatter(vals[0],
                    vals[1],
                    marker=marker)
        plt.title('(x): {} {} [{},{},{}]; (y): {} {} [{},{},{}]'.format(groups[0], keys[0],
                                                                        str(idx0s[0]), str(idx1s[0]), str(idx2s[0]),
                                                                        groups[1], keys[1],
                                                                        str(idx0s[1]), str(idx1s[1]), str(idx2s[1])
                                                                        ))
        plt.show(block=block)

    # show histogram of main parameter
    def show_hist(self,
                  which_mp='pulse_height',
                  only_idx=None,
                  which_channel=0,
                  type='events',
                  which_labels=None,
                  which_predictions=None,
                  pred_model=None,
                  bins=100,
                  block=False,
                  ran=None):
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

        f_h5 = h5py.File(self.path_h5, 'r')
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
        plt.close()
        if which_labels is not None:
            for p, l in zip(pars, which_labels):
                plt.hist(p,
                         bins=bins,
                         # color=self.colors[which_channel],
                         label='Label ' + str(l), alpha=0.8,
                         range=ran)
        elif which_predictions is not None:
            for p, l in zip(pars, which_predictions):
                plt.hist(p,
                         bins=bins,
                         # color=self.colors[which_channel],
                         label='Prediction ' + str(l), alpha=0.8,
                         range=ran)
        else:
            plt.hist(par,
                     bins=bins,
                     # color=self.colors[which_channel],
                     range=ran)
        plt.title(type + ' ' + which_mp + ' Channel ' + str(which_channel))
        plt.legend()
        plt.show(block=block)

        print('Histogram for {} created.'.format(which_mp))
        f_h5.close()

    # show light yield plot
    def show_LY(self,
                x_channel=0,
                y_channel=1,
                only_idx=None,
                type='events',
                which_labels=None,
                good_y_classes=None,
                which_predictions=None,
                pred_model=None,
                block=False,
                marker='.',
                alpha=0.8,
                s=1):
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

        f_h5 = h5py.File(self.path_h5, 'r')
        x_par = f_h5[type]['mainpar'][x_channel, :, 0]
        y_par = f_h5[type]['mainpar'][y_channel, :, 0]
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
        plt.close()
        if which_labels is not None:
            for xp, yp, l in zip(x_pars, y_pars, which_labels):
                plt.scatter(xp,
                            yp / xp,
                            marker=marker,
                            label='Label ' + str(l),
                            alpha=alpha,
                            s=s)
        elif which_predictions is not None:
            for xp, yp, l in zip(x_pars, y_pars, which_predictions):
                plt.scatter(xp,
                            yp / xp,
                            marker=marker,
                            label='Prediction ' + str(l),
                            alpha=alpha,
                            s=s)
        else:
            plt.scatter(x_par,
                        y_par / y_par,
                        marker=marker,
                        alpha=alpha,
                        s=s)
        plt.title(type + ' LY x_ch ' + str(x_channel) + ' y_ch ' + str(y_channel))
        plt.legend()
        plt.show(block=block)

        print('LY Plot created.')
        f_h5.close()

    def show_saturation(self, show_fit=True, channel=0, marker='.', s=1, only_idx=None):
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

        f_h5 = h5py.File(self.path_h5, 'r')

        if only_idx is None:
            only_idx = list(range(len(f_h5['testpulses']['testpulseamplitude'])))

        tpa = f_h5['testpulses']['testpulseamplitude']
        ph = f_h5['testpulses']['mainpar'][channel, only_idx, 0]

        x = np.linspace(0, np.max(tpa))

        plt.close()
        plt.scatter(tpa, ph,
                    marker=marker, s=s, label='TPA vs PH')
        if show_fit:
            fitpar = f_h5['saturation']['fitpar'][channel]
            plt.plot(x, logistic_curve(x, *fitpar),
                     label='Fitted Log Curve')
        plt.title('Saturation Ch {}'.format(channel))
        plt.show()

        f_h5.close()