*******************
VDAQ Viewer
*******************

.. code:: python

    """
    vdaq_viewer.py

    This script produces a plot of a stream segment from a *.bin file, recorded with the VDAQ2. Run the file with
    the -h flag, to get a description of all available arguments.

    """

    import numpy as np
    import matplotlib.pyplot as plt
    import argparse
    import cait as ai

    if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('path_bin', type=str, default='../../COSINUS_DATA/data_newOP_ncal_57Co_002.bin', help='the path to the bin file')
        parser.add_argument('--plt_sec', type=int, default=7, help='the number of seconds to plot')
        parser.add_argument('--rec_freq', type=int, default=1000000, help='the record sampling frequency')
        parser.add_argument('--plt_freq', type=int, default=2500, help='the sampling frequency in the plot')
        parser.add_argument('--start_from', type=int, default=1633, help='the second to start the plot from')
        parser.add_argument('--plt_dac_channels', type=int, nargs='+', default=[1, 3], help='the dac channels to include in the plot')
        parser.add_argument('--plt_adc_channels', type=int, nargs='+', default=[1, 2], help='the adc channels to include in the plot')
        args = vars(parser.parse_args())

        # --------------------------------------------------------
        # READ HEADER
        # --------------------------------------------------------

        # open file

        header, keys, adc_bits, dac_bits, dt_tcp = ai.trigger.read_header(args['path_bin'])

        print('ADC precision: {} bit'.format(adc_bits))
        print('DAC precision: {} bit'.format(dac_bits))
        print('Channels in file: ', keys)

        read_steps = int(args['plt_sec'] * args['rec_freq'] / header['downsamplingFactor'])
        plt_offset = int(args['start_from'] * args['rec_freq'] / header['downsamplingFactor'])

        # --------------------------------------------------------
        # READ DATA
        # --------------------------------------------------------

        data = np.fromfile(args['path_bin'], dtype=dt_tcp, count=read_steps, offset=header.nbytes + plt_offset*dt_tcp.itemsize)

        # --------------------------------------------------------
        # PLOTS
        # --------------------------------------------------------

        down_rate = int(args['rec_freq'] / header['downsamplingFactor'] / args['plt_freq'])
        plt_steps = int(read_steps / down_rate)
        plot_dac_channels = ['DAC' + str(i) for i in args['plt_dac_channels'] if 'DAC' + str(i) in keys]
        plot_adc_channels = ['ADC' + str(i) for i in args['plt_adc_channels'] if 'ADC' + str(i) in keys]

        # time axis
        t = (1 / args['plt_freq']) * np.linspace(0, plt_steps - 1, plt_steps) + args['start_from']

        # Plot Channels
        plt.close()
        fig, axes = plt.subplots(len(plot_adc_channels) + len(plot_dac_channels), 1, sharex='col')
        if len(plot_adc_channels) + len(plot_dac_channels) == 1:
            axes = [axes]
        for i, c in enumerate(plot_adc_channels):
            axes[i].plot(t, ai.data.convert_to_V(np.mean(data[c].reshape(-1, down_rate), axis=1),
                                                 bits=adc_bits, min=-20, max=20))
            axes[i].set_title(c)
        for i, c in enumerate(plot_dac_channels):
            axes[i + len(plot_adc_channels)].plot(t, ai.data.convert_to_V(np.mean(data[c].reshape(-1, down_rate), axis=1),
                                                                          bits=dac_bits, min=-20, max=20))
            axes[i + len(plot_adc_channels)].set_title(c)
        fig.supxlabel('Time (s)')
        fig.supylabel('Amplitude (V)')
        plt.tight_layout()
        plt.show()
