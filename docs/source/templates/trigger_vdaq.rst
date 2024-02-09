***************************
Example of triggering VDAQ2
***************************

.. code:: python

    """
    trigger_vdaq.py

    Usage:

    python trigger_vdaq.py

    First, change the parameters in the args dictionary according to your needs.

    We use here a "mock" filter, instead of the actual optimum filter. In case you have already a filter that you want
    to use, you can load it in the indicated section in the file.

    adc_channels: List of the ADC channel numbers that we want to trigger.
    dac_channels: List of the DAC channel numbers that we want to trigger.
    record_length: The record length with which we want to trigger.
    sample_frequency: The sample frequency of the recording.
    adc_thresholds: The thresholds for all ADC channels.
    dac_thresholds: The thresholds for all DAC channels.
    path_h5: The path to the directory in which the bin file is stored.
    fname: The name of the BIN file, without extension.
    path_bin: The full path to the BIN file.
    nmbr_noise: Number of noise baselines we want to take from the file.
    """

    import cait as ai
    import numpy as np

    args = {'adc_channels': [1, 2],
            'dac_channels': [1, 3],
            'record_length': 8192,
            'sample_frequency': 100000,
            'adc_thresholds': [0.2, 0.2],
            'dac_thresholds': [0.2, 0.2],
            'path_h5': '../COSINUS_DATA/',
            'fname': 'data_newOP_ncal_57Co_002',
            'path_bin': '../COSINUS_DATA/data_newOP_ncal_57Co_002.bin',
            'nmbr_noise': 5000,
            }

    header, keys, adc_bits, dac_bits, dt_tcp = ai.trigger.read_header(args['path_bin'])

    dh = ai.DataHandler(channels=args['adc_channels'], record_length=args['record_length'], sample_frequency=args['sample_frequency'])
    dh.set_filepath(path_h5=args['path_h5'], fname=args['fname'], appendix=False)

    dh.init_empty()

    # TODO load the optimum filter here! If you have none, you can use the mock filter constructed below.
    of = [np.ones(int(args['record_length'] / 2 + 1)) for c in args['adc_channels']]

    # trigger the adc channels (data)

    dh.include_bin_triggers(path=args['path_bin'],
                            dtype=dt_tcp,
                            keys=['ADC' + str(c) for c in args['adc_channels']],
                            header_size=header.nbytes,
                            thresholds=args['adc_thresholds'],
                            adc_bits=adc_bits,
                            of=of,
                            )

    # trigger the dac channels (test pulses)

    dh.include_dac_triggers(path=args['path_bin'],
                            dtype=dt_tcp,
                            keys=['DAC' + str(c) for c in args['dac_channels']],
                            header_size=header.nbytes,
                            thresholds=args['dac_thresholds'],
                            dac_bits=dac_bits,
                            )


    # include triggered events and test pulses

    dh.include_triggered_events_vdaq(path=args['path_bin'],
                                     dtype=dt_tcp,
                                     keys=['ADC' + str(i) for i in args['adc_channels']],
                                     header_size=header.itemsize,
                                     adc_bits=adc_bits,
                                     max_time_diff=3 / 4 * args['record_length'] / args['sample_frequency'],  # in sec
                                     exclude_tp=True,
                                     min_tpa=[0.001, 0.001],
                                     min_cpa=[100.1, 100.1],
                                     )

    # include noise triggers

    dh.include_noise_triggers(nmbr=args['nmbr_noise'],
                              min_distance=3 / 4 * args['record_length'] / args['sample_frequency'],
                              max_distance=60,
                              max_attempts=5,
                              )

    # include noise events

    dh.include_noise_events_vdaq(path=args['path_bin'],
                                 dtype=dt_tcp,
                                 keys=['ADC' + str(i) for i in args['adc_channels']],
                                 header_size=header.itemsize,
                                 adc_bits=adc_bits,
                                 )