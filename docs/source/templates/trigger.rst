*******************
Stream Triggering
*******************

.. code:: python

    """
    trigger.py

    The main script for triggering CSMPL files.

    All parameters are explained when you run the script with -h flag. You need to have the file trigger_utils in the same directory.

    """

    # ---------------------------------------------
    # Imports
    # ---------------------------------------------

    import cait as ai
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm.auto import tqdm
    import argparse
    from trigger_utils import read_hw_data, get_filestart
    import pickle
    import os


    if __name__ == '__main__':

        # ---------------------------------------------
        # Read command line arguments
        # ---------------------------------------------

        # Construct the argument parser
        ap = argparse.ArgumentParser()

        # Add the REQUIRED arguments to the parser
        ap.add_argument("-l", "--file_list", type=str, nargs='+', default=None, help="List of the file numbers.")
        ap.add_argument("-p", "--rdt_path", type=str, help="Path to hardware data, e.g. /eos/vbc/group/darkmatter/cresst/gsdata/hwtrig/Run36/bck/")
        ap.add_argument("-s", "--csmpl_path", type=str, help="Path to stream data, e.g. /eos/vbc/group/darkmatter/cresst/gsdata/cstream/Run36/data/")
        ap.add_argument("-x", "--xy_files", type=str, help="Path to the folder that contains filter, nps, sev, sev mainpar and sev fitpar files.")
        ap.add_argument("-y", "--path_pulser_model", type=str, help="Path where the pulser models get written - the naming is <path_you_put>_<file_nmbr>.pm ; e.g. /.../Li1_040.pm")
        ap.add_argument("-z", "--path_h5", type=str, help="Path where the HDF5 files should be stored.")

        # Add the OPTIONAL arguments to the parser
        ap.add_argument("-c", "--rdt_channels", type=int, default=[0, 1],  nargs='+', help="List of the rdt channels of this module, e.g. 12 13.")
        ap.add_argument("-d", "--csmpl_channels", type=int, default=[0, 1], nargs='+', help="List of the csmpl channels of this module, e.g. 12 13.")
        ap.add_argument("-g", "--no_features", action='store_true', help="Skip the calculation of features and do only the triggering (not recommended).")
        ap.add_argument("-k", "--no_trigger", action='store_true', help="Skip the file triggering, e.g. if you already triggered and want to do onle the featurecalculation.")
        ap.add_argument("--no_ecal", action='store_true', help="Skip the energy calibration, e.g. if you dont have CPE factors.")
        ap.add_argument("-i", "--run_nmbr", type=str, default='36', help="Number of the Run, e.g. 36.")
        ap.add_argument("-m", "--merge", action='store_true', help="Only merge the files list. You need to call this once you converted all the files.")
        ap.add_argument("-n", "--naming", type=str, default='bck', help="Naming of the files.")
        ap.add_argument("-o", "--out_name", type=str, default='stream', help="Naming of the output file.")
        ap.add_argument("-q", "--sample_frequency", type=int, default=25000, help="The sample frequency.")
        ap.add_argument("-r", "--record_length", type=int, default=16384, help="The length of the record window.")
        ap.add_argument("-t", "--trigger_thresholds", type=float, default=[5.428, 11.017], nargs='+', help="List of the trigger threshold for all channels in mV, e.g. 5.428 11.017.")
        ap.add_argument("-u", "--truncation_levels", type=float, default=[0.9, 1.5], nargs='+', help="List of the truncation levels for all channels in V, e.g. 0.9 1.5.")
        ap.add_argument("-v", "--processes", type=int, default=4, help="The number of processes to use for the sev fit.")
        ap.add_argument("-w", "--uncorrelated", action='store_true', help="Do the height evaluations uncorrelated, if not activated the first channel is dominant, i.e. evaluate the height in the other channels at the maximum position of the first channel.")
        ap.add_argument("--cpe_factors", type=float, default=[2.956196019429851, 18.24242689975974], nargs='+', help="List of the CPE factors for all channels.")

        args = vars(ap.parse_args())

        # ---------------------------------------------
        # Constants and Paths
        # ---------------------------------------------

        THRESHOLDS = np.array(args['trigger_thresholds']) * 0.001

        datasets = {
            'event': 1,
            'mainpar': 1,
            'add_mainpar': 1,
            'true_ph': 1,
            'true_onset': 0,
            'of_ph': 1,
            'of_ph_direct': 1,
            'arr_fit_par': 1,
            'arr_fit_rms': 1,
            'arr_fit_par_direct': 1,
            'arr_fit_rms_direct': 1,
            'hours': 0,
            'labels': 1,
            'testpulseamplitude': 0,
            'time_s': 0,
            'time_mus': 0,
            'pulse_height': 1,
            'tp_hours': 0,
            'tp_time_mus': 0,
            'tp_time_s': 0,
            'tpa': 0,
            'trigger_hours': 0,
            'trigger_time_mus': 0,
            'trigger_time_s': 0,
            'start_s': -1,
            'start_mus': -1,
            'stop_s': -1,
            'stop_mus': -1,
            'sample_frequency': -1,
            'record_length': -1,
            'runtime': -1,
            'recoil_energy': 1,
            'recoil_energy_sigma': 1,
            'tpa_equivalent': 1,
            'tpa_equivalent_sigma': 1,
            'testpulse_stability': 1,
                   }

        merge_keywords = {
            'groups_to_merge': ['events', 'testpulses', 'controlpulses', 'stream', 'metainfo'],
            'sets_to_merge': list(datasets.keys()),
            'concatenate_axis': list(datasets.values()),
            'continue_hours': True,
            'keep_original_files': True,
            'groups_from_a': ['optimumfilter', 'optimumfilter_tp', 'optimumfilter_direct', 'stdevent', 'stdevent_tp', 'stdevent_direct', 'noise'],
                         }

        # ---------------------------------------------
        # Get Infos from HW Data
        # ---------------------------------------------

        xy_files = read_hw_data(args)

        # ---------------------------------------------
        # Start the Trigger Loop
        # ---------------------------------------------

        for i, fn in enumerate(args['file_list']):

            print('-----------------------------------------------------')
            print('>> {} WORKING ON FILE: {}'.format(i, fn))


            if not args['merge']:
                dh = ai.DataHandler(channels=args['rdt_channels'],
                                    record_length=args['record_length'],
                                    sample_frequency=args['sample_frequency'])

                dh.set_filepath(path_h5=args['path_h5'],
                                fname=args['out_name'] + '_' + args['naming'] + '_' + fn,
                                appendix=False)

                csmpl_paths = [args['csmpl_path'] + 'Ch' + str(c+1) + '/' + 'Run' + args['run_nmbr'] + '_' + args['naming'] + '_' + fn + '_Ch' + str(c+1) + '.csmpl' for c in args['csmpl_channels']]

                if not args['no_trigger']:

                    # --------------------------------------------------
                    # Trigger Files
                    # --------------------------------------------------

                    # include metadata
                    dh.init_empty()
                    dh.include_metainfo(args['rdt_path'] + args['naming'] + '_' + fn + '.par')

                    dh.include_csmpl_triggers(csmpl_paths=csmpl_paths,
                                              thresholds=THRESHOLDS,
                                              of=xy_files['of'],
                                              path_dig=args['rdt_path'] + args['naming'] + '_' + fn + '.dig_stamps',
                                              read_triggerstamps=False,
                                              )

                    # --------------------------------------------------
                    # Include Test Pulse Time Stamps
                    # --------------------------------------------------

                    dh.include_test_stamps(path_teststamps=args['rdt_path'] + args['naming'] + '_' + fn + '.test_stamps',
                                           path_dig_stamps=args['rdt_path'] + args['naming'] + '_' + fn + '.dig_stamps',
                                          )

                    # --------------------------------------------------
                    # Include Triggered Events
                    # --------------------------------------------------

                    dh.include_triggered_events(csmpl_paths=csmpl_paths,
                                                max_time_diff=0.5, # in sec - this prevents all pile up with test pulses
                                                exclude_tp=True,
                                                sample_duration=1/args['sample_frequency'],
                                                datatype='float32')

                if not args['no_features']:

                    # ----------------------------------------------------------
                    # Include OF, SEV, NPS to first set (we keep them at merge)
                    # ----------------------------------------------------------

                    dh.include_sev(sev=xy_files['sev'],
                                   fitpar=xy_files['sev_fitpar'],
                                   mainpar=xy_files['sev_mainpar'])

                    dh.include_nps(nps=xy_files['nps'])

                    dh.include_of(of_real=np.real(xy_files['of']),
                                  of_imag=np.imag(xy_files['of']))

                    # for tp

                    if 'sev_tp' in xy_files:

                        dh.include_sev(sev=xy_files['sev_tp'],
                                       fitpar=xy_files['sev_tp_fitpar'],
                                       mainpar=xy_files['sev_tp_mainpar'],
                                       group_name_appendix='_tp')

                        dh.include_of(of_real=np.real(xy_files['of_tp']),
                                      of_imag=np.imag(xy_files['of_tp']),
                                      group_name_appendix='_tp')

                    # for direct hits

                    if 'sev_direct' in xy_files:

                        dh.include_sev(sev=xy_files['sev_direct'],
                                       fitpar=xy_files['sev_direct_fitpar'],
                                       mainpar=xy_files['sev_direct_mainpar'],
                                       group_name_appendix='_direct')

                    if 'of_direct' in xy_files:

                        dh.include_of(of_real=np.real(xy_files['of_direct']),
                                      of_imag=np.imag(xy_files['of_direct']),
                                      group_name_appendix='_direct')

                    # --------------------------------------------------
                    # Calc Mainpar for Events and Testpulses
                    # --------------------------------------------------

                    dh.calc_mp(type='events')
                    dh.calc_mp(type='testpulses')
                    dh.calc_additional_mp(type='events')
                    dh.calc_additional_mp(type='testpulses')

                    # --------------------------------------------------
                    # Apply OF for Events and Testpulses
                    # --------------------------------------------------

                    dh.apply_of(first_channel_dominant=not args['uncorrelated'])
                    if 'of_tp' in xy_files:
                        dh.apply_of(type='testpulses', name_appendix_group='_tp')
                    if 'of_direct' in xy_files:
                        dh.apply_of(name_appendix_group='_direct', name_appendix_set='_direct')

                    # --------------------------------------------------
                    # Do SEV Fit for Events and Testpulses
                    # --------------------------------------------------

                    # get the sevs with the fit parameters

                    t = dh.record_window()
                    sev_array = []
                    for i,c in enumerate(args['rdt_channels']):
                        sev_array.append(ai.fit.pulse_template(t, *xy_files['sev_fitpar'][c]))

                    if 'sev_direct' in xy_files:
                        sev_direct_array = []
                        for i,c in enumerate(args['rdt_channels']):
                            sev_direct_array.append(ai.fit.pulse_template(t, *xy_files['sev_direct_fitpar'][c]))

                    if 'sev_tp' in xy_files:
                        sev_tp_array = []
                        for i,c in enumerate(args['rdt_channels']):
                            sev_tp_array.append(ai.fit.pulse_template(t, *xy_files['sev_tp_fitpar'][c]))

                    # do the fits

                    dh.apply_array_fit(processes=args['processes'],
                                       truncation_level=args['truncation_levels'],
                                       first_channel_dominant=not args['uncorrelated'], use_this_array=sev_array)

                    if 'sev_tp' in xy_files:
                        dh.apply_array_fit(type='testpulses', group_name_appendix='_tp',
                                           processes=args['processes'],
                                           truncation_level=args['truncation_levels'],
                                           use_this_array=sev_tp_array)

                    # do the fit for the direct hits

                    if 'sev_direct' in xy_files:
                        dh.apply_array_fit(group_name_appendix = '_direct', name_appendix = '_direct',
                                           processes=args['processes'],
                                           truncation_level=args['truncation_levels'], only_channels=[1],
                                           use_this_array=sev_direct_array)

                if not args['no_ecal']:

                    # --------------------------------------------------
                    # Energy calibration
                    # --------------------------------------------------

                    tp_tpa = dh.get('testpulses', 'testpulseamplitude')
                    tp_ph = dh.get('testpulses', 'pulse_height')[:, :]
                    unique_tp = np.unique(tp_tpa)
                    print('Unique testpulse heights: ', unique_tp)

                    lb = []
                    ub = []

                    for c in range(len(args['rdt_channels'])):

                        medians = [np.median(tp_ph[c][tp_tpa == tpa]) for tpa in unique_tp]
                        lower_quantiles = [np.quantile(tp_ph[c][tp_tpa == tpa], 0.18) for tpa in unique_tp]
                        upper_quantiles = [np.quantile(tp_ph[c][tp_tpa == tpa], 0.82) for tpa in unique_tp]
                        mean_deviations = [u - l for l,u in zip(lower_quantiles, upper_quantiles)]

                        lb.append([ l - 5*m for l,m in zip(lower_quantiles, mean_deviations)])
                        ub.append([ u + 5*m for u,m in zip(upper_quantiles, mean_deviations)])

                    for c in range(len(args['rdt_channels'])):
                        dh.calc_testpulse_stability(c, significance=3, ub = ub[c], lb = lb[c])

                    pm = dh.calc_calibration(starts_saturation=[1.6, 0.3],  # stop energy calibration at these values
                        cpe_factor=args['cpe_factors'],
                        plot=False,
                        only_stable=True,
                        exclude_tpas=[],
                        interpolation_method='linear',
                        method='of',
                        return_pulser_models=True,
                        use_interpolation=True,
                        )

                    with open(args['path_pulser_model'] + args['out_name'] + '_' + args['naming'] + '_' + fn + '.pm', 'wb') as f:
                        pickle.dump(pm, f)

            # --------------------------------------------------
            # Merge the files
            # --------------------------------------------------

            if i > 0 and args['merge']:

                merge_keywords_ = merge_keywords.copy()

                merge_keywords_['path_h5_a'] = args['path_h5'] + args['out_name'] + '_' + args['naming'] + '_{}.h5'.format(args['file_list'][0]) if i == 1 else args['path_h5'] + '{}_{:03d}.h5'.format(args['out_name'], i-1)
                merge_keywords_['a_name'] = args['out_name'] + '_' + args['naming'] + '_{}'.format(args['file_list'][0]) if i == 1 else 'keep'
                merge_keywords_['path_h5_b'] = args['path_h5'] + args['out_name'] + '_' + args['naming'] + '_{}.h5'.format(fn)
                merge_keywords_['b_name'] = args['out_name'] + '_' + args['naming'] + '_{}'.format(fn)
                merge_keywords_['path_h5_merged'] = args['path_h5'] + '{}_{:03d}.h5'.format(args['out_name'], i)

                start_a = get_filestart(merge_keywords_['path_h5_a'], args)
                start_b = get_filestart(merge_keywords_['path_h5_b'], args)

                merge_keywords_['second_file_start'] = (start_b[0] + 1e-6*start_b[1] - start_a[0] - 1e-6*start_a[1])/3600

                ai.data.merge_h5_sets(verb=False,
                                      **merge_keywords_,
                                      )

                if i > 1 and merge_keywords['keep_original_files']:
                    try:
                        os.remove(merge_keywords_['path_h5_a'])
                        print('File removed ', merge_keywords_['path_h5_a'])
                    except:
                        print('Could not remove file ', merge_keywords_['path_h5_a'])

        # ---------------------------------------------
        # Finishing Notes
        # ---------------------------------------------

        print('-----------------------------------------------------')
        print('>> DONE WITH ALL FILES.')

