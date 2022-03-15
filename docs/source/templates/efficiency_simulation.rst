*************************
Efficiency Simulation
*************************

.. code:: python

    """
    efficiency_simulation.py

    The main script for an efficiency simulation, based on CSMPL files.

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
    import pickle
    import os
    from trigger_utils import read_hw_data, get_filestart

    if __name__ == '__main__':

        # ---------------------------------------------
        # Read command line arguments
        # ---------------------------------------------

        # Construct the argument parser
        ap = argparse.ArgumentParser()

        # Add the REQUIRED arguments to the parser
        ap.add_argument("-a", "--name_stream_h5", type=str, help="Name of the stream HDF5 file, e.g. stream_008.")
        ap.add_argument("-l", "--file_list", type=str, nargs='+', default=None, help="List of the file numbers.")
        ap.add_argument("-p", "--rdt_path", type=str, help="Path to hardware data, e.g. /eos/vbc/group/darkmatter/cresst/gsdata/hwtrig/Run36/bck/")
        ap.add_argument("-s", "--csmpl_path", type=str, help="Path to stream data, e.g. /eos/vbc/group/darkmatter/cresst/gsdata/cstream/Run36/data/")
        ap.add_argument("-x", "--xy_files", type=str, help="Path to the folder that contains filter, nps, sev, sev mainpar and sev fitpar files.")
        ap.add_argument("-y", "--path_pulser_model", type=str, help="Path to the pulser models - the naming must be <path_you_put>_<file_nmbr>.pm ; e.g. /.../Li1_040.pm")
        ap.add_argument("-z", "--path_h5", type=str, help="Path where the HDF5 files should be stored.")

        # Add the OPTIONAL arguments to the parser
        ap.add_argument("-b", "--nmbr_events", type=int, default=40000, help="Number of events to simulate per file.")
        ap.add_argument("-c", "--rdt_channels", type=int, default=[0, 1],  nargs='+', help="List of the rdt channels of this module, e.g. 12 13.")
        ap.add_argument("-d", "--csmpl_channels", type=int, default=[0, 1], nargs='+', help="List of the csmpl channels of this module, e.g. 12 13.")
        ap.add_argument("-e", "--max_height", type=float, default=[3.5, 2.5],  nargs='+', help="Maximal event height to simulate.")
        ap.add_argument("-f", "--min_height", type=float, default=[0.0015, 0.0035],  nargs='+', help="Minimal event height to simualte.")
        ap.add_argument("-g", "--no_features", action='store_true', help="Skip the calculation of features and do only the triggering (not recommended).")
        ap.add_argument("-i", "--run_nmbr", type=str, default='36', help="Number of the Run, e.g. 36.")
        ap.add_argument("-k", "--no_simulation", action='store_true', help="Skip the file simulation, do only the feature calculations.")
        ap.add_argument("--no_ecal", action='store_true', help="Skip the energy calibration, e.g. if you dont have CPE factors.")
        ap.add_argument("-m", "--merge", action='store_true', help="Only merge the files list. You need to call this once you converted all the files.")
        ap.add_argument("-n", "--naming", type=str, default='bck', help="Naming of the csmpl stream files, e.g. bcl or ncal.")
        ap.add_argument("-o", "--out_name", type=str, default='efficiency', help="Naming of the output file, e.g. efficiency. This will always be combined with the naming, to obtain unique file names.")
        ap.add_argument("-q", "--sample_frequency", type=int, default=25000, help="The sample frequency.")
        ap.add_argument("-r", "--record_length", type=int, default=16384, help="The length of the record window.")
        ap.add_argument("-t", "--trigger_thresholds", type=float, default=[5.428, 11.017], nargs='+', help="List of the trigger threshold for all channels in mV, e.g. 5.428 11.017.")
        ap.add_argument("-u", "--truncation_levels", type=float, default=[0.9, 1.5], nargs='+', help="List of the truncation levels for all channels in V, e.g. 0.9 1.5.")
        ap.add_argument("-v", "--processes", type=int, default=4, help="The number of processes to use for the sev fit.")
        ap.add_argument("-w", "--uncorrelated", action='store_true', help="Do the height evaluations uncorrelated, if not activated the first channel is dominant, i.e. evaluate the height in the other channels at the maximum position of the first channel.")
        ap.add_argument("--cpe_factors", type=float, default=[2.956196019429851, 18.24242689975974],  nargs='+', help="List of the CPE factors for all channels.")

        args = vars(ap.parse_args())

        # ---------------------------------------------
        # Constants and Paths
        # ---------------------------------------------

        THRESHOLDS = np.array(args['trigger_thresholds']) * 0.001

        discrete_ph = np.array([np.logspace(start=np.log10(mi), stop=np.log10(ma), num=args['nmbr_events']) for mi,ma in zip(args['min_height'], args['max_height'])])

        datasets = {
            # 'event': 1,
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
            'recoil_energy_true': 1,
            'recoil_energy_sigma_true': 1,
            'tpa_equivalent_true': 1,
            'tpa_equivalent_sigma_true': 1,
            'recoil_energy_reconstructed': 1,
            'recoil_energy_sigma_reconstructed': 1,
            'tpa_equivalent_reconstructed': 1,
            'tpa_equivalent_sigma_reconstructed': 1,
            'cnn_cut': 1,
            'cnn_prob': 1,
            'start_s': -1,
            'start_mus': -1,
            'stop_s': -1,
            'stop_mus': -1,
            'sample_frequency': -1,
            'record_length': -1,
            'runtime': -1,
                   }

        merge_keywords = {
            'groups_to_merge': ['events', 'stream', 'metainfo'],
            'sets_to_merge': list(datasets.keys()),
            'concatenate_axis': list(datasets.values()),
            'continue_hours': True,
            'keep_original_files': True,
            'groups_from_a': ['optimumfilter', 'optimumfilter_tp', 'optimumfilter_direct', 'stdevent', 'stdevent_tp', 'stdevent_direct', 'noise'],
                         }

        # ---------------------------------------------
        # Get Handle to Stream Data
        # ---------------------------------------------

        dh_stream = ai.DataHandler(channels=args['rdt_channels'],
                                   record_length=args['record_length'],
                                   sample_frequency=args['sample_frequency'])

        dh_stream.set_filepath(path_h5=args['path_h5'],
                               fname=args['name_stream_h5'],
                               appendix=False)

        start_hours = dh_stream.get('metainfo', 'startstop_hours')[:, 0]

        # ---------------------------------------------
        # Get Infos from HW Data
        # ---------------------------------------------

        xy_files = read_hw_data(args)

        # ---------------------------------------------
        # Start the Loop
        # ---------------------------------------------

        for i, fn in enumerate(args['file_list']):

            print('-----------------------------------------------------')
            print('>> {} WORKING ON FILE: {}'.format(i, fn))

            if not args['merge']:
                empty_name = 'empty_' + args['naming'] + '_' + fn
                sim_name = args['out_name'] + '_' + args['naming'] + '_' + fn

                if not args['no_simulation']:

                    dh_empty = ai.DataHandler(channels=args['rdt_channels'],
                                              record_length=args['record_length'],
                                              sample_frequency=args['sample_frequency'])

                    dh_empty.set_filepath(path_h5=args['path_h5'],
                                          fname=empty_name,
                                          appendix=False)

                    csmpl_paths = [
                        args['csmpl_path'] + 'Ch' + str(c + 1) + '/' + 'Run' + args['run_nmbr'] + '_' + args['naming'] + '_' + fn + '_Ch' + str(
                            c + 1) + '.csmpl' for c in args['csmpl_channels']]

                    # --------------------------------------------------
                    # Include Test Pulse Time Stamps
                    # --------------------------------------------------

                    # include metadata
                    dh_empty.init_empty()
                    dh_empty.include_metainfo(args['rdt_path'] + args['naming'] + '_' + fn + '.par')

                    dh_empty.include_test_stamps(path_teststamps=args['rdt_path'] + args['naming'] + '_' + fn + '.test_stamps',
                                                 path_dig_stamps=args['rdt_path'] + args['naming'] + '_' + fn + '.dig_stamps',
                                          )

                    # --------------------------------------------------
                    # Include the Random Triggers Events
                    # --------------------------------------------------

                    dh_empty.include_noise_triggers(
                        nmbr=args['nmbr_events'],
                        min_distance=0.5,
                        max_distance=60,
                        max_attempts=5,
                        no_pileup=False,
                    )

                    dh_empty.include_noise_events(
                        csmpl_paths,
                        datatype='float32',
                    )

                    # ----------------------------------------------------------
                    # Include OF, SEV, NPS
                    # ----------------------------------------------------------


                    dh_empty.include_sev(sev=xy_files['sev'],
                                   fitpar=xy_files['sev_fitpar'],
                                   mainpar=xy_files['sev_mainpar'])

                    dh_empty.include_nps(nps=xy_files['nps'])

                    dh_empty.include_of(of_real=np.real(xy_files['of']),
                                  of_imag=np.imag(xy_files['of']))

                    # for tp

                    if 'sev_tp' in xy_files:

                        dh_empty.include_sev(sev=xy_files['sev_tp'],
                                       fitpar=xy_files['sev_tp_fitpar'],
                                       mainpar=xy_files['sev_tp_mainpar'],
                                       group_name_appendix='_tp')

                        dh_empty.include_of(of_real=np.real(xy_files['of_tp']),
                                      of_imag=np.imag(xy_files['of_tp']),
                                      group_name_appendix='_tp')

                    # for direct hits

                    if 'sev_direct' in xy_files:

                        dh_empty.include_sev(sev=xy_files['sev_direct'],
                                       fitpar=xy_files['sev_direct_fitpar'],
                                       mainpar=xy_files['sev_direct_mainpar'],
                                       group_name_appendix='_direct')

                    if 'of_direct' in xy_files:

                        dh_empty.include_of(of_real=np.real(xy_files['of_direct']),
                                      of_imag=np.imag(xy_files['of_direct']),
                                      group_name_appendix='_direct')


                    # --------------------------------------------------
                    # Simulate Events
                    # --------------------------------------------------

                    dh_empty.calc_bl_coefficients()

                    dh_empty.simulate_pulses(path_sim=args['path_h5'] + sim_name + '.h5',
                                          size_events=args['nmbr_events'],
                                          reuse_bl=True,
                                          ev_discrete_phs=discrete_ph,
                                          t0_interval=[-10, 0],  # in ms
                                          rms_thresholds=[1e5, 1e5],
                                          fake_noise=False)

                    # --------------------------------------------------
                    # Delete original empty set
                    # --------------------------------------------------

                    # Delete the empty bl hdf5 set
                    del dh_empty
                    print('Deleting {}.'.format(args['path_h5'] + empty_name + '.h5'))
                    os.remove(args['path_h5'] + empty_name + '.h5')

                # --------------------------------------------------
                # Include data from PAR and XY files
                # --------------------------------------------------

                dh_sim = ai.DataHandler(channels=args['rdt_channels'],
                                        record_length=args['record_length'],
                                        sample_frequency=args['sample_frequency'])

                dh_sim.set_filepath(path_h5=args['path_h5'],
                                    fname=sim_name,
                                    appendix=False)

                if not args['no_features']:

                    dh_sim.include_metainfo(args['rdt_path'] + args['naming'] + '_' + fn + '.par')

                    dh_sim.include_sev(sev=xy_files['sev'],
                       fitpar=xy_files['sev_fitpar'],
                       mainpar=xy_files['sev_mainpar'])

                    dh_sim.include_nps(nps=xy_files['nps'])

                    dh_sim.include_of(of_real=np.real(xy_files['of']),
                                  of_imag=np.imag(xy_files['of']))

                    # for tp

                    if 'sev_tp' in xy_files:

                        dh_sim.include_sev(sev=xy_files['sev_tp'],
                                       fitpar=xy_files['sev_tp_fitpar'],
                                       mainpar=xy_files['sev_tp_mainpar'],
                                       group_name_appendix='_tp')

                        dh_sim.include_of(of_real=np.real(xy_files['of_tp']),
                                      of_imag=np.imag(xy_files['of_tp']),
                                      group_name_appendix='_tp')

                    # for direct hits

                    if 'sev_direct' in xy_files:

                        dh_sim.include_sev(sev=xy_files['sev_direct'],
                                       fitpar=xy_files['sev_direct_fitpar'],
                                       mainpar=xy_files['sev_direct_mainpar'],
                                       group_name_appendix='_direct')

                    if 'of_direct' in xy_files:

                        dh_sim.include_of(of_real=np.real(xy_files['of_direct']),
                                      of_imag=np.imag(xy_files['of_direct']),
                                      group_name_appendix='_direct')


                    # --------------------------------------------------
                    # Calc Parameters
                    # --------------------------------------------------

                    dh_sim.calc_mp(type='events')
                    dh_sim.calc_additional_mp()
                    dh_sim.apply_of()

                    if 'of_direct' in xy_files:
                        dh_sim.apply_of(name_appendix_group='_direct', name_appendix_set='_direct')

                    # get the sevs with the fit parameters

                    t = dh_sim.record_window()
                    sev_array = []
                    for i,c in enumerate(args['rdt_channels']):
                        sev_array.append(ai.fit.pulse_template(t, *xy_files['sev_fitpar'][c]))

                    if 'sev_direct' in xy_files:
                        sev_direct_array = []
                        for i,c in enumerate(args['rdt_channels']):
                            sev_direct_array.append(ai.fit.pulse_template(t, *xy_files['sev_direct_fitpar'][c]))

        #             # do the fits

        #             dh_sim.apply_array_fit(processes=args['processes'],
        #                                truncation_level=args['truncation_levels'],
        #                                first_channel_dominant=not args['uncorrelated'], use_this_array=sev_array)

        #             dh_sim.apply_array_fit(processes=args['processes'],
        #                    truncation_level=args['truncation_levels'],
        #                    first_channel_dominant=False, use_this_array=sev_array)

        #             # do the fit for the direct hits

        #             if 'sev_direct' in xy_files:
        #                 dh_sim.apply_array_fit(group_name_appendix = '_direct', name_appendix = '_direct',
        #                                    processes=args['processes'],
        #                                    truncation_level=args['truncation_levels'], only_channels=[1],
        #                                    use_this_array=sev_direct_array)

                if not args['no_ecal']:

                    # --------------------------------------------------
                    # Assign Energies
                    # --------------------------------------------------

                    with open(args['path_pulser_model'] + args['naming'] + '_' + fn + '.pm', 'rb') as f:
                        pm = pickle.load(f)

                    dh_sim.calc_calibration(starts_saturation=args['max_height'],
                                            cpe_factor=args['cpe_factors'],
                                            plot=False,
                                            method='of',
                                            pulser_models=pm,
                                            name_appendix_energy='_reconstructed',
                                            use_interpolation=True,
                                            )

                    dh_sim.calc_calibration(starts_saturation=args['max_height'],
                                            cpe_factor=args['cpe_factors'],
                                            plot=False,
                                            method='true_ph',
                                            pulser_models=pm,
                                            name_appendix_energy='_true',
                                            use_interpolation=True,
                                            )

                # --------------------------------------------------
                # Apply neural network and other cuts
                # --------------------------------------------------

                ckp_path = ai.resources.get_resource_path('cnn-clf-binary-v0.ckpt')

                for c in range(len(args['rdt_channels'])):

                    for group in ['events']:

                        ai.models.nn_predict(h5_path=dh_sim.path_h5,
                                   model=ai.models.CNNModule.load_from_checkpoint(ckp_path),
                                   feature_channel=c,
                                   group_name=group,
                                   prediction_name='cnn_cut',
                                   keys=['event'],
                                   no_channel_idx_in_pred=False,
                                   use_prob=False)

                        ai.models.nn_predict(h5_path=dh_sim.path_h5,
                                   model=ai.models.CNNModule.load_from_checkpoint(ckp_path),
                                   feature_channel=c,
                                   group_name=group,
                                   prediction_name='cnn_prob',
                                   keys=['event'],
                                   no_channel_idx_in_pred=False,
                                   use_prob=True)

                # --------------------------------------------------
                # Delete Raw Events
                # --------------------------------------------------

                dh_sim.drop_raw_data(type='events')

            # --------------------------------------------------
            # Merge the files
            # --------------------------------------------------

            if i > 0 and args['merge']:

                merge_keywords_ = merge_keywords.copy()

                merge_keywords_['path_h5_a'] = args['path_h5'] + args['out_name'] + '_' + args['naming'] + '_{}.h5'.format(
                    args['file_list'][0]) if i == 1 else args['path_h5'] + args['out_name'] + '_{:03d}.h5'.format(i - 1)
                merge_keywords_['a_name'] = args['out_name'] + '_' + args['naming'] + '_{}'.format(args['file_list'][0]) if i == 1 else 'keep'
                merge_keywords_['path_h5_b'] = args['path_h5'] + args['out_name'] + '_' + args['naming'] + '_{}.h5'.format(fn)
                merge_keywords_['b_name'] = args['out_name'] + '_' + args['naming'] + '_{}'.format(fn)
                merge_keywords_['path_h5_merged'] = args['path_h5'] + args['out_name'] + '_{:03d}.h5'.format(i)

                start_a = get_filestart(merge_keywords_['path_h5_a'], args)
                start_b = get_filestart(merge_keywords_['path_h5_b'], args)

                merge_keywords_['second_file_start'] = (start_b[0] + 1e-6*start_b[1] - start_a[0] - 1e-6*start_a[1])/3600

                ai.data.merge_h5_sets(verb=False,
                                      **merge_keywords_,
                                      )

        # ---------------------------------------------
        # Finishing Notes
        # ---------------------------------------------

        print('-----------------------------------------------------')
        print('>> DONE WITH ALL FILES.')
