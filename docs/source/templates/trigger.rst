*******************
Stream Triggering
*******************

.. code:: python

    """
    trigger.py

    This script is for the efficient triggering of CSMPL files.

    Usage:
    - Adapt the section 'Constants and Paths' to your measurement.
    - Adapt the section 'Apply Cuts' (around line 220) to your individual cut values.
    - If you start the script without command line arguments, it will trigger all files and merge them one after another.
    - If you start the script with the flag -f n, if will only trigger the n'th file from the list of files.
    - If you start the script with the flag -m, it will only do the merge between all files.
    - For time efficient triggering, a good workflow is to write a bash script, that starts the triggering of all files
        simultaneously with the -f flags, then call the script again with the -m flag when all files are done.
    """

    # ---------------------------------------------
    # Imports
    # ---------------------------------------------

    import cait as ai
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm.auto import tqdm
    import argparse

    if __name__ == '__main__':

        # ---------------------------------------------
        # Read command line arguments
        # ---------------------------------------------

        # Construct the argument parser
        ap = argparse.ArgumentParser()

        # Add the arguments to the parser
        ap.add_argument("-f", "--file", type=int, required=False, default=None, help="Trigger only this index from the files list.")
        ap.add_argument("-m", "--merge", action='store_true', help="Only merge the files list.")
        args = vars(ap.parse_args())

        THIS_FILE_ONLY = args['file']
        MERGE_ONLY = args['merge']

        assert THIS_FILE_ONLY is None or THIS_FILE_ONLY >= 0, "The file number must be integer >= 0!"
        assert not (THIS_FILE_ONLY is not None and MERGE_ONLY), "Attention, you cannot choose a specific file and merge only together!"

        # ---------------------------------------------
        # Constants and Paths
        # ---------------------------------------------

        RUN = ...  # put an string for the number of the experiments run, e.g. '34'
        MODULE = ...  # put a name for the detector, e.g. 'DetA'
        PATH_HW_DATA = ...  # path to the directory in which the RDT and CON files are stored
        PATH_STREAM_DATA = ...  # path to the directory in which the CSMPL file directories are stored
        PATH_DB = ...  # path to the SQL data base with information about the CSMPL files
        PATH_PROC_DATA = ...  # path to where you want to store the HDF5 files
        FILE_NMBRS = []  # a list of string, the file number you want to analyse, e.g. ['001', '002', '003']
        FNAMING = ...  # the naming of the files, typically 'bck', for calibration data 'cal'
        RDT_CHANNELS = []  # a list of strings of the channels, e.g. [0, 1] (written in PAR file - attention, the PAR file counts from 1, Cait from 0)
        CSMPL_CHANNELS = [0, 1]  # the channel numbers of the CDAQ, written in the SQL data base
        RECORD_LENGTH = 16384  # the number of samples within a record window  (read in PAR file)
        SAMPLE_FREQUENCY = 25000  # the sample frequency of the measurement (read in PAR file)
        DOWN_SEF = 4  # the downsample rate for the standard event fit
        DOWN_BLF = 16  # the downsample rate for the baseline fit
        PROCESSES = 8  # the number of processes for parallelization
        PCA_COMPONENTS = 2  # the number of pca components to calculate
        SKIP_FILE_NMBRS = []  # in case the loop crashed at some point and you want to start from a specific file number, write here the numbers to ignore, e.g. ['001', '002']
        THRESHOLDS = []  # a list of the trigger thresholds in V
        TRUNCATION_LEVELS = []  # list of the truncation levels


        # typically you need not change the values below this line!

        FNAME_HW = 'hw_{:03d}'.format(len(FILE_NMBRS) - 1)
        FNAME_STREAM = 'stream_{:03d}'.format(len(FILE_NMBRS) - 1)
        H5_CHANNELS = list(range(len(RDT_CHANNELS)))
        SEF_APP = '_down{}'.format(DOWN_SEF) if DOWN_SEF > 1 else ''
        CORRELATED_FIT = True

        assert len(FILE_NMBRS) > 0, "Choose some file numbers!"
        assert THIS_FILE_ONLY not in SKIP_FILE_NMBRS, "Attention, you chose a file that is in the skip list!"

        if THIS_FILE_ONLY is not None:
            SKIP_FILE_NMBRS = FILE_NMBRS.copy()
            del SKIP_FILE_NMBRS[THIS_FILE_ONLY]

        print('OF thresholds in V: ', THRESHOLDS)

        # ---------------------------------------------
        # Get Filter from HW Data
        # ---------------------------------------------

        dh_hw = ai.DataHandler(run=RUN,
                            module=MODULE,
                            channels=RDT_CHANNELS)

        dh_hw.set_filepath(path_h5=PATH_PROC_DATA,
                        fname='hw_{:03d}'.format(len(FILE_NMBRS)-1),
                        appendix=False)

        of_r = dh_hw.get(group='optimumfilter', dataset='optimumfilter_real')
        of_i = dh_hw.get(group='optimumfilter', dataset='optimumfilter_imag')
        of = of_r + 1j*of_i

        # ---------------------------------------------
        # Start the Trigger Loop
        # ---------------------------------------------

        for i, fn in enumerate(FILE_NMBRS):

            print('-----------------------------------------------------')
            print('>> {} WORKING ON FILE: {}'.format(i, fn))

            if fn in SKIP_FILE_NMBRS:
                print('Skipping this file.')

            else:
                if not MERGE_ONLY:
                    dh = ai.DataHandler(run=RUN,
                                        channels=RDT_CHANNELS,
                                        record_length=RECORD_LENGTH,
                                        sample_frequency=SAMPLE_FREQUENCY)

                    dh.set_filepath(path_h5=PATH_PROC_DATA,
                                    fname='stream_' + FNAMING + '_' + fn,
                                    appendix=False)

                    csmpl_paths = [PATH_STREAM_DATA + 'Ch' + str(c+1) + '/' + 'Run' + RUN + '_' + FNAMING + '_' + fn + '_Ch' + str(c+1) + '.csmpl' for c in CSMPL_CHANNELS]

                    # --------------------------------------------------
                    # Trigger Files
                    # --------------------------------------------------

                    dh.include_csmpl_triggers(csmpl_paths=csmpl_paths,
                                              thresholds=THRESHOLDS,
                                              of=of,
                                              path_sql=PATH_DB,
                                              csmpl_channels=CSMPL_CHANNELS,  # the channel numbers in the csmpl file are different from rdt
                                              sql_file_label=FNAMING + '_{}'.format(fn),
                                              down=1,  # downsampling not properly implemented yet
                                              )

                    # --------------------------------------------------
                    # Include Test Pulse Time Stamps
                    # --------------------------------------------------

                    dh.include_test_stamps(path_teststamps=PATH_HW_DATA + FNAMING + '_' + fn + '.test_stamps',
                                           path_dig_stamps=PATH_HW_DATA + FNAMING + '_' + fn + '.dig_stamps',
                                           path_sql=PATH_DB,
                                           csmpl_channels=CSMPL_CHANNELS,
                                           sql_file_label=FNAMING + '_' + fn,
                                           fix_offset=True)

                    # --------------------------------------------------
                    # Include Triggered Events
                    # --------------------------------------------------

                    dh.include_triggered_events(csmpl_paths=csmpl_paths,
                                                max_time_diff=0.5, # in sec - this prevents all pile up with test pulses
                                                exclude_tp=True,
                                                sample_duration=1/SAMPLE_FREQUENCY,
                                                datatype='float32',
                                                min_tpa=0.001,
                                                min_cpa=10.1,
                                                down=1)

                    # ----------------------------------------------------------
                    # Include OF, SEV, NPS to first set (we keep them at merge)
                    # ----------------------------------------------------------

                    dh.include_sev(sev=dh_hw.get('stdevent','event'),
                                   fitpar=dh_hw.get('stdevent','fitpar'),
                                   mainpar=dh_hw.get('stdevent','mainpar'))

                    dh.include_nps(nps=dh_hw.get('noise','nps'))

                    dh.include_of(of_real=dh_hw.get('optimumfilter','optimumfilter_real'),
                                  of_imag=dh_hw.get('optimumfilter','optimumfilter_imag'))

                    dh.include_sev(sev=dh_hw.get('stdevent_tp','event'),
                                   fitpar=dh_hw.get('stdevent_tp','fitpar'),
                                   mainpar=dh_hw.get('stdevent_tp','mainpar'),
                                   group_name_appendix='_tp')

                    dh.include_of(of_real=dh_hw.get('optimumfilter_tp','optimumfilter_real'),
                                  of_imag=dh_hw.get('optimumfilter_tp','optimumfilter_imag'),
                                  group_name_appendix='_tp')

                    # --------------------------------------------------
                    # Calc Mainpar for Events and Testpulses
                    # --------------------------------------------------

                    dh.calc_mp(type='events')
                    dh.calc_mp(type='testpulses')
                    dh.calc_additional_mp()

                    # --------------------------------------------------
                    # Apply OF for Events and Testpulses
                    # --------------------------------------------------

                    dh.apply_of(first_channel_dominant=CORRELATED_FIT)
                    dh.apply_of(type='testpulses', name_appendix_group='_tp')

                    # --------------------------------------------------
                    # Do SEV Fit for Events and Testpulses
                    # --------------------------------------------------

                    dh.apply_sev_fit(down=DOWN_SEF, name_appendix='_down{}'.format(DOWN_SEF), processes=PROCESSES,
                                     truncation_level=TRUNCATION_LEVELS, verb=True, first_channel_dominant=CORRELATED_FIT)
                    dh.apply_sev_fit(type='testpulses', group_name_appendix='_tp',
                                     down=DOWN_SEF, name_appendix='_down{}'.format(DOWN_SEF), processes=PROCESSES,
                                     truncation_level=TRUNCATION_LEVELS, verb=True)

                    # --------------------------------------------------
                    # Apply Cuts
                    # --------------------------------------------------

                    # change this to your individual cut values!

                    clean_events = ai.cuts.LogicalCut(initial_condition=np.abs(dh.get('events', 'mainpar')[0,:,8]) < 2e-6)
                    clean_events.add_condition(np.abs(dh.get('events', 'mainpar')[1,:,8]) < 2e-6)
                    clean_events.add_condition(dh.get('events', 'mainpar')[0,:,0] < 1)
                    clean_events.add_condition(dh.get('events', 'mainpar')[1,:,0] < 1.5)
                    clean_events.add_condition(dh.get('events', 'mainpar')[0,:,3] < 4500)
                    clean_events.add_condition(dh.get('events', 'mainpar')[0,:,3] > 3900)
                    clean_events.add_condition(dh.get('events', 'mainpar')[1,:,3] < 4500)
                    clean_events.add_condition(dh.get('events', 'mainpar')[1,:,3] > 3900)

                    # typically you need not change anything below here

                    for c in H5_CHANNELS:
                        dh.apply_logical_cut(cut_flag=clean_events.get_flag(),
                                             naming='clean_events',
                                             channel=c,
                                             type='events',
                                             delete_old=False)

                    # --------------------------------------------------
                    # PCA
                    # --------------------------------------------------

                    dh.apply_pca(nmbr_components=PCA_COMPONENTS,
                                 down=DOWN_SEF,
                                 fit_idx=clean_events.get_idx())

                else:
                    print('Doing only the merge.')

                # --------------------------------------------------
                # Merge the files
                # --------------------------------------------------

                if i > 0 and THIS_FILE_ONLY is None:

                    file_name_a = PATH_PROC_DATA + 'stream_' + FNAMING + '_{}.h5'.format(FILE_NMBRS[0]) if i == 1 else PATH_PROC_DATA + 'stream_{:03d}.h5'.format(i-1)
                    a_name = 'stream_' + FNAMING + '_{}'.format(FILE_NMBRS[0]) if i == 1 else 'keep'

                    ai.data.merge_h5_sets(path_h5_a=file_name_a,
                                          path_h5_b=PATH_PROC_DATA + 'stream_' + FNAMING + '_{}.h5'.format(fn),
                                          path_h5_merged=PATH_PROC_DATA + 'stream_{:03d}.h5'.format(i),
                                          groups_to_merge=['events', 'testpulses', 'controlpulses', 'stream'],
                                          sets_to_merge=['event', 'mainpar', 'true_ph', 'true_onset', 'of_ph',
                                                         'sev_fit_par' + SEF_APP, 'sev_fit_rms' + SEF_APP,
                                                         'hours', 'labels', 'testpulseamplitude', 'time_s',
                                                         'time_mus', 'pulse_height', 'pca_error', 'pca_projection', 'tp_hours',
                                                         'tp_time_mus', 'tp_time_s', 'tpa',
                                                         'trigger_hours', 'trigger_time_mus', 'trigger_time_s'],
                                          concatenate_axis=[1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                          continue_hours=True,
                                          keep_original_files=False,
                                          groups_from_a=['optimumfilter', 'optimumfilter_tp', 'stdevent', 'stdevent_tp', 'noise'],
                                          a_name=a_name,
                                          b_name='stream_' + FNAMING + '_{}'.format(fn),
                                          verb=False,
                                          )

        # ---------------------------------------------
        # Finishing Notes
        # ---------------------------------------------

        print('-----------------------------------------------------')
        print('>> DONE WITH ALL FILES.')
