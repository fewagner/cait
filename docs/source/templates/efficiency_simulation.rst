*************************
Efficiency Simulation
*************************

.. code:: python

    """
    efficiency_simulation.py

    This script is for the simulation of a data set to determine a cut efficiency.

    Usage:
    - Adapt the section 'Constants and Paths' to your measurement.
    - Adapt the section 'Apply Cuts' (around line 240) to your individual cut values.
    - If you start the script without command line arguments, it will simulate all files and merge them one after another.
    - If you start the script with the flag -f n, if will only simulate the n'th file from the list of files.
    - If you start the script with the flag -m, it will only do the merge between all files.
    - For time efficient simulation, a good workflow is to write a bash script, that starts the simulation of all files
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
    import pickle
    import os

    if __name__ == '__main__':

        # ---------------------------------------------
        # Read command line arguments
        # ---------------------------------------------

        # Construct the argument parser
        ap = argparse.ArgumentParser()

        # Add the arguments to the parser
        ap.add_argument("-f", "--file", type=int, required=False, default=None,
                        help="Use only this index from the files list.")
        ap.add_argument("-m", "--merge", action='store_true', help="Only merge the files list.")
        args = vars(ap.parse_args())

        THIS_FILE_ONLY = args['file']
        MERGE_ONLY = args['merge']

        assert THIS_FILE_ONLY is None or THIS_FILE_ONLY >= 0, "The file number must be integer >= 0!"
        assert not (
                THIS_FILE_ONLY is not None and MERGE_ONLY), "Attention, you cannot choose a specific file and merge only together!"

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
        MAXIMAL_EVENT_HEIGHTS = []  # list of the maximal event heights to be included in the simulation
        MINIMAL_EVENT_HEIGHTS = []  # list of the minimal event heights to be included in the simulation
        PATH_PULSER_MODEL = ... # put an string of the path where you want to store the pulser model
        POLY_ORDER = 5  # the order of the polynomial used for the energy calibration
        ONLY_ECAL = False  # if this is set to true, only the energy calibration is done for the file
        CPE_FACTOR = []  # list of the values source_peak_energy/source_peak_equivalent_tpa_value
        NMBR_SIMULATED_EVENTS = 20000  # this number of events is simulated for each file in the list
        XSCALE = 'log'
        BINS = 2000

        # typically you do not need to change the values below

        FNAME_STREAM = 'stream_{:03d}'.format(len(FILE_NMBRS) - 1)
        H5_CHANNELS = list(range(len(RDT_CHANNELS))
        print('OF thresholds in V: ', THRESHOLDS)
        SEF_APP = '_down{}'.format(DOWN_SEF) if DOWN_SEF > 1 else ''
        
        if XSCALE == 'linear':
        discrete_ph = np.array([np.linspace(mi, ma, BINS + 1) for mi,ma in zip(MINIMAL_EVENT_HEIGHTS, MAXIMAL_EVENT_HEIGHTS)])
    elif XSCALE == 'log':
        if any(np.array(MINIMAL_EVENT_HEIGHTS) <= 0):
            print('Changing lower end of non-positive MINIMAL_EVENT_HEIGHTS to 1e-3!')
            MINIMAL_EVENT_HEIGHTS[MINIMAL_EVENT_HEIGHTS <= 0] = 1e-3
        discrete_ph = np.array([np.logspace(start=np.log10(mi), stop=np.log10(ma), num=BINS + 1) for mi,ma in zip(MINIMAL_EVENT_HEIGHTS, MAXIMAL_EVENT_HEIGHTS)])
    else:
        raise ValueError('The argument of XSCALE must be either linear or log!')
    discrete_ph = discrete_ph[:, :-1] + (discrete_ph[:, 1:] - discrete_ph[:, :-1]) / 2

        assert len(FILE_NMBRS) > 0, "Choose some file numbers!"
        assert THIS_FILE_ONLY not in SKIP_FILE_NMBRS, "Attention, you chose a file that is in the skip list!"

        if THIS_FILE_ONLY is not None:
            SKIP_FILE_NMBRS = FILE_NMBRS.copy()
            del SKIP_FILE_NMBRS[THIS_FILE_ONLY]

        # ---------------------------------------------
        # Get Handle to Stream Data
        # ---------------------------------------------

        dh_stream = ai.DataHandler(run=RUN,
                                   module=MODULE,
                                   channels=RDT_CHANNELS)

        dh_stream.set_filepath(path_h5=PATH_PROC_DATA,
                               fname='stream_{:03d}'.format(len(FILE_NMBRS) - 1),
                               appendix=False)

        start_hours = dh_stream.get('metainfo', 'startstop_hours')[:, 0]

        # ---------------------------------------------
        # Start the Loop
        # ---------------------------------------------

        for i, fn in enumerate(FILE_NMBRS):

            print('-----------------------------------------------------')
            print('>> {} WORKING ON FILE: {}'.format(i, fn))

            if fn in SKIP_FILE_NMBRS:
                print('Skipping this file.')

            else:
                if not MERGE_ONLY:
                    empty_name = 'empty_' + FNAMING + '_' + fn
                    sim_name = 'sim_' + FNAMING + '_' + fn
                    
                    if os.path.isfile(PATH_PROC_DATA + empty_name + '.h5'):
                        os.remove(PATH_PROC_DATA + empty_name + '.h5')

                    dh_empty = ai.DataHandler(run=RUN,
                                              channels=RDT_CHANNELS,
                                              record_length=RECORD_LENGTH,
                                              sample_frequency=SAMPLE_FREQUENCY)

                    dh_empty.set_filepath(path_h5=PATH_PROC_DATA,
                                          fname=empty_name,
                                          appendix=False)

                    csmpl_paths = [
                        PATH_STREAM_DATA + 'Ch' + str(c + 1) + '/' + 'Run' + RUN + '_' + FNAMING + '_' + fn + '_Ch' + str(
                            c + 1) + '.csmpl' for c in CSMPL_CHANNELS]

                    if not ONLY_ECAL:

                        # --------------------------------------------------
                        # Include Test Pulse Time Stamps
                        # --------------------------------------------------

                        dh_empty.include_test_stamps(path_teststamps=PATH_HW_DATA + FNAMING + '_' + fn + '.test_stamps',
                                                     path_dig_stamps=PATH_HW_DATA + FNAMING + '_' + fn + '.dig_stamps',
                                                     path_sql=PATH_DB,
                                                     csmpl_channels=CSMPL_CHANNELS,
                                                     sql_file_label=FNAMING + '_' + fn,
                                                     fix_offset=True)

                        # --------------------------------------------------
                        # Include the Random Triggers Events
                        # --------------------------------------------------

                        dh_empty.include_noise_triggers(
                            nmbr=NMBR_SIMULATED_EVENTS,
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

                        dh_empty.include_sev(sev=dh_stream.get('stdevent', 'event'),
                                             fitpar=dh_stream.get('stdevent', 'fitpar'),
                                             mainpar=dh_stream.get('stdevent', 'mainpar'))

                        dh_empty.include_nps(nps=dh_stream.get('noise', 'nps'))

                        dh_empty.include_of(of_real=dh_stream.get('optimumfilter', 'optimumfilter_real'),
                                            of_imag=dh_stream.get('optimumfilter', 'optimumfilter_imag'))

                    # --------------------------------------------------
                    # Simulate Events
                    # --------------------------------------------------

                        dh_empty.calc_bl_coefficients(down=DOWN_BLF)

                        dh_empty.simulate_pulses(path_sim=PATH_PROC_DATA + sim_name + '.h5',
                                              size_events=NMBR_SIMULATED_EVENTS,
                                              reuse_bl=True,
                                              # ev_ph_intervals=[(0, m) for m in MAXIMAL_EVENT_HEIGHTS],
                                              ev_discrete_phs=discrete_ph,
                                              t0_interval=[-10, 10],  # in ms
                                              rms_thresholds=[1e5, 1e5],
                                              fake_noise=False)

                    # --------------------------------------------------
                    # Delete original set
                    # --------------------------------------------------

                        # Delete the empty bl hdf5 set
                        del dh_empty
                        print('Deleting {}.'.format(PATH_PROC_DATA + empty_name + '.h5'))
                        os.remove(PATH_PROC_DATA + empty_name + '.h5')

                    dh_sim = ai.DataHandler(run=RUN,
                                            channels=RDT_CHANNELS,
                                            record_length=RECORD_LENGTH,
                                            sample_frequency=SAMPLE_FREQUENCY)

                    dh_sim.set_filepath(path_h5=PATH_PROC_DATA,
                                        fname=sim_name,
                                        appendix=False)



                    # --------------------------------------------------
                    # Calc Parameters
                    # --------------------------------------------------

                    if not ONLY_ECAL:

                        dh_sim.calc_mp(type='events')
                        dh_sim.calc_additional_mp()

                        dh_sim.apply_of()

                        dh_sim.apply_sev_fit(down=DOWN_SEF, name_appendix='_down{}'.format(DOWN_SEF), processes=PROCESSES,
                                             truncation_level=TRUNCATION_LEVELS, verb=True)

                    # --------------------------------------------------
                    # Apply Cuts
                    # --------------------------------------------------

                    if not ONLY_ECAL:

                        # change this to your individual cut values!

                        surviving = ai.cuts.LogicalCut(
                            initial_condition=np.abs(dh_sim.get('events', 'mainpar')[0, :, 8]) < 1e-6)
                        surviving.add_condition(np.abs(dh_sim.get('events', 'mainpar')[1, :, 8]) < 1e-6)
                        surviving.add_condition(dh_sim.get('events', 'mainpar')[0, :, 0] < 1.6)
                        surviving.add_condition(dh_sim.get('events', 'mainpar')[1, :, 0] < 0.3)
                        surviving.add_condition(dh_sim.get('events', 'mainpar')[0, :, 3] < 4500)
                        surviving.add_condition(dh_sim.get('events', 'mainpar')[0, :, 3] > 3900)
                        surviving.add_condition(dh_sim.get('events', 'of_ph')[0,:] > THRESHOLDS[0])  # threshold

                        # typically you need not change anything below here

                        for c in H5_CHANNELS:
                            dh_sim.apply_logical_cut(cut_flag=surviving.get_flag(),
                                                     naming='surviving',
                                                     channel=c,
                                                     type='events',
                                                     delete_old=False)

                        # --------------------------------------------------
                        # PCA
                        # --------------------------------------------------

                        dh_sim.apply_pca(nmbr_components=PCA_COMPONENTS,
                                         down=DOWN_SEF,
                                         fit_idx=surviving.get_idx())

                    # --------------------------------------------------
                    # Assign Energies
                    # --------------------------------------------------

                    with open(PATH_PULSER_MODEL, 'rb') as f:
                        pm = pickle.load(f)

                    dh_sim.calc_calibration(starts_saturation=MAXIMAL_EVENT_HEIGHTS,
                                            cpe_factor=CPE_FACTOR,
                                            poly_order=POLY_ORDER,
                                            plot=False,
                                            method='of',
                                            pulser_models=pm,
                                            name_appendix_energy='_reconstructed',
                                            )

                    dh_sim.calc_calibration(starts_saturation=MAXIMAL_EVENT_HEIGHTS,
                                            cpe_factor=CPE_FACTOR,
                                            poly_order=POLY_ORDER,
                                            plot=False,
                                            method='true_ph',
                                            pulser_models=pm,
                                            name_appendix_energy='_true',
                                            )

                else:
                    print('Doing only the merge.')

                # --------------------------------------------------
                # Merge the files
                # --------------------------------------------------

                if i > 0 and THIS_FILE_ONLY is None:
                    file_name_a = PATH_PROC_DATA + 'sim_' + FNAMING + '_{}.h5'.format(
                        FILE_NMBRS[0]) if i == 1 else PATH_PROC_DATA + 'efficiency_{:03d}.h5'.format(i - 1)
                    a_name = 'sim_' + FNAMING + '_{}'.format(FILE_NMBRS[0]) if i == 1 else 'keep'

                    ai.data.merge_h5_sets(path_h5_a=file_name_a,
                                          path_h5_b=PATH_PROC_DATA + 'sim_' + FNAMING + '_{}.h5'.format(fn),
                                          path_h5_merged=PATH_PROC_DATA + 'efficiency_{:03d}.h5'.format(i),
                                          groups_to_merge=['events', 'testpulses', 'controlpulses', 'stream'],
                                          sets_to_merge=['event', 'mainpar', 'true_ph', 'true_onset', 'of_ph',
                                                         'sev_fit_par' + SEF_APP, 'sev_fit_rms' + SEF_APP,
                                                         'hours', 'labels', 'testpulseamplitude', 'time_s',
                                                         'time_mus', 'pulse_height', 'pca_error', 'pca_projection',
                                                         'tp_hours',
                                                         'tp_time_mus', 'tp_time_s', 'tpa',
                                                         'trigger_hours', 'trigger_time_mus', 'trigger_time_s',
                                                         'surviving', 'recoil_energy_true', 'recoil_energy_sigma_true',
                                                         'tpa_equivalent_true', 'tpa_equivalent_sigma_true',
                                                         'recoil_energy_reconstructed', 'recoil_energy_sigma_reconstructed',
                                                         'tpa_equivalent_reconstructed', 'tpa_equivalent_sigma_reconstructed'],
                                          concatenate_axis=[1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                                            0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                          continue_hours=True,
                                          second_file_start=start_hours[i],
                                          keep_original_files=False,
                                          groups_from_a=['optimumfilter', 'optimumfilter_tp', 'stdevent', 'stdevent_tp',
                                                         'noise'],
                                          a_name=a_name,
                                          b_name='sim_' + FNAMING + '_{}'.format(fn),
                                          verb=False,
                                          )

        # ---------------------------------------------
        # Finishing Notes
        # ---------------------------------------------

        print('-----------------------------------------------------')
        print('>> DONE WITH ALL FILES.')
