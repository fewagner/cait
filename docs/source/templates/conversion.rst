*******************
Conversion to HDF5
*******************

.. code:: python

    """
    conversion.py

    Usage:

    First convert all files rdt -> HDF5

    python conversion.py -f <filename_1> … <filename_n> -c <channel_number> -p /eos/vbc/group/darkmatter/cosinus/data/cryo/fridge2/run577/run577/ -s /scratch-cbe/shared/darkmatter/cosinus/run577/ -t 100000 -q 100000 -r 32768 -d 1 -i 6


    python conversion.py -f <filename_1> -c <channel_number> -p <path_to_rdt_files> -s <path_to_store_hdf5_files> -t <clock_frequency> -q <sample_frequency> -r <record_length> -d <nmbr_dvm_channels> -i <integers_in_header>

    ...

    python conversion.py -f <filename_n> -c <channel_number> -p <path_to_rdt_files> -s <path_to_store_hdf5_files> -t <clock_frequency> -q <sample_frequency> -r <record_length> -d <nmbr_dvm_channels> -i <integers_in_header>

    (this can be done in parallel, or you can list the file names in the same script call)

    python conversion.py -f <filename_1> … <filename_n> -c <channel_number> -p <path_to_rdt_files> -s <path_to_store_hdf5_files> -t <clock_frequency> -q <sample_frequency> -r <record_length> -d <nmbr_dvm_channels> -i <integers_in_header>

    Then call the same script again with the -m flag to merge them together to one HDF5 file.

    python conversion.py -f <filename_1> … <filename_n> -c <channel_number> -p <path_to_rdt_files> -s <path_to_store_hdf5_files> -t <clock_frequency> -q <sample_frequency> -r <record_length> -d <nmbr_dvm_channels> -i <integers_in_header> -m
    """

    import cait as ai
    import argparse
    import os

    if __name__ == '__main__':

        # Construct the argument parser
        ap = argparse.ArgumentParser()

        # Add the arguments to the parser
        ap.add_argument("-f", "--file_list", type=str, nargs='+', help="List of the file names, e.g. bck_001 bck_002.")
        ap.add_argument("-c", "--channels", type=int, nargs='+', help="List of the channels of this module, e.g. 12 13.")
        ap.add_argument("-p", "--path_raw", type=str, help="Path to the RDT, PAR and CON files.")
        ap.add_argument("-s", "--path_h5", type=str, help="Path to the converted HDF5 files.")
        ap.add_argument("-t", "--clock_frequency", type=int, help="The frequency of the timer clock.")
        ap.add_argument("-q", "--sample_frequency", type=int, help="The sample frequency.")
        ap.add_argument("-r", "--record_length", type=int, help="The length of the record window.")
        ap.add_argument("-d", "--dvm_channels", type=int, help="The number of DVM channels.")
        ap.add_argument("-i", "--ints_in_header", type=int, help="The number of integers in the header.")
        ap.add_argument("-m", "--merge", action='store_true', help="Only merge the files list.")
        args = vars(ap.parse_args())

        for i, fname in enumerate(args['file_list']):

            if not args['merge']:

                # conversion rdt --> h5

                dh = ai.DataHandler(channels=args['channels'],
                                    record_length=args['record_length'],
                                    sample_frequency=args['sample_frequency'])

                dh.convert_dataset(path_rdt=args['path_raw'],
                                   fname=fname,
                                   path_h5=args['path_h5'],
                                   tpa_list=[0, 1, -1],
                                   ints_in_header=args['ints_in_header'],
                                   dvm_channels=args['dvm_channels'],
                                  )

                # include par and con file

                # dh.include_metainfo(args['path_raw'] + fname + '.par')
                # dh.include_con_file(path_con_file=args['path_raw'] + fname + '.con')

                # calc main parameters

                dh.calc_mp('events')
                dh.calc_mp('noise')
                dh.calc_mp('testpulses')

                dh.calc_additional_mp('events', no_of=True)
                dh.calc_additional_mp('noise', no_of=True)
                dh.calc_additional_mp('testpulses', no_of=True)

                if len(args['channels']) == 2:

                    dh.calc_ph_correlated()
                    dh.include_values(dh_stream.get('events', 'ph_corr')[1]/dh_stream.get('events', 'ph_corr')[0],
                                 naming='pseudo_ly', channel=0)

                dh.calc_peakdet(type='events')
                dh.calc_peakdet(type='testpulses')
                dh.calc_peakdet(type='noise')

                try:
                    ckp_path = ai.resources.get_resource_path('cnn-clf-binary-v1.ckpt')
                    model = ai.models.CNNModule.load_from_checkpoint(ckp_path)

                    if args['record_length'] == 8192:
                        model.down = 2
                    elif args['record_length'] == 4096:
                        model.down = 1

                    for type in ['events', 'testpulses', 'noise']:
                        for c in range(len(dh.channels)):
                            ai.models.nn_predict(h5_path=dh.path_h5,
                                       model=model,
                                       feature_channel=c,
                                       group_name=type,
                                       prediction_name='cnn_cut',
                                       keys=['event'],
                                       no_channel_idx_in_pred=False,
                                       use_prob=False)

                except:
                    pass

                # --------------------------------------------------
                # Merge the files
                # --------------------------------------------------

            if i > 0 and args['merge']:

                naming = ''
                for c in args['channels']:
                    naming += 'ch{}_'.format(c)

                if len(args['channels']) == 2:
                    file_app = '-P_Ch{}-L_Ch{}'.format(*args['channels'])
                else:
                    file_app = ''
                    for j, c in enumerate(args['channels']):
                        file_app += '-{}_Ch{}'.format(j+1,c)

                file_path_a = args['path_h5'] + args['file_list'][i-1] + file_app + '.h5' if i == 1 else args['path_h5'] + naming + '{:03d}.h5'.format(i-1)
                a_name = args['file_list'][i-1] if i == 1 else 'keep'

                file_path_b = args['path_h5'] + fname + file_app + '.h5'
                b_name = fname

                sets_merge = {
                    'event': 1,
                    'mainpar': 1,
                    'hours': 0,
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
                    'add_mainpar': 1,
                    'ph_corr': 1,
                    'cnn_cut': 1,
                    'pseudo_ly': 1,
                    'nmbr_peaks': 1,
                }

                ai.data.merge_h5_sets(path_h5_a=file_path_a,
                                  path_h5_b=file_path_b,
                                  path_h5_merged=args['path_h5'] + naming + '{:03d}.h5'.format(i),
                                  sets_to_merge=list(sets_merge.keys()),
                                  concatenate_axis=list(sets_merge.values()),
                                  continue_hours=False,
                                  keep_original_files=True,
                                  a_name=a_name,
                                  b_name=fname,
                                  verb=False,
                                 )

                if i > 1:
                    os.remove(file_path_a)
