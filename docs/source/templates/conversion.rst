*******************
Conversion to HDF5
*******************

.. code:: python

    """
    conversion.py

    This script is for the efficient conversion and merge of several RDT and CON files, into one HDF5 file.

    Usage:
    - Adapt the section 'Constants and Paths' to your measurement.
    - If you start the script without command line arguments, it will convert all files and merge them one after another.
    - If you start the script with the flag -f n, if will only convert the n'th file from the list of files.
    - If you start the script with the flag -m, it will only do the merge between all files.
    - For time efficient conversion, a good workflow is to write a bash script, that starts the conversion of all files
        simultaneously with the -f flags, then call the script again with the -m flag when all conversions are done.
    """

    import cait as ai
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

        # adapt these values to your measurement!

        RUN = ...  # put an string for the number of the experiments run, e.g. '34'
        MODULE = ...  # put a name for the detector, e.g. 'DetA'
        PATH_HW_DATA = ...  # path to the directory in which the RDT and CON files are stored
        PATH_PROC_DATA = ...  # path to where you want to store the HDF5 files
        FILE_NMBRS = []  # a list of string, the file number you want to analyse, e.g. ['001', '002', '003']
        FNAMING = ...  # the naming of the files, typically 'bck', for calibration data 'cal'
        RDT_CHANNELS = []  # a list of strings of the channels, e.g. [0, 1] (written in PAR file - attention, the PAR file counts from 1, Cait from 0)
        RECORD_LENGTH = 16384  # the number of samples within a record window  (read in PAR file)
        SAMPLE_FREQUENCY = 25000  # the sample frequency of the measurement (read in PAR file)
        SKIP_FILE_NMBRS = []  # in case the loop crashed at some point and you want to start from a specific file number, write here the numbers to ignore, e.g. ['001', '002']

        # do not change anything below here!

        FNAME_HW = 'hw_{:03d}'.format(len(FILE_NMBRS) - 1)
        if len(RDT_CHANNELS) == 2:
            FILE_APP = '-P_Ch{}-L_Ch{}'.format(*RDT_CHANNELS)
        else:
            FILE_APP = ''
            for i, c in enumerate(RDT_CHANNELS):
                FILE_APP += '-{}_Ch{}'.format(i+1,c)
        H5_CHANNELS = list(range(len(RDT_CHANNELS)))

        assert THIS_FILE_ONLY not in SKIP_FILE_NMBRS, "Attention, you chose a file that is in the skip list!"
        assert len(FILE_NMBRS) > 0, "Choose some file numbers!"

        if THIS_FILE_ONLY is not None:
            SKIP_FILE_NMBRS = FILE_NMBRS.copy()
            del SKIP_FILE_NMBRS[THIS_FILE_ONLY]

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
                    # --------------------------------------------------
                    # Convert Rdt to H5
                    # --------------------------------------------------

                    dh = ai.DataHandler(run=RUN,
                                        module=MODULE,
                                        channels=RDT_CHANNELS,
                                        record_length=RECORD_LENGTH,
                                        sample_frequency=SAMPLE_FREQUENCY)

                    dh.convert_dataset(path_rdt=PATH_HW_DATA,
                                       fname='{}_{}'.format(FNAMING, fn),
                                       path_h5=PATH_PROC_DATA,
                                       tpa_list=[0, 1, -1],
                                       calc_mp=False,
                                       calc_nps=False,
                                       memsafe=True,  # this option is currently under testing for bugs
                                       trace=True,  # plot memory usage and runtime
                                       lazy_loading=True,
                                       batch_size=1000,  # usually this does not affect memory usuage or runtime a lot - 1000 should be fine
                                       )

                    # --------------------------------------------------
                    # Include control pulses
                    # --------------------------------------------------

                    dh.include_con_file(path_con_file=PATH_HW_DATA + '{}_{}.con'.format(FNAMING, fn))

                    del dh

                # --------------------------------------------------
                # Merge the files
                # --------------------------------------------------

                if i > 0 and THIS_FILE_ONLY is None:

                    file_name_a = PATH_PROC_DATA + '{}_{}{}.h5'.format(FNAMING, FILE_NMBRS[0], FILE_APP) if i == 1 else PATH_PROC_DATA + 'hw_{:03d}.h5'.format(i-1)
                    a_name = '{}_{}'.format(FNAMING, FILE_NMBRS[0]) if i == 1 else 'keep'

                    ai.data.merge_h5_sets(path_h5_a=file_name_a,
                                      path_h5_b=PATH_PROC_DATA + '{}_{}{}.h5'.format(FNAMING, fn, FILE_APP),
                                      path_h5_merged=PATH_PROC_DATA + 'hw_{:03d}.h5'.format(i),
                                      continue_hours=True,
                                      keep_original_files=False,
                                      a_name=a_name,
                                      b_name='{}_{}'.format(FNAMING, fn),
                                      verb=False,
                                      trace=True,
                                     )
