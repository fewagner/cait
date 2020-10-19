import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cait as ai

if __name__ == '__main__':
    # create Instance of the DataHandler Class
    dh = ai.DataHandler(run=35,
                        module='DetF',
                        channels=[26, 27],
                        record_length=16384)

    # set the file path for the hdf5 file
    dh.set_filepath(path_h5='toy_data',
                    fname='bck_001')

    # add labels
    dh.import_labels(path_labels='toy_data')
