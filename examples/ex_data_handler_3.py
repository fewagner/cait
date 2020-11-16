import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import h5py
import cait as ai

if __name__ == '__main__':
    # create Instance of the DataHandler Class
    dh = ai.DataHandler(run=33,
                        module='TUM38',
                        channels=[36, 37],
                        record_length=8192)

    # set the file path for the hdf5 file
    dh.set_filepath(path_h5='toy_data',
                    fname='bck_013')

    dh.apply_sev_fit(type='events',
                     order_bl_polynomial=3)
    dh.apply_of()