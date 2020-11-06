"""
This is an example how to add labels to a HDF5 file that already exists.
See also the How-Tos on the Wiki page!
https://git.cryocluster.org/fwagner/cait/-/wikis/4.1-How-To:-Convert-an-*.rdt-file-to-an-hdf5-dataset
"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import h5py
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

    # recalculate SEV and OF and plot them all
    dh.recalc_sev(use_labels=True,
                  pulse_height_intervall=[0.1, 2],
                  scale_fit_height=True)
    dh.recalc_of()

    # plot SEV
    dh.show_SEV()

    # plot NPS
    dh.show_NPS()

    # plot OF
    dh.show_OF()
