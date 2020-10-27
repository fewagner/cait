"""
This file plots histograms of the testpulse mp
and calculates a SEV for the testpulses
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

    # plot the histogram of the main parameters
    dh.show_hist(which_mp='pulse_height',
                 which_channel=0,
                 type='testpulses')

    dh.show_hist(which_mp='pulse_height',
                 which_channel=1,
                 type='testpulses')

    # calc the TP SEV
    dh.calc_SEV_tp(pulse_height_intervall=[[0.8, 1.2], [0.7, 0.9]],
                   scale_fit_height=True)

    # show the TP SEV
    dh.show_SEV(type='stdevent_tp')

    # show the particle SEV
    dh.show_SEV(type='stdevent')