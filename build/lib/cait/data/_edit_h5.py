# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from multiprocessing import Pool
import numpy as np
import h5py
from ..features._mp import calc_main_parameters


def edit_h5_dataset(path,
                    fname,
                    path_plots,
                    phonon_channel,
                    light_channel,
                    recalc_main_par=True,
                    processes=4):
    """
    THIS FUNCTION IS NO LONGER USED!
    Edit the HDF5 dataset, e.g. add labels and calculate main parameters.
    """
    path_to_eventlabels = '{}labels_{}_events.csv'.format(path, fname)
    h5f = h5py.File("{}{}-P_Ch{}-L_Ch{}.h5".format(path_plots, fname,
                                                   phonon_channel, light_channel), 'r+')

    # ################# ADD LABELS TO EVENTS #################

    if path_to_eventlabels != '' and os.path.isfile(path_to_eventlabels):
        labels_event = np.genfromtxt(path_to_eventlabels)
        labels_event = labels_event.astype('int32')
        length = len(labels_event)
        labels_event.resize((2, int(length / 2)))

        print(h5f.keys())

        events = h5f['events']

        if "labels" in events:
            events['labels'][...] = labels_event
            print('Edited Labels.')

        else:
            events.create_dataset('labels', data=labels_event)
            events['labels'].attrs.create(name='unlabeled', data=0)
            events['labels'].attrs.create(name='Event_Pulse', data=1)
            events['labels'].attrs.create(name='Test/Control_Pulse', data=2)
            events['labels'].attrs.create(name='Noise', data=3)
            events['labels'].attrs.create(name='Squid_Jump', data=4)
            events['labels'].attrs.create(name='Spike', data=5)
            events['labels'].attrs.create(name='Early_or_late_Trigger', data=6)
            events['labels'].attrs.create(name='Pile_Up', data=7)
            events['labels'].attrs.create(name='Carrier_Event', data=8)
            events['labels'].attrs.create(name='Strongly_Saturated_Event_Pulse', data=9)
            events['labels'].attrs.create(name='Strongly_Saturated_Test/Control_Pulse', data=10)
            events['labels'].attrs.create(name='Decaying_Baseline', data=11)
            events['labels'].attrs.create(name='Temperature Rise', data=12)
            events['labels'].attrs.create(name='Stick Event', data=13)
            events['labels'].attrs.create(name='Sawtooth Cycle', data=14)
            events['labels'].attrs.create(name='unknown/other', data=99)

            print('Added Labels.')

        # ################# ADD MAINPAR TO EVENTS #################

        if recalc_main_par:
            print('CALCULATE MAIN PARAMETERS.')

            with Pool(processes) as p:  # basically a for loop running on 4 processes
                p_mainpar_list_event = p.map(calc_main_parameters, events['event'][0, :, :])
                l_mainpar_list_event = p.map(calc_main_parameters, events['event'][1, :, :])
            mainpar_event = np.array([[o.getArray() for o in p_mainpar_list_event],
                                      [o.getArray() for o in l_mainpar_list_event]])

            events['mainpar'][...] = mainpar_event

    elif (path_to_eventlabels != ''):
        print("File '{}' does not exist.".format(path_to_eventlabels))


if __name__ == '__main__':
    edit_h5_dataset(path='data/run35_DetF/',
                    fname='bck_001',
                    path_plots='data/run35_DetF/',
                    phonon_channel=26,
                    light_channel=27,
                    recalc_main_par=True,
                    processes=4,
                    )
