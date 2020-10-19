"""
"""

# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import os
import numpy as np
import h5py
from multiprocessing import Pool
from .data._gen_h5 import gen_dataset_from_rdt
from .features._mp import calc_main_parameters

# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class DataHandler:

    def __init__(self, run, module, channels, record_length):
        # ask user things like which detector working on etc
        if len(channels) != 2:
            raise NotImplementedError('Only for 2 channels implemented.')
        self.run = run
        self.module = module
        self.record_length = record_length
        self.nmbr_channels = len(channels)
        self.channels = channels

        print('DataHandler Instance created.')

    # Converts a bck to a hdf5 for one module with 2 or 3 channels
    def convert_dataset(self, path_rdt,
                        fname, path_h5,
                        tpa_list=[0],
                        calc_mp=True, calc_fit=False,
                        calc_sev=False, processes=4):

        print('Start converting.')

        gen_dataset_from_rdt(path_rdt=path_rdt,
                             fname=fname,
                             path_h5=path_h5,
                             phonon_channel=self.channels[0],
                             light_channel=self.channels[1],
                             tpa_list=tpa_list,
                             calc_mp=calc_mp,
                             calc_fit=calc_fit,
                             calc_sev=calc_sev,
                             processes=processes
                             )

        print('Hdf5 dataset created in  {}'.format(path_h5))

        if self.nmbr_channels == 2:
            self.path_h5 = "{}/run{}_{}/{}-P_Ch{}-L_Ch{}.h5".format(path_h5, self.run, self.module,
                                                                  fname, self.channels[0],
                                                                  self.channels[1])
            self.fname = fname

        else:
            raise NotImplementedError('Only for two channels implemented!')

        print('Filepath and -name saved.')

    def set_filepath(self, path_h5, fname):

        if self.nmbr_channels == 2:
            self.path_h5 = "{}/run{}_{}/{}-P_Ch{}-L_Ch{}.h5".format(path_h5, self.run, self.module,
                                                                  fname, self.channels[0],
                                                                  self.channels[1])
            self.fname = fname

        else:
            raise NotImplementedError('Only for two channels implemented!')

    # Recalculate MP
    def recalc_mp(self, type, path_hdf5=None, processes = 4):

        if not path_hdf5:
            path_hdf5 = self.path_h5

        h5f = h5py.File(path_hdf5, 'r+')
        events = h5f[type]

        print('CALCULATE MAIN PARAMETERS.')

        with Pool(processes) as p:  # basically a for loop running on 4 processes
            p_mainpar_list_event = p.map(calc_main_parameters, events['event'][0, :, :])
            l_mainpar_list_event = p.map(calc_main_parameters, events['event'][1, :, :])
        mainpar_event = np.array([[o.getArray() for o in p_mainpar_list_event],
                                  [o.getArray() for o in l_mainpar_list_event]])

        events['mainpar'][...] = mainpar_event


    # Recalculate Fit
    def recalc_fit(self, path):
        raise NotImplementedError('Not implemented.')


    # Import label CSV file in hdf5 file
    def import_labels(self, path_labels, path_hdf5=None):

        if not path_hdf5:
            path_hdf5 = self.path_h5

        path_labels = '{}/run{}_{}/labels_{}_events.csv'.format(path_labels, self.run, self.module, self.fname)

        h5f = h5py.File(path_hdf5, 'r+')

        if path_labels != '' and os.path.isfile(path_labels):
            labels_event = np.genfromtxt(path_labels)
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

        elif (path_labels != ''):
            print("File '{}' does not exist.".format(path_labels))

    # Set parameters for pulse simulations
    def prep_events(self, path_stdevent, path_baselines):
        raise NotImplementedError('Not implemented.')

    # Set parameters for testpulse simulations
    def prep_tp(self, path_stdevent, path_baselines):
        raise NotImplementedError('Not implemented.')

    # Set parameters for noise simulation
    def prep_noise(self, path_baselines):
        raise NotImplementedError('Not implemented.')

    # Set parameters for carrier simulation
    def prep_carrier(self, path_stdevent, path_baseline):
        raise NotImplementedError('Not implemented.')

    # Simulate Dataset with specific classes
    def simulate_fakenoise_dataset(self, classes_size):
        raise NotImplementedError('Not implemented.')

    # Simulate Dataset with real noise
    def simulate_realnoise_dataset(self, path_noise, classes_size):
        raise NotImplementedError('Not implemented.')

    # Calculate OF from NPS and Stdevent
    def calc_of(self, path):
        raise NotImplementedError('Not implemented.')

    # Create SEV from Labels
    def calc_SEV(self):
        raise NotImplementedError('Not implemented.')

    # Calculate NPS directly from noise events
    def calc_NPS(self):
        raise NotImplementedError('Not implemented.')

    # Create Optimum Filter Function
    def calc_OF(self):
        raise NotImplementedError('Not implemented.')
