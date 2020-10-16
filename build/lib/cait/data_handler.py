"""
"""

# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

from .data import gen_dataset_from_rdt


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class DataHandler:

    def __init__(self, run, module, record_length, nmbr_channels):
        # ask user things like which detector working on etc
        if nmbr_channels != 2:
            raise NotImplementedError('Only for 2 channels implemented.')
        self.run = run
        self.module = module
        self.record_length = record_length
        self.nmbr_channels = nmbr_channels

        print('DataHandler Instance created.')

    # Converts a bck to a hdf5 for one module with 2 or 3 channels
    def convert_dataset(self, path_rdt,
                        fname, path_h5,
                        channels, tpa_list=[0.],
                        calc_mp=True, calc_fit=False,
                        calc_sev=False, processes=4):

        if not len(channels) == self.nmbr_channels:
            raise ValueError('Channels must be a list with length nmbr_channels!')

        print('Start converting.')

        gen_dataset_from_rdt(path_rdt=path_rdt,
                             fname=fname,
                             path_h5=path_h5,
                             phonon_channel=channels[0],
                             light_channel=channels[1],
                             tpa_list=tpa_list,
                             calc_mp=calc_mp,
                             calc_fit=calc_fit,
                             calc_sev=calc_sev,
                             processes=processes
                             )

        print('Hdf5 dataset created in  {}'.format(path_h5))

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

    # Recalculate MP
    def recalc_mp(self, path):
        raise NotImplementedError('Not implemented.')

    # Recalculate Fit
    def recalc_fit(self, path):
        raise NotImplementedError('Not implemented.')

    # Import label CSV file in hdf5 file
    def import_labels(self, path_hdf5, path_labels):
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
