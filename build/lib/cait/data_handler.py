"""
"""

# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class DataHandler:

    def __init__(self):
        # ask user things like which detector working on etc
        pass

    # Converts a bck to a hdf5 for one module with 2 or 3 channels
    def convert_dataset(self, path_rtd, path_hdf5, calc_mp, calc_fit):
        pass

    # Set parameters for pulse simulations
    def prep_events(self, path_stdevent, path_baselines):
        pass

    # Set parameters for testpulse simulations
    def prep_events(self, path_stdevent, path_baselines):
        pass

    # Set parameters for noise simulation
    def prep_noise(self, path_baselines):
        pass

    # Set parameters for carrier simulation
    def prep_carrier(self, path_stdevent, path_baseline):
        pass

    # Simulate Dataset with specific classes
    def simulate_fakenoise_dataset(self, classes_size):
        pass

    # Simulate Dataset with real noise
    def simulate_realnoise_dataset(self, path_noise, classes_size):
        pass

    # Recalculate MP
    def recalc_mp(self, path):
        pass

    # Recalculate Fit
    def recalc_fit(self, path):
        pass

    # Import label CSV file in hdf5 file
    def import_labels(self, path_hdf5, path_labels):
        pass

    # Calculate OF from NPS and Stdevent
    def calc_of(self, path):
        pass

    # Create SEV from Labels
    def calc_SEV(self):
        pass

    # Calculate NPS directly from noise events
    def calc_NPS(self):
        pass

    # Create Optimum Filter Function
    def calc_OF(self):
        pass