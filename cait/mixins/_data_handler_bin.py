# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import h5py
from ..features._mp import calc_main_parameters
from ..trigger._csmpl import trigger_csmpl, get_record_window, align_triggers, sample_to_time, \
    exclude_testpulses, get_starttime, get_test_stamps, get_offset
from tqdm.auto import tqdm
from ..fit._pm_fit import fit_pulse_shape
from ..fit._templates import pulse_template
import os


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class BinMixin(object):
    """
    A Mixin Class to the DataHandler Class with methods for the triggering of *.bin files.
    """

    def include_vtrigger_stamps(self,
                                ):
        # TODO
        pass

    def include_triggered_events_vdaq(self, ):
        # TODO
        pass

    def include_test_stamps_vdaq(self):
        # TODO
        pass

    def include_noise_triggers_vdaq(self):
        # TODO
        pass

    def include_noise_events_vdaq(self):
        # TODO
        pass
