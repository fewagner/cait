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
                                triggers: list,  # in seconds
                                name_appendix: str = '',
                                trigger_block: int = None,
                                file_start: float = None,  # in seconds
                                ):
        # TODO

        if trigger_block is None:
            trigger_block = self.record_length

        # fix the number of triggers
        if len(triggers) > 1:
            print('ALIGN TRIGGERS')
            aligned_triggers = align_triggers(triggers=triggers,  # in seconds
                                              trigger_block=trigger_block,
                                              sample_duration=1 / self.sample_frequency,
                                              )
        else:
            aligned_triggers = triggers[0]

        with h5py.File(self.path_h5, 'a') as h5f:

            stream = h5f.require_group(name='stream')
            # write them to file
            print('ADD DATASETS TO HDF5')
            if "trigger_hours{}".format(name_appendix) in stream:
                print('overwrite old trigger_hours{}'.format(name_appendix))
                del stream['trigger_hours{}'.format(name_appendix)]
            stream.create_dataset(name='trigger_hours' + name_appendix,
                                  data=aligned_triggers / 3600)

            if file_start is not None:
                # get file handle
                if "trigger_time_s{}".format(name_appendix) in stream:
                    print('overwrite old trigger_time_s{}'.format(name_appendix))
                    del stream['trigger_time_s{}'.format(name_appendix)]
                stream.create_dataset(name='trigger_time_s' + name_appendix,
                                      shape=(aligned_triggers.shape),
                                      dtype=int)
                if "trigger_time_mus{}".format(name_appendix) in stream:
                    print('overwrite old trigger_time_mus{}'.format(name_appendix))
                    del stream['trigger_time_mus{}'.format(name_appendix)]
                stream.create_dataset(name='trigger_time_mus{}'.format(name_appendix),
                                      shape=(aligned_triggers.shape),
                                      dtype=int)

                stream['trigger_time_s{}'.format(name_appendix)][...] = np.array(aligned_triggers + file_start,
                                                                                 dtype='int32')
                stream['trigger_time_mus{}'.format(name_appendix)][...] = np.array(1e6 * (
                        aligned_triggers + file_start - np.floor(aligned_triggers + file_start)), dtype='int32')

            print('DONE')

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
