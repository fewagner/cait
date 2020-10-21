# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import numpy as np
from ..fit import fit_pulse_shape


# ------------------------------------------------------------
# FUNCTION
# ------------------------------------------------------------

def generate_standard_event(events,
                            main_parameters,
                            labels=None,
                            pulse_height_intervall=None,
                            left_right_cutoff=None,
                            rise_time_intervall=None,
                            decay_time_intervall=None,
                            onset_intervall=None,
                            remove_offset=True,
                            verb=False):
    if verb:
        print('{} Events handed.'.format(len(main_parameters)))

    use_indices = np.ones(len(main_parameters))

    # use only those that are labeled as event

    if not labels is None:
        use_indices[labels != 1] = 0

        if verb:
            print('{} with Label Event Pulse.'.format(len(np.where(use_indices == 1)[0])))

    # pulse height cut
    if not pulse_height_intervall is None:
        use_indices[main_parameters[:, 0] < pulse_height_intervall[0]] = 0
        use_indices[main_parameters[:, 0] > pulse_height_intervall[1]] = 0

        if verb:
            print('{} left after PH cut.'.format(len(np.where(use_indices == 1)[0])))

    # left - right cut
    if not left_right_cutoff is None:
        use_indices[np.abs(main_parameters[:, 8]) > left_right_cutoff] = 0

        if verb:
            print('{} left after left - right cut.'.format(len(np.where(use_indices == 1)[0])))

    # rise time and decay cut
    if not rise_time_intervall is None:
        use_indices[main_parameters[:, 2] > main_parameters[:, 1] + rise_time_intervall[1]] = 0
        use_indices[main_parameters[:, 2] < main_parameters[:, 1] + rise_time_intervall[0]] = 0

        if verb:
            print('{} left after rise time cut.'.format(len(np.where(use_indices == 1)[0])))

    if not decay_time_intervall is None:
        use_indices[main_parameters[:, 5] > main_parameters[:, 3] + decay_time_intervall[1]] = 0
        use_indices[main_parameters[:, 5] < main_parameters[:, 3] + decay_time_intervall[0]] = 0

        if verb:
            print('{} left after decay time cut.'.format(len(np.where(use_indices == 1)[0])))

    # onset cut
    if not onset_intervall is None:
        use_indices[main_parameters[:, 3] > onset_intervall[1]] = 0
        use_indices[main_parameters[:, 3] < onset_intervall[0]] = 0

        if verb:
            print('{} left after onset cut.'.format(len(np.where(use_indices == 1)[0])))

    if verb:
        print('{} Events used to generate Standardevent.'.format(len(np.where(use_indices == 1)[0])))

    # remove offset
    if not remove_offset is None:
        events = events - np.mean(events[:, :1000], axis=1)[:, np.newaxis]

    # generate the standardevent
    standardevent = np.mean(events[use_indices == 1], axis=0)
    standardevent /= np.max(standardevent)

    par = fit_pulse_shape(standardevent)

    return standardevent, par
