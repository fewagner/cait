# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import numpy as np
from cait.fit.pm_fit import fit_pulse_shape


# ------------------------------------------------------------
# FUNCTION
# ------------------------------------------------------------

def generate_standard_event(events,
                            main_parameters,
                            pulse_height_intervall=[0.05, 1.5],
                            left_right_cutoff=0.1,
                            rise_time_intervall=[25, 80],
                            decay_time_intervall=[100, 5000],
                            onset_intervall=[3000, 6000],
                            remove_offset=True,
                            verb=False):
    if verb:
        print('{} Events handed.'.format(len(main_parameters)))

    use_indices = np.ones(len(main_parameters))

    # pulse height cut
    use_indices[main_parameters[:, 0] < pulse_height_intervall[0]] = 0
    use_indices[main_parameters[:, 0] > pulse_height_intervall[1]] = 0

    if verb:
        print('{} left after PH cut.'.format(len(np.where(use_indices == 1)[0])))

    # left - right cut
    use_indices[np.abs(main_parameters[:, 8]) > left_right_cutoff] = 0

    if verb:
        print('{} left after left - right cut.'.format(len(np.where(use_indices == 1)[0])))

    # rise time and decay cut
    use_indices[main_parameters[:, 2] > main_parameters[:, 1] + rise_time_intervall[1]] = 0
    use_indices[main_parameters[:, 2] < main_parameters[:, 1] + rise_time_intervall[0]] = 0

    if verb:
        print('{} left after rise time cut.'.format(len(np.where(use_indices == 1)[0])))

    use_indices[main_parameters[:, 5] > main_parameters[:, 3] + decay_time_intervall[1]] = 0
    use_indices[main_parameters[:, 5] < main_parameters[:, 3] + decay_time_intervall[0]] = 0

    if verb:
        print('{} left after decay time cut.'.format(len(np.where(use_indices == 1)[0])))

    # onset cut
    use_indices[main_parameters[:, 3] > onset_intervall[1]] = 0
    use_indices[main_parameters[:, 3] < onset_intervall[0]] = 0

    if verb:
        print('{} left after onset cut.'.format(len(np.where(use_indices == 1)[0])))

    if verb:
        print('{} Events used to generate Standardevent.'.format(len(np.where(use_indices == 1)[0])))

    # remove offset
    if remove_offset:
        events = np.subtract(events.T, np.mean(events[:, :1000], axis=1).T).T

    # generate the standardevent
    standardevent = np.mean(events[use_indices == 1], axis=0)
    standardevent /= np.max(standardevent)

    par = fit_pulse_shape(standardevent)

    return standardevent, par
