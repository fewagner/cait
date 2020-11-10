# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import numpy as np
from ..fit._pm_fit import fit_pulse_shape
from ..fit._templates import pulse_template

# ------------------------------------------------------------
# FUNCTION
# ------------------------------------------------------------

def generate_standard_event(events,
                            main_parameters,
                            labels=None,
                            correct_label = 1,
                            pulse_height_intervall=None,
                            left_right_cutoff=None,
                            rise_time_intervall=None,
                            decay_time_intervall=None,
                            onset_intervall=None,
                            remove_offset=True,
                            verb=False,
                            scale_fit_height=True,
                            sample_length=0.04):
    """
    Calculates the standard event and fits the pulse shape model.

    :param events: 2D array (nmbr_events, record_length), the events to calculate the SEV from
    :param main_parameters: 2D array (nmbr_events, nmbr_mp=10), the mp of the events
    :param labels: None or 1D array (nmbr_events), the labels of the events, if set only the correct
        labels are included in the SEV generation
    :param correct_label: int, the correct label to calc SEV from, 1==events, 2==testpulses
    :param pulse_height_intervall: None or 2-tuple or list, the interval in which the PH may be to be included
    :param left_right_cutoff: None or float, the maximal slope of the event
    :param rise_time_intervall: None or 2-tuple or list, the interval in which the rise time may be to be included
    :param decay_time_intervall: None or 2-tuple or list, the interval in which the decay time may be to be included
    :param onset_intervall: None or 2-tuple or list, the interval in which the onset time may be to be included
    :param remove_offset: bool, if True the offset of the events is removed before building mean for SEV;
        highly recommended!
    :param verb: bool, if True verbal feedback about the progress of the program is provided
    :param scale_fit_height: bool, if True the fitpar of the sev are scaled to height 1 after the fit
    :param sample_length: float, the length of one sample in milliseconds --> needed for the fit!
    :return: tuple of (sev, fitpar): sev is 1D array with len=record_length ... the calculated sev,
        fitpar is 1D array with len=parameters of fit (i think 6) ... the fit parameters
    """
    if verb:
        print('{} Events handed.'.format(len(main_parameters)))

    use_indices = np.ones(len(main_parameters))

    # use only those that are labeled as event

    if not labels is None:
        use_indices[labels != correct_label] = 0

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

    par = fit_pulse_shape(standardevent, sample_length=sample_length)

    if scale_fit_height:
        t = (np.arange(0, len(standardevent), dtype=float) - len(standardevent) / 4) * sample_length
        fit_max = np.max(pulse_template(t, *par))
        par[1] /= fit_max
        par[2] /= fit_max

    return standardevent, par
