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
                            correct_label=1,
                            pulse_height_interval=None,
                            left_right_cutoff=None,
                            rise_time_interval=None,
                            decay_time_interval=None,
                            onset_interval=None,
                            remove_offset=True,
                            verb=False,
                            scale_fit_height=True,
                            scale_to_unit=True,
                            sample_length=0.04,
                            t0_start=None,
                            opt_start=False):
    """
    Calculates the standard event and fits the pulse shape model.

    :param events: The events to calculate the SEV from.
    :type events: 2D array of shape (nmbr_events, record_length)
    :param main_parameters: The main parameters of the events.
    :type main_parameters: 2D array of shape (nmbr_events, nmbr_mp=10)
    :param labels: The labels of the events, if set only the correct labels are included in the SEV generation.
    :type labels: None or 1D array of shape (nmbr_events)
    :param correct_label: The correct label to calc SEV from, 1==events, 2==testpulses.
    :type correct_label: int
    :param pulse_height_interval: The interval in which the PH may be to be included.
    :type pulse_height_interval: None or 2-tuple or list
    :param left_right_cutoff: The maximal abs(R - L) baseline difference of the event.
    :type left_right_cutoff: None or float
    :param rise_time_interval: The interval in ms in which the rise time may be to be included.
    :type rise_time_interval: None or 2-tuple or list
    :param decay_time_interval: The interval in ms in which the decay time may be to be included.
    :type decay_time_interval: None or 2-tuple or list
    :param onset_interval: The interval in which the onset time in ms may be to be included.
    :type onset_interval: None or 2-tuple or list
    :param remove_offset: If True the offset of the events is removed before building mean for SEV;
        highly recommended!
    :type remove_offset: bool
    :param verb: If True verbal feedback about the progress of the program is provided.
    :type verb: bool
    :param scale_fit_height: If True the fitpar of the sev are scaled to height 1 after the fit.
    :type scale_fit_height: bool
    :param sample_length: The length of one sample in milliseconds --> needed for the fit!
    :type sample_length: float
    :param t0_start: The start value for t0.
    :type t0_start: float
    :param opt_start: If activated the starting values are searched with a differential evolution algorithm.
    :type opt_start: bool
    :return: The calculated sev, the fit parameters.
    :rtype: tuple of two 1D arrays with shape (record_length, nmbr_fitpar)
    """
    if verb:
        print('{} Events handed.'.format(len(main_parameters)))

    record_length = events.shape[1]

    use_indices = np.ones(len(main_parameters))

    # use only those that are labeled as event

    if not labels is None:
        use_indices[labels != correct_label] = 0

        if verb:
            print('{} with Labels: {}'.format(len(np.where(use_indices == 1)[0]), correct_label))

    # pulse height cut
    if not pulse_height_interval is None:
        use_indices[main_parameters[:, 0] < pulse_height_interval[0]] = 0
        use_indices[main_parameters[:, 0] > pulse_height_interval[1]] = 0

        if verb:
            print('{} left after PH cut.'.format(len(np.where(use_indices == 1)[0])))

    # left - right cut
    if not left_right_cutoff is None:
        use_indices[np.abs(main_parameters[:, 8]*record_length) > left_right_cutoff] = 0

        if verb:
            print('{} left after left - right cut.'.format(len(np.where(use_indices == 1)[0])))

    # rise time cut
    if not rise_time_interval is None:
        use_indices[main_parameters[:, 2]*sample_length > (main_parameters[:, 1])*sample_length + rise_time_interval[1]] = 0
        use_indices[main_parameters[:, 2]*sample_length < (main_parameters[:, 1])*sample_length + rise_time_interval[0]] = 0

        if verb:
            print('{} left after rise time cut.'.format(len(np.where(use_indices == 1)[0])))

    # decay time cut
    if not decay_time_interval is None:
        use_indices[main_parameters[:, 6]*sample_length > (main_parameters[:, 4])*sample_length + decay_time_interval[1]] = 0
        use_indices[main_parameters[:, 6]*sample_length < (main_parameters[:, 4])*sample_length + decay_time_interval[0]] = 0

        if verb:
            print('{} left after decay time cut.'.format(len(np.where(use_indices == 1)[0])))

    # onset cut
    if not onset_interval is None:
        use_indices[(main_parameters[:, 1] - record_length/4)*sample_length > onset_interval[1]] = 0
        use_indices[(main_parameters[:, 1] - record_length/4)*sample_length < onset_interval[0]] = 0

        if verb:
            print('{} left after onset cut.'.format(len(np.where(use_indices == 1)[0])))

    if verb:
        print('{} Events used to generate Standardevent.'.format(len(np.where(use_indices == 1)[0])))

    # remove offset
    if not remove_offset is None:
        events = events - np.mean(events[:, :1000], axis=1)[:, np.newaxis]

    # generate the standardevent
    standardevent = np.mean(events[use_indices == 1], axis=0)
    if scale_to_unit:
        standardevent /= np.max(standardevent)

    if t0_start is None:
        t0_start = -3
    par = fit_pulse_shape(standardevent, sample_length=sample_length, t0_start=t0_start, opt_start=opt_start)

    if scale_fit_height:
        t = (np.arange(0, len(standardevent), dtype=float) - len(standardevent) / 4) * sample_length
        fit_max = np.max(pulse_template(t, *par))
        print('Parameters [t0, An, At, tau_n, tau_in, tau_t]:\n', par)
        if not np.isclose(fit_max, 0):
            par[1] /= fit_max
            par[2] /= fit_max

    return standardevent, par
