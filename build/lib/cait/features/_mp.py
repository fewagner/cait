# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from cait.fit import baseline_template_quad
from cait.filter import box_car_smoothing

# ------------------------------------------------------------
# MAIN PARAMETERS CLASS
# ------------------------------------------------------------

class MainParameters():

    def __init__(self,
                 pulse_height=0,
                 t_zero=0,
                 t_rise=0,
                 t_max=0,
                 t_decaystart=0,
                 t_half=0,
                 t_end=0,
                 offset=0,
                 linear_drift=0,
                 quadratic_drift=0):
        self.pulse_height = pulse_height
        self.t_zero = t_zero
        self.t_rise = t_rise
        self.t_max = t_max
        self.t_decaystart = t_decaystart
        self.t_half = t_half
        self.t_end = t_end
        self.offset = offset
        self.linear_drift = linear_drift
        self.quadratic_drift = quadratic_drift

    def print_all(self):
        print('Pulse height: ', self.pulse_height)
        print('Index of Rise Start: ', self.t_zero)
        print('Index of Rise End: ', self.t_rise)
        print('Index of Maximum: ', self.t_max)
        print('Index of Decay Start: ', self.t_decaystart)
        print('Index of Half Decay: ', self.t_half)
        print('Index of End Decay: ', self.t_end)
        print('Offset: ', self.offset)
        print('Linear Drift: ', self.linear_drift)
        print('Quadratic drift: ', self.quadratic_drift)

    def compare(self, other):
        if self.pulse_height != other.pulse_height:
            return False
        if self.t_zero != other.t_zero:
            return False
        if self.t_rise != other.t_rise:
            return False
        if self.t_max != other.t_max:
            return False
        if self.t_decaystart != other.t_decaystart:
            return False
        if self.t_half != other.t_half:
            return False
        if self.t_end != other.t_end:
            return False
        if self.offset != other.offset:
            return False
        if self.linear_drift != other.linear_drift:
            return False
        if self.quadratic_drift != other.quadratic_drift:
            return False

        return True

    def getArray(self):
        return np.array([self.pulse_height,
                         self.t_zero,
                         self.t_rise,
                         self.t_max,
                         self.t_decaystart,
                         self.t_half,
                         self.t_end,
                         self.offset,
                         self.linear_drift,
                         self.quadratic_drift])

    def plotParameters(self,
                       down=1,
                       offset_in_samples=0,
                       color = 'r', zorder=0):

        # t = np.linspace(0, nmbr_samples-1, nmbr_samples) - nmbr_samples/4
        # start rise, top rise, max, start decay, half decay, end decay
        x_values = [(self.t_zero - offset_in_samples) / down,
                    (self.t_rise - offset_in_samples) / down,
                    (self.t_max - offset_in_samples) / down,
                    (self.t_decaystart - offset_in_samples) / down,
                    (self.t_half - offset_in_samples) / down,
                    (self.t_end - offset_in_samples) / down]

        y_values = [
            self.offset + 0.2 * self.pulse_height,
            self.offset + 0.8 * self.pulse_height,
            self.offset + self.pulse_height,
            self.offset + 0.9 * self.pulse_height,
            self.offset + 0.736 * self.pulse_height,
            self.offset + 0.368 * self.pulse_height]

        if zorder == 0:
            plt.scatter(x_values, y_values, color=color)
        else:
            plt.scatter(x_values, y_values, color=color, zorder=zorder)


    def get_differences(self):
        length_rise = self.t_rise - self.t_zero
        length_peak = self.t_decaystart - self.t_rise
        length_decay = self.t_end - self.t_decaystart

        return length_rise, length_peak, length_decay


# ------------------------------------------------------------
# CALCULATE MAIN PARAMETERS FUNCTION
# ------------------------------------------------------------

def get_times(t_zero, t_rise, t_decaystart, t_half, t_end):
    length_rise = t_rise - t_zero
    length_peak = t_decaystart - t_rise
    length_firsthalfdecay = t_half - t_decaystart
    length_secondhalfdecay = t_end - t_half

    return length_rise, length_peak, length_firsthalfdecay, length_secondhalfdecay


def calc_main_parameters(event, down=1):

    length_event = len(event)

    offset = np.mean(event[:500])

    # box car smoothing
    if down == 1:
        event_smoothed = box_car_smoothing(event - offset)
    else:
        event_smoothed = event.reshape(int(length_event / down), down)
        event_smoothed = np.mean(event_smoothed, axis=1)
        event_smoothed = event_smoothed - offset

    length_event_smoothed = len(event_smoothed)

    # get the maximal pulse height and the time of the maximum
    maximum_pulse_height = np.max(event_smoothed)  # [idx_lower_region : idx_upper_region]
    # maximum_index = np.argmax(event_smoothed)  # [idx_lower_region : idx_upper_region]  + idx_lower_region

    if maximum_pulse_height > np.std(event_smoothed):  # typically this will be the case

        # get baseline offset and linear drift
        # offset = event_smoothed[0]
        linear_drift = (np.mean(event_smoothed[-int(500/down):]) - np.mean(event_smoothed[:int(500/down)])) / length_event
        quadratic_drift = 0

    else:  # if the pulse is very small then fit a quadratic baseline
        offset_fit, linear_drift, quadratic_drift = fit_quadratic_baseline(event_smoothed)
        offset = offset + offset_fit

    # get rid of the offset and linear drift
    event_nodrift = event_smoothed - linear_drift * np.linspace(0, length_event_smoothed - 1, length_event_smoothed)
    if quadratic_drift != 0:
        event_nodrift = event_nodrift - quadratic_drift * np.linspace(0, length_event_smoothed - 1, length_event_smoothed) ** 2

    maximum_index = np.argmax(event_nodrift)
    maximum_pulse_height = event_smoothed[maximum_index]
    # maximum_pulse_height_condition = event_nodrift[maximum_index]

    # get the times
    t_zero = np.where(event_smoothed[:maximum_index] < 0.2 * maximum_pulse_height)
    if t_zero[0].size > 0:
        t_zero = t_zero[0][-1]
    else:
        t_zero = 0

    t_rise = np.where(event_smoothed[t_zero:] > 0.8 * maximum_pulse_height)
    if t_rise[0].size > 0:
        t_rise = t_rise[0][0] + t_zero
    else:
        t_rise = 0

    t_end = np.where(event_smoothed[maximum_index:] < 0.368 * maximum_pulse_height)
    if t_end[0].size > 0:
        t_end = t_end[0][0] + maximum_index
    else:
        t_end = length_event_smoothed - 1

    t_half = np.where(event_smoothed[:t_end+1] > 0.736 * maximum_pulse_height)
    if t_half[0].size > 0:
        t_half = t_half[0][-1]
    else:
        t_half = t_end

    t_decaystart = np.where(event_smoothed[:t_half+1] > 0.9 * maximum_pulse_height)
    if t_decaystart[0].size > 0:
        t_decaystart = t_decaystart[0][-1]
    else:
        t_decaystart = t_half

    # return all parameters
    main_parameters = MainParameters(pulse_height=maximum_pulse_height,
                                     t_zero=t_zero * down,
                                     t_rise=t_rise * down,
                                     t_max=maximum_index * down,
                                     t_decaystart=t_decaystart * down,
                                     t_half=t_half * down,
                                     t_end=t_end * down,
                                     offset=offset,
                                     linear_drift=linear_drift,
                                     quadratic_drift=quadratic_drift)

    return main_parameters

def calc_additional_parameters(event, down=64):

    #TODO

    return