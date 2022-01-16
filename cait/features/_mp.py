# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
from ..fit._bl_fit import fit_quadratic_baseline
from ..filter._ma import box_car_smoothing
from ..filter._of import filter_event

# ------------------------------------------------------------
# MAIN PARAMETERS CLASS
# ------------------------------------------------------------

class MainParameters():
    """
    Class to contain the main parameters.

    :param pulse_height: float, the height of the event
    :param t_zero: int, the sample index where the rise starts
    :param t_rise: int, the sample index where the rise reaches 80%
    :param t_max: int, the sample index of the max event
    :param t_decaystart: int, the sample index where the peak falls down to 90%
    :param t_half: int, the sample index where the peak falls down to 73%
    :param t_end: int, the sample index where the peak falls down to 36%
    :param offset: float, the mean of the first 1/8 of the record length
    :param linear_drift: float, the linear slope of the event baseline
    :param quadratic_drift: float, the quadratic slope of the event baseline
    """

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
        """
        Method to print all the stored main parameters
        """
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
        """
        Method to compare the main parameters with those of another instance.

        :param other: the other instance of main parameters
        :return: bool, states if the main parameters are the same

        """
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
        """
        Returns an array with the main parameters that are stored

        :return: 1D array length 10, the main parameters
        """
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
                       color = 'r', zorder=10, fig=None):
        """
        Plots the main parameters on overlaid to an event

        :param down: int, the downsample rate of the event that should be overlaid
        :param offset_in_samples: int, set if the x axis does not start at zero
        :param color: string, the color of the main parameters in the scatter plot
        :param zorder: int, the plot with the highest zorder is plot on top of the others
            this should be choosen high, such that the main parameters are visible
        :param fig: object, the pyplot figure to which we want to plot the parameters
        """

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
            if fig is None:
                fig.scatter(x_values, y_values, color=color)
            else:
                plt.scatter(x_values, y_values, color=color)
        else:
            if fig is None:
                fig.scatter(x_values, y_values, color=color, zorder=zorder)
            else:
                plt.scatter(x_values, y_values, color=color, zorder=zorder)


    def get_differences(self):
        """
        Return the differences in the samples times

        :return: 1D array with length 2: (length_rise, length_peak, length_decay)
        """
        length_rise = self.t_rise - self.t_zero
        length_peak = self.t_decaystart - self.t_rise
        length_decay = self.t_end - self.t_decaystart

        return length_rise, length_peak, length_decay


# ------------------------------------------------------------
# CALCULATE MAIN PARAMETERS FUNCTION
# ------------------------------------------------------------

def get_times(t_zero, t_rise, t_decaystart, t_half, t_end):
    """
    Calculate the Rise Length, Peak Length, Length of First and Seconf Half of Decay

    :param t_zero: integer, time of the pulse onset
    :param t_rise: integer, time when rise is finished
    :param t_decaystart: integer, end of peak
    :param t_half: integer, half of decay
    :param t_end: integer, decay over time
    :return: (int: length of the rise,
            int: length of peak,
            int: length of first decay half,
            int: length of second decay half)
    """
    length_rise = t_rise - t_zero
    length_peak = t_decaystart - t_rise
    length_firsthalfdecay = t_half - t_decaystart
    length_secondhalfdecay = t_end - t_half

    return length_rise, length_peak, length_firsthalfdecay, length_secondhalfdecay


def calc_main_parameters(event, down=1, max_bounds=None, quad_drift=False):
    """
    Calculates the Main Parameters for an Event.
    Optional, the event can be downsampled by a given factor befor the calculation

    :param event: 1D array, the event
    :param down: integer, power of 2, the factor for downsampling
    :param max_bounds: tuple, the lower and upper index of the interval within which the maximum is searched
    :param quad_drift: bool, include quadratic drift in the calculation
    :return: instance of MainParameters, see :class:`~simulate.MainParameters`
    """

    length_event = len(event)
    if max_bounds is None:
        max_bounds = [0, length_event]

    offset = np.mean(event[:int(length_event/8)])

    # smoothing or downsampling
    if down == 1:
        event_smoothed = box_car_smoothing(event - offset)
    else:
        event_smoothed = event.reshape(int(length_event / down), down)
        event_smoothed = np.mean(event_smoothed, axis=1)
        event_smoothed = event_smoothed - offset
        max_bounds[0] = int(max_bounds[0]/down)
        max_bounds[1] = int(max_bounds[1] / down)

    length_event_smoothed = len(event_smoothed)

    # get the maximal pulse height and the time of the maximum
    maximum_pulse_height = np.max(event_smoothed[max_bounds[0]:max_bounds[1]])  # [idx_lower_region : idx_upper_region]
    # maximum_index = np.argmax(event_smoothed)  # [idx_lower_region : idx_upper_region]  + idx_lower_region

    if maximum_pulse_height > np.std(event_smoothed) or not quad_drift:  # typically this will be the case

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

    maximum_index = int(np.argmax(event_nodrift[max_bounds[0]:max_bounds[1]]) + max_bounds[0])
    # maximum_pulse_height = event_smoothed[maximum_index]
    maximum_pulse_height_condition = event_smoothed[maximum_index]

    # get the times
    t_zero = np.where(event_smoothed[:maximum_index] < 0.2 * maximum_pulse_height_condition)
    if t_zero[0].size > 0:
        t_zero = t_zero[0][-1]
    else:
        t_zero = 0

    t_rise = np.where(event_smoothed[t_zero:] > 0.8 * maximum_pulse_height_condition)
    if t_rise[0].size > 0:
        t_rise = t_rise[0][0] + t_zero
    else:
        t_rise = t_zero

    t_end = np.where(event_smoothed[maximum_index:] < 0.368 * maximum_pulse_height_condition)
    if t_end[0].size > 0:
        t_end = t_end[0][0] + maximum_index
    else:
        t_end = length_event_smoothed - 1

    t_half = np.where(event_smoothed[:t_end+1] > 0.736 * maximum_pulse_height_condition)
    if t_half[0].size > 0:
        t_half = t_half[0][-1]
    else:
        t_half = t_end

    t_decaystart = np.where(event_smoothed[:t_half+1] > 0.9 * maximum_pulse_height_condition)
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

def expectation(x, dist):
    """
    Calculate the expectation value of a random value function

    :param x: 1D array, the function of the random vaiable, applied to the x value array
    :param dist: 1D array of same length as the other array, the values of the distribution
    """
    return (1/len(dist))*np.dot(x, dist)

def distribution_skewness(density):
    """
    Calculate the skewness of a distribution, given the density as array

    :param density: 1D array, the density of the distribution we want the skewness from
    :return: float, the skewness of the given distribution
    """

    density = density/np.sum(density)

    x = np.arange(len(density))

    mu = expectation(x, density)
    sigma = expectation((x-mu)**2, density)
    std_scores = (x-mu)/sigma
    skew = expectation(std_scores**3, density)

    return skew


def calc_additional_parameters(event,
                               optimal_transfer_function,
                               down=1):
    """
    Calculate parameters additionally to the main parameters

    :param event: array, the event of which we want to calculate the additional main parameters
    :param optimal_transfer_function: array, the optimum filter transfer function; if None, the of values are filled
        with zeros instead
    :param down: int, if we want to downsample the event by a factor, this can be done here
    :return: List of float values parameters (maximum of array,
                     minimum of array,
                     variance of first 1/8 or array,
                     mean of first 1/8 or array,
                     variance of last 1/8 or array,
                     mean of last 1/8 or array,
                     variance of whole array,
                     mean of whole array,
                     skewness of whole array,
                     maximum of the derivative of the array,
                     index of maximum of the derivative of the array,
                     minimum of the derivative of the array,
                     index of minimum of the derivative of the array,
                     maximum of the filtered array,
                     index of the maximum of the filtered array,
                     distribution skewness of the filtered samples around the max of filtered array)
    """

    length_event = len(event)
    event = event - np.mean(event[:int(length_event/8)])

    # smoothing
    if down == 1:
        event_smoothed = box_car_smoothing(event)
    else:
        event_smoothed = event.reshape(int(length_event / down), down)
        event_smoothed = np.mean(event_smoothed, axis=1)

    length_event_smoothed = len(event_smoothed)

    # Max and Min
    max = np.max(event_smoothed)
    min = np.min(event_smoothed)

    # Variance and Mean of first 1 / 8 and last 1 / 8
    var_start = np.var(event_smoothed[:int(length_event_smoothed/8)])
    mean_start = np.mean(event_smoothed[:int(length_event_smoothed/8)])
    var_end = np.var(event_smoothed[-int(length_event_smoothed/ 8):])
    mean_end = np.mean(event_smoothed[-int(length_event_smoothed/ 8):])

    # Variance, Mean and Skewness of whole array
    var = np.var(event_smoothed)
    mean = np.mean(event_smoothed)
    skew = sci.skew(event_smoothed)

    # Max, Min Derivative and position
    der = np.diff(event_smoothed)
    der_max = np.max(der)
    der_maxind = np.argmax(der)
    der_min = np.min(der)
    der_minind = np.argmin(der)

    # Min Value OF: Position maximum and skewness of samples around
    if optimal_transfer_function is not None:
        filtered = filter_event(event, transfer_function=optimal_transfer_function)
        filtered_max = np.max(filtered[int(length_event/8):-int(length_event/8)])
        filtered_maxind = np.argmax(filtered[int(length_event/8):-int(length_event/8)]) + int(length_event/8)
        filtered_skew = distribution_skewness(filtered[filtered_maxind-int(length_event/32):filtered_maxind+int(length_event/32)])
    else:
        filtered_max = 0
        filtered_maxind = 0
        filtered_skew = 0

    return np.array([max,  # 0
                     min,  # 1
                     var_start,  # 2
                     mean_start,  # 3
                     var_end,  # 4
                     mean_end,  # 5
                     var,  # 6
                     mean,  # 7
                     skew,  # 8
                     der_max,  # 9
                     der_maxind,  # 10
                     der_min,  # 11
                     der_minind,  # 12
                     filtered_max,  # 13
                     filtered_maxind,  # 14
                     filtered_skew], # 15
                    )
