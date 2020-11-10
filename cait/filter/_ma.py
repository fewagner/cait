
import numpy as np

def box_car_smoothing(event, length=50):
    """
    Calculates a moving average on an event array and returns the smoothed event

    :param event: 1D array, the event to calcualte the MA
    :param length: the length of the moving average
    :return: 1D array the smoothed array
    """
    event = np.pad(event, length, 'edge')
    event = 0.02 * np.convolve(event, np.array([1]).repeat(50), 'same')
    return event[length:-length]