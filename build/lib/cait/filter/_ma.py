
import numpy as np

def box_car_smoothing(event, length=50):
    event = np.pad(event, length, 'edge')
    event = 0.02 * np.convolve(event, np.array([1]).repeat(50), 'same')
    return event[length:-length]