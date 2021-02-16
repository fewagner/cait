# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# FUNCTIONS FOR REFINEMENT
# -----------------------------------------------------------


def interpol(x, b, a, yb, ya):
    """
    Return a the value of a linear interpolation between (ya,yb) in the interval (a,b) at position x

    :param x: float, the position where we evaluate the interpolation
    :param b: float, the upper limit of the interpolation interval
    :param a: float, the lower limit of the interpolation interval
    :param yb: float, the value at the upper limit of the interpolation interval
    :param ya: float, the value at the lower limit of the interpolation interval
    :return: float, the interpolated value
    """
    return (yb - ya) / (b - a) * (x - a) + ya


def plot_S1(this_event, elements, color='r'):
    """
    Plot the function projected to S1 elements, i.e. piecewise affine

    :param this_event: 1D array, the event that we project on S1
    :param elements: List of 2-tuples or lists, the elements of the grid
    :param color: string, the color in which we plot the refinement
    :return: -
    """
    x_val = [el[0] for el in elements]
    y_val = [this_event[el[0]] for el in elements]
    x_val.append(elements[-1][1] - 1)
    y_val.append(this_event[elements[-1][1] - 1])
    plt.plot(x_val, y_val, color=color, zorder=15)
    plt.scatter(x_val, y_val, color=color)


def refine(elements, to_refine):
    """
    Refines a given grid

    :param elements: List of 2-tuples or lists, the current elements of the grid
    :param to_refine: List of bools, same length as elements, flags all elements that are to refine
    :return: List of 2-tuples or lists, the elements of the refined grid
    """
    new_elements = []

    for ref, el in zip(to_refine, elements):
        if ref:
            new_length = (el[1] - el[0]) / 2
            new_elements.append((el[0], int(el[0] + new_length)))
            new_elements.append((int(el[1] - new_length), el[1]))
        else:
            new_elements.append(el)

    return new_elements


def calc_stds(this_event, elements):
    """
    Calculate the standard deviations of the event from the S1 projection to the elements

    :param this_event: 1D array, the event
    :param elements: list of 2-tuples or lists, the elements
    :return: 1D array, the standard deviations for all the elements
    """
    stds = []

    for el in elements:
        N = el[1] - el[0]
        interpolation = np.array(
            [interpol(x=i, a=el[0], b=el[1], ya=this_event[el[0]], yb=this_event[el[1] - 1]) for i in
             range(el[0], el[1])])
        # print('Interpolation: ', interpolation)
        residual = interpolation - this_event[el[0]:el[1]]
        # print("Residual: ", residual)
        stds.append(np.sum(np.abs(residual)) / N)

    return np.array(stds)


def get_elements(this_event, std_thres, verb=False):
    """
    Calculate the projection to S1 with algorithm for automatic refinement, depending on
    given standard deviation threshold

    :param this_event: 1D array, the event to project on S1
    :param std_thres: float, the threshold for the standard deviation of the deviation proj-true
    :param verb: bool, if true give verbal feedback about the progress
    :return: list of 2-tuples of lists, the elements of the grid of the S1 projection
    """
    event_length = len(this_event)

    # create first elements
    elements = [(0, event_length)]

    # refine one step
    stds = calc_stds(this_event, elements)
    if verb:
        print("Standardabweichungen: ", stds)
    to_refine = np.array([val > std_thres for val in stds])
    # print(elements, to_refine)

    while any(to_refine):
        elements = refine(elements=elements,
                          to_refine=to_refine)
        stds = calc_stds(this_event, elements)
        if verb:
            print("Standardabweichungen: ", stds)
        to_refine = np.array([val > std_thres for val in stds])
        # print(elements, to_refine)

    return elements

