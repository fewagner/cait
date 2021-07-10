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
    Return a the value of a linear interpolation between (ya,yb) in the interval (a,b) at position x.

    :param x: The position where we evaluate the interpolation.
    :type x: float
    :param b: The upper limit of the interpolation interval.
    :type b: float
    :param a: The lower limit of the interpolation interval.
    :type a: float
    :param yb: The value at the upper limit of the interpolation interval.
    :type yb: float
    :param ya: The value at the lower limit of the interpolation interval.
    :type ya: float
    :return: The interpolated value.
    :rtype: float
    """
    return (yb - ya) / (b - a) * (x - a) + ya


def plot_S1(this_event, elements, color='r', xlim=None, offset=0):
    """
    Plot the function projected to S1 elements, i.e. piecewise affine.

    :param this_event: The event that we project on S1.
    :type this_event: 1D array
    :param elements: The elements of the grid.
    :type elements: List of 2-tuples or lists
    :param color: The color in which we plot the refinement.
    :param color: string
    """
    x_val = [el[0] for el in elements]
    y_val = [this_event[el[0]] for el in elements]
    x_val.append(elements[-1][1] - 1)
    y_val.append(this_event[elements[-1][1] - 1])
    x_val = np.array(x_val)
    y_val = np.array(y_val) - offset
    if xlim is None:
        plt.plot(x_val, y_val, color=color, zorder=15)
        plt.scatter(x_val, y_val, color=color)
    else:
        mask = (x_val >= np.array(xlim[0])) & (x_val <= np.array(xlim[1]))
        plt.plot(x_val[mask], y_val[mask], color=color, zorder=15)
        plt.scatter(x_val[mask], y_val[mask], color=color)


def refine(elements, to_refine):
    """
    Refines a given grid.

    :param elements: The current elements of the grid.
    :param elements: List of 2-tuples or lists
    :param to_refine: Same length as elements, flags all elements that are to refine.
    :type to_refine: List of bools
    :return: The elements of the refined grid.
    :rtype: List of 2-tuples or lists
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
    Calculate the standard deviations of the event from the S1 projection to the elements.

    :param this_event: The event.
    :type this_event: 1D array
    :param elements: The elements.
    :param elements: list of 2-tuples or lists
    :return: The standard deviations for all the elements.
    :return: 1D array
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
    given standard deviation threshold.

    :param this_event: The event to project on S1.
    :param this_event: 1D array
    :param std_thres: The threshold for the standard deviation of the deviation proj-true.
    :param std_thres: float
    :param verb: If true give verbal feedback about the progress.
    :param verb: bool
    :return: The elements of the grid of the S1 projection.
    :return: list of 2-tuples of lists
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

