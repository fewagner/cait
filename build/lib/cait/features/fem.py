"""
"""

# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# FUNCTIONS FOR REFINEMENT
# -----------------------------------------------------------


def interpol(x, b, a, yb, ya):
    return (yb - ya) / (b - a) * (x - a) + ya


def plot_S1(this_event, elements, color='r'):
    x_val = [el[0] for el in elements]
    y_val = [this_event[el[0]] for el in elements]
    x_val.append(elements[-1][1] - 1)
    y_val.append(this_event[elements[-1][1] - 1])
    plt.plot(x_val, y_val, color=color)
    plt.scatter(x_val, y_val, color=color)


def refine(elements, to_refine):
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
        elements = refine(elements=elements, to_refine=to_refine)
        stds = calc_stds(this_event, elements)
        if verb:
            print("Standardabweichungen: ", stds)
        to_refine = np.array([val > std_thres for val in stds])
        # print(elements, to_refine)

    return elements

# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------



