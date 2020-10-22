# -----------------------------------------------------
# IMPORTS
# -----------------------------------------------------

import numpy as np

# -----------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------

def baseline_template_quad(t, c0, c1, c2):
    return c0 + t * c1 + t ** 2 * c2


def baseline_template_cubic(t, c0, c1, c2, c3):
    return c0 + t * c1 + t ** 2 * c2 + t ** 3 * c3


def pulse_template(t, t0, An, At, tau_n, tau_in, tau_t):

    pulse = (t>=t0).astype(float)
    t_red = t[t>=t0]
    pulse[t>=t0] *= (An*(np.exp(-(t_red-t0)/tau_n) - np.exp(-(t_red-t0)/tau_in)) + \
                     At*(np.exp(-(t_red-t0)/tau_t) - np.exp(-(t_red-t0)/tau_n)))
    return pulse
