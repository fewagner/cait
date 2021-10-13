import numpy as np

def outside(ex, ey, x, y1, y2):
    # TODO
    retval = np.zeros(len(ex), dtype=bool)
    cond = ex <= x[-1]
    cond = np.logical_and(cond, x[0] <= ex)
    retval[cond] = ey[cond] <= np.interp(ex[cond], x, y2)
    retval[cond] = np.logical_and(retval[cond], np.interp(ex[cond], x, y1) <= ey[cond])
    return np.logical_not(retval)