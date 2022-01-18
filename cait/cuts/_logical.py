import numpy as np

def outside(ex, ey, x, y1, y2):
    """
    Cut all events inside a connected set of trapezoids in an XY plane.

    :param ex: The x coordinates of the events.
    :type ex: 1D numpy array
    :param ey: The y coordinates of the events.
    :type ey: 1D numpy array
    :param x: The x values of the region, a list of arbirary length.
    :type x: list
    :param y1: The lower y values of the region, a list of same length as x.
    :type y1: list
    :param y2: The upper y values of the region, a list of same length as x, the values must be larger than the
        corresponding ones in y1.
    :type y2: list
    :return: The cut flag corresponding to the events.
    :rtype: 1D numpy bool array

    >>> import cait as ai
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> nx, ny = (1000, 1000)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = np.meshgrid(x, y)

    >>> cutflag = ai.cuts.outside(ex=xv.reshape(-1),
    ...                          ey=yv.reshape(-1),
    ...                          x=[0.1, 0.4, 0.9],
    ...                          y1=[0.1, 0.2, 0.1],
    ...                          y2=[0.7, 0.9, 0.3],
    ...                          )

    >>> plt.contourf(xv, yv, cutflag.reshape((nx, ny)))
    >>> plt.show()
    """
    assert len(x) == len(y1) == len(y2), ''
    retval = np.zeros(len(ex), dtype=bool)
    cond = ex <= x[-1]
    cond = np.logical_and(cond, x[0] <= ex)
    retval[cond] = ey[cond] <= np.interp(ex[cond], x, y2)
    retval[cond] = np.logical_and(retval[cond], np.interp(ex[cond], x, y1) <= ey[cond])
    return np.logical_not(retval)