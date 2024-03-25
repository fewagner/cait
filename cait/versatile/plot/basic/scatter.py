from typing import Union, List

import numpy as np

from ..viewer import Viewer

class Scatter(Viewer):
    """
    Plot a scatter graph. 

    :param y: The y-data to plot. You can either hand a list-like object (simple plotting of one scatter, or multiple scatters if multi-dimensional) or a dictionary whose keys are line names and whose values are again list-like objects (for each key a scatter is plotted and a legend entry is created. If the values in the dictionary are lists, the first entry is interpreted as x-values and the second as y-values).
    :type y: Union[List[float], dict]
    :param x: The x-data to plot (for any lines for which x-data was not explicitly specified, see above). If None is specified, the y-data is plotted over the data index. Defaults to None.
    :type x: List[float], optional
    :param kwargs: Keyword arguments for `Viewer` like 'width', 'height', 'xrange', 'ylabel'.
    :type kwargs: Any

    **Example:**
    ::
        import cait.versatile as vai

        vai.Scatter([1,2,3])
        vai.Scatter([[1,2,3], [5,6,7]])
        vai.Scatter({"first scatter": [1,2,3], "second scatter with x data": [[0,1,2],[3,4,5]]})
        vai.Scatter([1,2,3], x=[1,2,3], xrange=(-1, 4), backend="mpl")
    """
    def __init__(self, y: Union[List[float], dict], x: List[float] = None, **kwargs):

        super().__init__(**kwargs)

        if isinstance(y, dict):
            for key, value in y.items():
                value = np.squeeze(np.array(value))
                if value.ndim > 1:
                    self.add_scatter(x=value[0], y=value[1], name=key)
                else:
                    self.add_scatter(x=x, y=value, name=key)
        else:
            scatters = np.squeeze(np.array(y))
            if scatters.ndim > 1:
                for scatter in scatters:
                    self.add_scatter(x=x, y=scatter)
            else:
                self.add_scatter(x=x, y=scatters)

        self.show()