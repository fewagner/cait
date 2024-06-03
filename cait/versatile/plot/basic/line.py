from typing import Union, List

import numpy as np

from ..viewer import Viewer

class Line(Viewer):
    """
    Plot a line graph. 

    :param y: The y-data to plot. You can either hand a list-like object (simple plotting of one line, or multiple lines if multi-dimensional) or a dictionary whose keys are line names and whose values are again list-like objects (for each key a line is plotted and a legend entry is created. If the values in the dictionary are lists, the first entry is interpreted as x-values and the second as y-values).
    :type y: Union[List[float], dict]
    :param x: The x-data to plot (for any lines for which x-data was not explicitly specified, see above). If None is specified, the y-data is plotted over the data index. Defaults to None.
    :type x: List[float], optional
    :param kwargs: Keyword arguments for `Viewer` like 'width', 'height', 'xrange', 'ylabel'.
    :type kwargs: Any

    **Example:**
    ::
    
        import cait.versatile as vai

        vai.Line([1,2,3])
        vai.Line([[1,2,3], [5,6,7]])
        vai.Line({"first line": [1,2,3], "second line with x data": [[0,1,2],[3,4,5]]})
        vai.Line([1,2,3], x=[1,2,3], xrange=(-1, 4), backend="mpl")
    """
    def __init__(self, y: Union[List[float], List[List[float]], dict], x: List[float] = None, **kwargs):

        super().__init__(**kwargs)

        if isinstance(y, dict):
            for key, value in y.items():
                value = np.squeeze(np.array(value))
                if value.ndim > 1:
                    self.add_line(x=value[0], y=value[1], name=key)
                else:
                    self.add_line(x=x, y=value, name=key)
        else:
            lines = np.squeeze(np.array(y))
            if lines.ndim > 1:
                for line in lines:
                    self.add_line(x=x, y=line)
            else:
                self.add_line(x=x, y=lines)

        self.show()