from typing import Union, List

import numpy as np

from ..viewer import Viewer

class Heatmap(Viewer):
    """
    Plot a Heatmap. 

    :param x: The x-data to bin and plot.
    :type x: List[float]
    :param y: The y-data to bin and plot.
    :type y: List[float]
    :param bins: The binning data to use. If None, the binning is done automatically. An integer is interpreted as the desired total number of bins on both axes. You can also parse a tuple of the form `(start, end, nbins)` to bin the data between `start` and `end` into a total of `nbins` bins. A numpy array is interpreted as the desired bin edges. If you pass a tuple of length two, you can specify either of the aforementioned arguments for both axes separately.
    :type bins: Union[None, int, tuple, np.ndarray], optional
    :param kwargs: Keyword arguments for `Viewer` like 'width', 'height', 'xrange', 'ylabel'.
    :type kwargs: Any

    **Example:**

    .. code-block:: python
    
        import cait.versatile as vai
        import numpy as np

        xdata = np.random.normal(size=10000)
        ydata = np.random.normal(loc=10, scale=3, size=10000)

        vai.Heatmap(xdata, 
                    ydata, 
                    bins=(np.linspace(-5, 5, 30), np.linspace(2, 20, 30)), 
                    xlabel="xdata", 
                    ylabel="ydata")
    """
    def __init__(self, x: List[float], y: List[float], bins: Union[tuple, int] = None, **kwargs):

        super().__init__(**kwargs)

        x = np.squeeze(np.array(x))
        y = np.squeeze(np.array(y))
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError(f"x and y must be 1d arrays. Got {x.ndim}, {y.ndim}")
        
        self.add_heatmap(x=x, y=y, bins=bins)

        self.show()