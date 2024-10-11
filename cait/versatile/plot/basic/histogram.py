from typing import Union, List

import numpy as np

from ..viewer import Viewer

class Histogram(Viewer):
    """
    Plot a Histogram. 

    :param data: The data to bin and plot. You can either hand a list-like object (simple plotting of one histogram, or multiple histograms if multi-dimensional) or a dictionary whose keys are histogram names and whose values are again list-like objects (for each key a histogram is binned, plotted and a legend entry is created).
    :type data: Union[List[float], dict]
    :param bins: The binning data to use. If None, the binning is done automatically. An integer is interpreted as the desired total number of bins. You can also parse a tuple of the form `(start, end, nbins)` to bin the data between `start` and `end` into a total of `nbins` bins
    :type bins: Union[None, int, tuple], optional
    :param weight: The bar hights in the histogram are scaled by this factor.
    :type weight: float, optional
    :param kwargs: Keyword arguments for `Viewer` like 'width', 'height', 'xrange', 'ylabel'.
    :type kwargs: Any

    **Example:**

    .. code-block:: python
    
        import cait.versatile as vai

        vai.Histogram([1,2,3])
        vai.Histogram([[1,2,3], [5,6,7]])
        vai.Histogram({"first hist": [1,2,3], "second hist": [0,1,2]})
        vai.Histogram([1,2,3], bins=100, backend="mpl")
        vai.Histogram([1,2,3], bins=(0,5,100), backend="mpl")
    """
    def __init__(self, data: Union[List[float], dict], bins: Union[tuple, int] = None, weight: float = 1., **kwargs):

        super().__init__(**kwargs)

        if isinstance(data, dict):
            for key, value in data.items():
                self.add_histogram(bins=bins, data=value, weight=weight, name=key)
        else:
            hists = np.squeeze(np.array(data))
            if hists.ndim > 1:
                for hist in hists:
                    self.add_histogram(bins=bins, data=hist, weight=weight)
            else:
                self.add_histogram(bins=bins, data=hists, weight=weight)

        self.show()