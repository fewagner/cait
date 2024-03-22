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
    :param xlabel: x-label for the histogram.
    :type xlabel: str, optional
    :param ylabel: y-label for the histogram.
    :type ylabel: str, optional
    :param xscale: x-scale for the histogram. Either of ['linear', 'log'], defaults to 'linear'.
    :type xscale: str, optional
    :param yscale: y-scale for the histogram. Either of ['linear', 'log'], defaults to 'linear'.
    :type yscale: str, optional
    :param xrange: x-range for the plot. A tuple of (xmin, xmax), defaults to None, i.e. auto-scaling.
    :type xrange: tuple, optional
    :param yrange: y-range for the plot. A tuple of (ymin, ymax), defaults to None, i.e. auto-scaling.
    :type yrange: tuple, optional
    :param kwargs: Keyword arguments for `Viewer`.
    :type kwargs: Any

    **Example:**
    ::
        import cait.versatile as vai

        vai.Histogram([1,2,3])
        vai.Histogram([[1,2,3], [5,6,7]])
        vai.Histogram({"first hist": [1,2,3], "second hist": [0,1,2]})
        vai.Histogram([1,2,3], bins=100, backend="mpl")
        vai.Histogram([1,2,3], bins=(0,5,100), backend="mpl")
    """
    def __init__(self, data: Union[List[float], dict], bins: Union[tuple, int] = None, xlabel: str = None, ylabel: str = None, xscale: str = 'linear', yscale: str = 'linear', xrange: tuple = None, yrange: tuple = None, **kwargs):

        super().__init__(**kwargs)

        if isinstance(data, dict):
            for key, value in data.items():
                self.add_histogram(bins=bins, data=value, name=key)
        else:
            hists = np.squeeze(np.array(data))
            if hists.ndim > 1:
                for hist in hists:
                    self.add_histogram(bins=bins, data=hist)
            else:
                self.add_histogram(bins=bins, data=hists)

        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)
        self.set_xscale(xscale)
        self.set_yscale(yscale)
        self.set_xrange(xrange)
        self.set_yrange(yrange)

        self.show()