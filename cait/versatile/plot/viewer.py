from typing import Callable, Union, List, Tuple

from .backends.impl_plotly import BaseClassPlotly
from .backends.impl_matplotlib import BaseClassMPL
from .backends.impl_uniplot import BaseClassUniplot
from .backends.helper import auto_backend

class Viewer():
    """Class for plotting data given a dictionary of instructions (see below).
    For convenience, the axis properties can alternatively be set as keyword arguments, too, and will override the axis properties contained in the dictionary.

    :param data: Data dictionary containing line/scatter/axes information (see below), defaults to None
    :type data: dict, optional
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl', 'uniplot', 'auto'], i.e. plotly, matplotlib or uniplot (command line based), defaults to 'auto' which uses 'plotly' in notebooks and 'uniplot' otherwise.
    :type backend: str, optional
    :param xlabel: x-label for the plot.
    :type xlabel: str, optional
    :param ylabel: y-label for the plot.
    :type ylabel: str, optional
    :param clabel: Label for the colour axis of the plot (if applicable).
    :type clabel: str, optional
    :param xscale: x-scale for the plot. Either of ['linear', 'log'], defaults to 'linear'.
    :type xscale: str, optional
    :param yscale: y-scale for the plot. Either of ['linear', 'log'], defaults to 'linear'.
    :type yscale: str, optional
    :param cscale: Scale for the colour axis of the plot (if applicable). Either of ['linear', 'log'], defaults to 'linear'.
    :type cscale: str, optional
    :param xrange: x-range for the plot. A tuple of (xmin, xmax), defaults to None, i.e. auto-scaling.
    :type xrange: tuple, optional
    :param yrange: y-range for the plot. A tuple of (ymin, ymax), defaults to None, i.e. auto-scaling.
    :type yrange: tuple, optional
    :param crange: Range for colour axis of the plot (if applicable). A tuple of (cmin, cmax), defaults to None, i.e. auto-scaling.
    :type crange: tuple, optional
    :param cmap: Colour map to use (if applicable). If None, the default colour map for the chosen template of the chosen backend will be used.
    :type cmap: str, optional

    :param template: Valid backend theme. For `plotly` either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], for `mpl` either of ['default', 'classic', 'Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'], defaults to 'ggplot2' for `backend=plotly` and to 'seaborn' for `backend=mpl`. `template` has no effect for backend 'uniplot'.
    :type template: str, optional
    :param height: Figure height, defaults to 500 for `backend=plotly`, 3 for `backend=mpl` and 17 for `backend=uniplot`
    :type height: int, optional
    :param width: Figure width, defaults to 700 for `backend=plotly`, 5 for `backend=mpl` and 60 for `backend=uniplot`
    :type width: int, optional
    :param show_controls: Show button controls to interact with the figure. The available buttons depend on the plotting backend. Defaults to False
    :type show_controls: bool

    **Convention for 'data' Dictionary:**
    
    .. code-block:: python
    
        data = { 
                "line": { 
                    "line1": [x_data1, y_data1],
                    "line2": [x_data2, y_data2]
                    },
                "scatter": {
                    "scatter1": [x_data1, y_data1],
                    "scatter2": [x_data2, y_data2]
                    },
                "histogram": {
                    "hist1": [bin_data1, hist_data1],
                    "hist2": [bin_data2, hist_data2]
                    },
                "heatmap": {
                    "heat1": [bin_data1, xdata1, ydata1],
                    "heat2": [bin_data2, xdata2, ydata2]
                    },
                "axes": {
                    "xaxis": {
                        "label": "xlabel",
                        "scale": "linear",
                        "range": (0, 10)
                        },
                    "yaxis": {
                        "label": "ylabel",
                        "scale": "log",
                        "range": (0, 10)
                        },
                    "caxis": {
                        "label": "clabel",
                        "scale": "linear",
                        "range": (0, 10),
                        "cmap": "plasma"
                        }
                    }
                }
    """
    def __init__(self, 
                 data=None, 
                 backend="auto", 
                 xlabel: str = None, 
                 ylabel: str = None,
                 clabel: str = None, 
                 xscale: str = None, 
                 yscale: str = None,
                 cscale: str = None,
                 xrange: tuple = None, 
                 yrange: tuple = None, 
                 crange: tuple = None,
                 cmap: str = None,
                 **kwargs):

        if backend=="auto": backend = auto_backend()

        if backend=="plotly":
            self.fig_widget = BaseClassPlotly(**kwargs)
        elif backend=="mpl":
            self.fig_widget = BaseClassMPL(**kwargs)
        elif backend=="uniplot":
            self.fig_widget = BaseClassUniplot(**kwargs)
        else:
            raise NotImplementedError('Only backend "plotly", "mpl" and "uniplot" are supported.')

        self.visible = False

        if data is not None: 
            self.plot(data)  
            self.show()

        x_config = dict()
        if xlabel: x_config["label"] = xlabel
        if xscale: x_config["scale"] = xscale
        if xrange: x_config["range"] = xrange

        y_config = dict()
        if ylabel: y_config["label"] = ylabel
        if yscale: y_config["scale"] = yscale
        if yrange: y_config["range"] = yrange

        c_config = dict()
        if clabel: c_config["label"] = clabel
        if cscale: c_config["scale"] = cscale
        if crange: c_config["range"] = crange
        if cmap:   c_config["cmap"]  = cmap

        if x_config: self.fig_widget._set_axes(dict(xaxis=x_config))
        if y_config: self.fig_widget._set_axes(dict(yaxis=y_config))
        if c_config: self.fig_widget._set_axes(dict(caxis=c_config))

    def _add_button(self, text: str, callback: Callable, tooltip: str = None, where: int = -1, key: str = None):
        self.fig_widget._add_button(text, callback, tooltip, where, key)

    def show_legend(self, show: bool = True):
        """
        Show/hide legend.

        :param show: If True, legend is shown. If False, legend is hidden.
        :type show: bool
        """
        self.fig_widget._show_legend(show)

    def set_xlabel(self, xlabel: str):
        """
        Set the x-label of the axis.

        :param xlabel: x-label
        :type xlabel: str
        """
        self.fig_widget._set_axes(dict(xaxis={"label":xlabel}))

    def set_ylabel(self, ylabel: str):
        """
        Set the y-label of the axis.

        :param ylabel: y-label
        :type ylabel: str
        """
        self.fig_widget._set_axes(dict(yaxis={"label":ylabel}))

    def set_clabel(self, clabel: str):
        """
        Set the colour axis label (if applicable).

        :param clabel: colour axis label
        :type clabel: str
        """
        self.fig_widget._set_axes(dict(caxis={"label":clabel}))

    def set_xscale(self, xscale: str):
        """
        Set the x-scale of the axis. Either linear or logarithmic.

        :param xscale: x-scale. Either of ["linear", "log"]
        :type xscale: str
        """
        self.fig_widget._set_axes(dict(xaxis={"scale":xscale}))

    def set_yscale(self, yscale: str):
        """
        Set the y-scale of the axis. Either linear or logarithmic.

        :param yscale: y-scale. Either of ["linear", "log"]
        :type yscale: str
        """
        self.fig_widget._set_axes(dict(yaxis={"scale":yscale}))

    def set_cscale(self, cscale: str):
        """
        Set the scale of colour axis (if applicable). Either linear or logarithmic.

        :param cscale: Colour axis scale. Either of ["linear", "log"]
        :type cscale: str
        """
        self.fig_widget._set_axes(dict(caxis={"scale":cscale}))

    def set_xrange(self, xrange: tuple):
        """
        Set the x-range of the axis.

        :param xrange: x-range. A tuple of (xmin, xmax)
        :type xrange: tuple
        """
        self.fig_widget._set_axes(dict(xaxis={"range":xrange}))

    def set_yrange(self, yrange: tuple):
        """
        Set the y-range of the axis.

        :param yrange: y-range. A tuple of (ymin, ymax)
        :type yrange: tuple
        """
        self.fig_widget._set_axes(dict(yaxis={"range":yrange}))

    def set_crange(self, crange: tuple):
        """
        Set the colour axis range (if applicable).

        :param crange: Colour axis range. A tuple of (ymin, ymax)
        :type crange: tuple
        """
        self.fig_widget._set_axes(dict(caxis={"range":crange}))

    def add_line(self, x: List[float], y: List[float], name: str = None):
        """
        Add a line plot to the figure. If a name is provided, it is registered and can later be updated.

        :param x: The x-data of the line.
        :type x: List[float]
        :param y: The y-data of the line.
        :type y: List[float]
        :param name: The name of the line in the legend and its unique identifier for later updates. If None, the line does not show up in the legend and is not registered for later update.
        :type name: str, optional
        """
        self.fig_widget._add_line(x, y, name)

    def add_scatter(self, x: List[float], y: List[float], name: str = None):
        """
        Add a scatter plot to the figure. If a name is provided, it is registered and can later be updated.

        :param x: The x-data of the scatter.
        :type x: List[float]
        :param y: The y-data of the scatter.
        :type y: List[float]
        :param name: The name of the scatter in the legend and its unique identifier for later updates. If None, the scatter does not show up in the legend and is not registered for later update.
        :type name: str, optional
        """
        self.fig_widget._add_scatter(x, y, name)

    def add_histogram(self, bins: Union[int, tuple, list], data: List[float], name: str = None):
        """
        Add a histogram to the figure. If a name is provided, it is registered and can later be updated.

        :param bins: The binning data to use. If None, the binning is done automatically. An integer is interpreted as the desired total number of bins. You can also parse a tuple of the form `(start, end, nbins)` to bin the data between `start` and `end` into a total of `nbins` bins
        :type bins: Union[None, int, tuple], optional
        :param data: The data to bin.
        :type data: List[float]
        :param name: The name of the histogram in the legend and its unique identifier for later updates. If None, the histogram does not show up in the legend and is not registered for later update.
        :type name: str, optional
        """
        self.fig_widget._add_histogram(bins, data, name)

    def add_heatmap(self, x: List[float], y: List[float], bins: Union[int, tuple, list], name: str = None):
        """
        Add a heatmap to the figure. If a name is provided, it is registered and can later be updated.

        :param x: The x-data to bin and plot.
        :type x: List[float]
        :param y: The y-data to bin and plot.
        :type y: List[float]
        :param bins: The binning data to use. If None, the binning is done automatically. An integer is interpreted as the desired total number of bins on both axes. You can also parse a tuple of the form `(start, end, nbins)` to bin the data between `start` and `end` into a total of `nbins` bins. A numpy array is interpreted as the desired bin edges. If you pass a tuple of length two, you can specify either of the aforementioned arguments for both axes separately.
        :type bins: Union[None, int, tuple, np.ndarray], optional
        :param name: The name of the heatmap in the legend and its unique identifier for later updates. If None, the heatmap does not show up in the legend and is not registered for later update.
        :type name: str, optional
        """
        self.fig_widget._add_heatmap(x, y, bins, name)

    def add_vmarker(self, marker_pos: Union[float, List[float]], y_int: Tuple[float], name: str = None):
        """
        
        """
        self.fig_widget._add_vmarker(marker_pos, y_int, name)

    def update_line(self, name: str, x: List[float], y: List[float]):
        """
        Update the line called `name` with data `x` and `y`.
        See `func:add_line` for an explanation of the arguments.
        """
        self.fig_widget._update_line(name, x, y)

    def update_scatter(self, name: str, x: List[float], y: List[float]):
        """
        Update the scatter called `name` with data `x` and `y`.
        See `func:add_scatter` for an explanation of the arguments.
        """
        self.fig_widget._update_scatter(name, x, y)

    def update_histogram(self, name: str, bins: Union[int, tuple, list], data: List[float]):
        """
        Update the histogram called `name` with data `data` and bins `bins`.
        See `func:add_histogram` for an explanation of the arguments.
        """
        self.fig_widget._update_histogram(name, bins, data)

    def update_heatmap(self, name: str, x: List[float], y: List[float], bins: Union[int, tuple, list]):
        """
        Update the heatmap called `name` with data `x`, `y` and bins `bins`.
        See `func:add_heatmap` for an explanation of the arguments.
        """
        self.fig_widget._update_histogram(name, x, y, bins)

    def update_vmarker(self, name: str, marker_pos: Union[float, List[float]], y_int: Tuple[float]):
        """
        
        """
        self.fig_widget._update_vmarker(name, marker_pos, y_int)

    def get_figure(self):
        """
        Returns the figure object of the plot. Can be used to further manipulate the plot.
        """
        return self.fig_widget._get_figure()
    
    def get_artist(self, name: str):
        """
        Returns the artist object (e.g. plotly Scatter or matplotlib Line) of the plot. Can be used to further manipulate the plot. References on available functions can be found online, e.g. matplotlib lines (https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D) and plotly scatters (https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html).

        **Example:**

        .. code-block:: python

            import cait.versatile as vai

            # PLOTLY: 
            plot = vai.Viewer({"line": {"line1": [[1, 2, 3, 4], [1, 4, 9, 16]]}}, backend="plotly")
            plot.get_artist("line1").line.dash = "dash"

            # MATPLOTLIB:
            plot = vai.Viewer({"line": {"line1": [[1, 2, 3, 4], [1, 4, 9, 16]]}}, backend="mpl")
            plot.get_artist("line1").set_linestyle("--")
            plot.update() # required for matplotlib backend

            # works analogously for vai.Line({"line1": [[1, 2, 3, 4], [1, 4, 9, 16]]})
            # and other higher-level plotting functions
        """
        return self.fig_widget._get_artist(name=name)

    def plot(self, data: dict):
        """
        Plot data stored in dictionary.

        :param data: The data dictionary to plot (see :class:`Viewer` for details on how the dictionary needs to be structured)
        :type data: dict
        """
        for key, value in data.items():
            if key == "line":
                for line_name, line_data in value.items():
                    assert 2==len(line_data), "Line data has to be a tuple/list of length 2 containing x/y data respectively"
                    if line_name in self.fig_widget.line_names:
                        self.update_line(name=line_name, x=line_data[0], y=line_data[1])
                    else:
                        self.add_line(x=line_data[0], y=line_data[1], name=line_name)

            if key == "scatter":
                for scatter_name, scatter_data in value.items():
                    assert 2==len(scatter_data), "Scatter data has to be a tuple/list of length 2 containing x/y data respectively"
                    if scatter_name in self.fig_widget.scatter_names:
                        self.update_scatter(name=scatter_name, x=scatter_data[0], y=scatter_data[1])
                    else:
                        self.add_scatter(x=scatter_data[0], y=scatter_data[1], name=scatter_name)

            if key == "histogram":
                for histogram_name, histogram_data in value.items():
                    assert 2==len(histogram_data), "Histogram data has to be a tuple/list of length 2 containing x(bins)/y data respectively"
                    if histogram_name in self.fig_widget.histogram_names:
                        self.update_histogram(name=histogram_name, bins=histogram_data[0], data=histogram_data[1])
                    else:
                        self.add_histogram(bins=histogram_data[0], data=histogram_data[1], name=histogram_name)
            
            if key == "heatmap":
                for heatmap_name, heatmap_data in value.items():
                    assert 3==len(heatmap_data), "Heatmap data has to be a tuple/list of length 3 containing bins/x/y data respectively"
                    if heatmap_name in self.fig_widget.heatmap_names:
                        self.update_heatmap(name=heatmap_name, bins=heatmap_data[0], x=heatmap_data[1], y=heatmap_data[2])
                    else:
                        self.add_heatmap(bins=heatmap_data[0], x=heatmap_data[1], y=heatmap_data[2], name=heatmap_name)

            if key == "axes":
                self.fig_widget._set_axes(value)

        self.update()

    def show(self):
        """
        Show the plot in Jupyter.
        """
        self.fig_widget._show()
        self.visible = True

    def update(self):
        self.fig_widget._update()
    
    def close(self, b=None):
        """
        Hide the plot in Jupyter.
        """
        if self.visible: 
            self.fig_widget._close()
            self.visible = False