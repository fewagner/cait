import numpy as np
from ipywidgets import widgets
from typing import List, Union, Callable, Iterable

from IPython.display import display

from .stream import Stream
from ._baseClasses import BaseClassPlotly, BaseClassMPL, FncBaseClass

##### HELPER FUNCTIONS AND CLASSES #####
class PreviewEvent(FncBaseClass):
    """
    Helper class to use :class:`Preview` also to display iterables of events.

    >>> # Multiple channels
    >>> it = dh.get_event_iterator("events")
    >>> Preview(it)

    >>> # Single channel
    >>> it = dh.get_event_iterator("events", channel=0)
    >>> Preview(it)

    >>> # Using batches; this works (and shows events in batches
    >>> # as different channels) but should be avoided
    >>> it = dh.get_event_iterator("events", channel=0, batch_size=10)
    >>> Preview(it)
    """
    def __call__(self, event):
        return None
    def preview(self, event):
        if event.ndim > 1:
            lines = {f'channel {k}': [None, ev] for k, ev in enumerate(event)}
        else:
            lines = {'channel 0': [None, event]}

        return dict(line=lines, 
                    axes={"xaxis": {"label": "data index"},
                          "yaxis": {"label": "data (V)"}
                         })

##### PLOT ROUTINES #####
class Viewer():
    """Class for plotting data given a dictionary of instructions (see below).

    :param data: Data dictionary containing line/scatter/axes information (see below), defaults to None
    :type data: dict, optional
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl'], i.e. plotly or matplotlib, defaults to 'plotly'
    :type backend: str, optional
    :param show_controls: Set to True if plot controls should be shown. For the Viewer alone, this is just an "Exit" button which closes the plot, but inherited objects can add more buttons with arbitrary functionality. Defaults to False
    :type show_controls: bool, optional

    `Keyword Arguments` are passed to class:`BaseClassPlotly` or class:`BaseClassMPL`, depending on the chosen `backend` and can be either of the following:
    :param template: Valid backend theme. E.g. for `plotly` either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], defaults to 'ggplot2'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: float, optional
    :param width: Figure width, defaults to 700
    :type width: float, optional

    Convention for `data` Dictionary:
    ```
    { line: { line1: [x_data1, y_data1],
               line2: [x_data2, y_data2]
              },
      scatter: { scatter1: [x_data1, y_data1],
                 scatter2: [x_data2, y_data2]
                },
      histogram: { hist11: [bin_data1, hist_data1],
                   hist2: [bin_data2, hist_data2]
                },
      heatmap: { heat1: [(xbin_data1, ybin_data1), (hist_xdata1, hist_ydata1)],
                 heat2: [(xbin_data2, ybin_data2), (hist_xdata2, hist_ydata2)]
                },
      axes: { xaxis: { label: "xlabel",
                       scale: "linear"
                      },
              yaxis: { label: "ylabel",
                       scale: "log"
                      }
            }
      }
    ```
    """
    def __init__(self, data=None, backend="plotly", show_controls=False, **kwargs):
        if backend=="plotly":
            self.fig_widget = BaseClassPlotly(**kwargs)
        elif backend=="mpl":
            self.fig_widget = BaseClassMPL(**kwargs)
        else:
            raise NotImplementedError('Only backend "plotly" and "mpl" are supported.')
        
        self.show_controls = show_controls

        self.buttons = list()

        self._button_exit = widgets.Button(description="Exit", tooltip="Close plot widget.")
        self._button_exit.on_click(self.close)   
        self.buttons.append(self._button_exit)   

        self.visible = False

        if data is not None: 
            self.plot(data)  
            self.show()

    def _init_UI(self):
        if self.show_controls:
            self.UI = widgets.VBox([widgets.HBox(self.buttons), self.fig_widget.fig])
        else:
            self.UI = self.fig_widget.fig

    def show_legend(self, show: bool = True):
        """
        Show/hide legend.

        :param show: If True, legend is shown. If False, legend is hidden.
        :type show: bool
        """
        self.fig_widget._show_legend(show)

    def set_xlabel(self, xlabel: str):
        """
        Set the x-label of the figure.

        :param xlabel: x-label
        :type xlabel: str
        """
        self.fig_widget._set_axes(dict(xaxis={"label":xlabel}))

    def set_ylabel(self, ylabel: str):
        """
        Set the y-label of the figure.

        :param ylabel: y-label
        :type ylabel: str
        """
        self.fig_widget._set_axes(dict(yaxis={"label":ylabel}))

    def set_xscale(self, xscale: str):
        """
        Set the x-scale of the figure. Either linear or logarithmic.

        :param xscale: x-scale. Either of ["linear", "log"]
        :type xscale: str
        """
        self.fig_widget._set_axes(dict(xaxis={"scale":xscale}))

    def set_yscale(self, yscale: str):
        """
        Set the y-scale of the figure. Either linear or logarithmic.

        :param yscale: y-scale. Either of ["linear", "log"]
        :type yscale: str
        """
        self.fig_widget._set_axes(dict(yaxis={"scale":yscale}))

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

    def add_histogram(self, bins: Union[int, tuple], data: List[float], name: str = None):
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

    def update_histogram(self, name: str, bins: Union[int, tuple], data: List[float]):
        """
        Update the histogram called `name` with data `data` and bins `bins`.
        See `func:add_histogram` for an explanation of the arguments.
        """
        self.fig_widget._update_histogram(name, bins, data)

    def get_figure(self):
        """
        Returns the figure object of the plot. Can be used to further manipulate the plot.
        """
        return self.fig_widget.fig

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
                        self.fig_widget._update_line(name=line_name, x=line_data[0], y=line_data[1])
                    else:
                        self.fig_widget._add_line(x=line_data[0], y=line_data[1], name=line_name)

            if key == "scatter":
                for scatter_name, scatter_data in value.items():
                    assert 2==len(scatter_data), "Scatter data has to be a tuple/list of length 2 containing x/y data respectively"
                    if scatter_name in self.fig_widget.scatter_names:
                        self.fig_widget._update_scatter(name=scatter_name, x=scatter_data[0], y=scatter_data[1])
                    else:
                        self.fig_widget._add_scatter(x=scatter_data[0], y=scatter_data[1], name=scatter_name)

            if key == "histogram":
                for histogram_name, histogram_data in value.items():
                    assert 2==len(histogram_data), "Histogram data has to be a tuple/list of length 2 containing x(bins)/y data respectively"
                    if histogram_name in self.fig_widget.histogram_names:
                        self.fig_widget._update_histogram(name=histogram_name, bins=histogram_data[0], data=histogram_data[1])
                    else:
                        self.fig_widget._add_histogram(bins=histogram_data[0], data=histogram_data[1], name=histogram_name)

            if key == "axes":
                self.fig_widget._set_axes(value)

    def show(self):
        """
        Show the plot in Jupyter.
        """
        self._init_UI()
        display(self.UI)
        self.visible = True
    
    def close(self, b=None):
        """
        Hide the plot in Jupyter.
        """
        if self.visible: 
            self.UI.close()
            self.visible = False

class Line(Viewer):
    """
    Plot a line graph. 

    :param y: The y-data to plot. You can either hand a list-like object (simple plotting of one line) or a dictionary whose keys are line names and whose values are again list-like objects (for each key a line is plotted and a legend entry is created).
    :type y: Union[List[float], dict]
    :param x: The x-data to plot. If None is specified, the y-data is plotted over the data index. Defaults to None.
    :type x: List[float], optional
    :param xlabel: x-label for the plot.
    :type xlabel: str, optional
    :param ylabel: y-label for the plot.
    :type ylabel: str, optional
    :param xscale: x-scale for the plot. Either of ['linear', 'log'], defaults to 'linear'.
    :type xscale: str, optional
    :param yscale: y-scale for the plot. Either of ['linear', 'log'], defaults to 'linear'.
    :type yscale: str, optional
    :param kwargs: Keyword arguments (see below)
    :type kwargs: Any

    `Keyword Arguments` are passed to class:`InspectBaseClass` and can be either of the following:
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl'], i.e. plotly or matplotlib, defaults to 'plotly'.
    :type backend: str, optional
    :param template: Valid backend theme. E.g. for `plotly` backend either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], defaults to 'ggplot2'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: float, optional
    :param width: Figure width, defaults to 700
    :type width: float, optional
    """
    def __init__(self, y: Union[List[float], dict], x: List[float] = None, xlabel: str = None, ylabel: str = None, xscale: str = 'linear', yscale: str = 'linear', **kwargs):

        super().__init__(**kwargs)

        if isinstance(y, dict):
            for key, value in y.items():
                self.add_line(x=x, y=value, name=key)
        else:
            self.add_line(x=x, y=y)

        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)
        self.set_xscale(xscale)
        self.set_yscale(yscale)

        self.show()

class Scatter(Viewer):
    """
    Plot a scatter graph. 

    :param y: The y-data to plot. You can either hand a list-like object (simple plotting of one line) or a dictionary whose keys are line names and whose values are again list-like objects (for each key a line is plotted and a legend entry is created).
    :type y: Union[List[float], dict]
    :param x: The x-data to plot. If None is specified, the y-data is plotted over the data index. Defaults to None.
    :type x: List[float], optional
    :param xlabel: x-label for the plot.
    :type xlabel: str, optional
    :param ylabel: y-label for the plot.
    :type ylabel: str, optional
    :param xscale: x-scale for the plot. Either of ['linear', 'log'], defaults to 'linear'.
    :type xscale: str, optional
    :param yscale: y-scale for the plot. Either of ['linear', 'log'], defaults to 'linear'.
    :type yscale: str, optional
    :param kwargs: Keyword arguments (see below)
    :type kwargs: Any

    `Keyword Arguments` are passed to class:`InspectBaseClass` and can be either of the following:
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl'], i.e. plotly or matplotlib, defaults to 'plotly'.
    :type backend: str, optional
    :param template: Valid backend theme. E.g. for `plotly` backend either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], defaults to 'ggplot2'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: float, optional
    :param width: Figure width, defaults to 700
    :type width: float, optional
    """
    def __init__(self, y: Union[List[float], dict], x: List[float] = None, xlabel: str = None, ylabel: str = None, xscale: str = 'linear', yscale: str = 'linear', **kwargs):

        super().__init__(**kwargs)

        if isinstance(y, dict):
            for key, value in y.items():
                self.add_scatter(x=x, y=value, name=key)
        else:
            self.add_scatter(x=x, y=y)

        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)
        self.set_xscale(xscale)
        self.set_yscale(yscale)

        self.show()

class Histogram(Viewer):
    """
    Plot a Histogram. 

    :param data: The data to bin and plot. You can either hand a list-like object (simple plotting of one histogram) or a dictionary whose keys are histogram names and whose values are again list-like objects (for each key a histogram is binned, plotted and a legend entry is created).
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
    :param kwargs: Keyword arguments (see below)
    :type kwargs: Any

    `Keyword Arguments` are passed to class:`InspectBaseClass` and can be either of the following:
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl'], i.e. plotly or matplotlib, defaults to 'plotly'.
    :type backend: str, optional
    :param template: Valid backend theme. E.g. for `plotly` backend either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], defaults to 'ggplot2'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: float, optional
    :param width: Figure width, defaults to 700
    :type width: float, optional
    """
    def __init__(self, data: Union[List[float], dict], bins: Union[tuple, int] = None, xlabel: str = None, ylabel: str = None, xscale: str = 'linear', yscale: str = 'linear', **kwargs):

        super().__init__(**kwargs)

        if isinstance(data, dict):
            for key, value in data.items():
                self.add_histogram(bins=bins, data=value, name=key)
        else:
            self.add_histogram(bins=bins, data=data)

        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)
        self.set_xscale(xscale)
        self.set_yscale(yscale)

        self.show()

# Has no test case (yet)
class StreamViewer(Viewer):
    """
    Class to view stream data, i.e. for example the contents of a binary file as produced by vdaq2.

    :param hardware: The hardware that was used to record the file. Valid options are ['vdaq2']
    :type hardware: str
    :param fname: Stream file (full path including file extension)
    :type fname: str
    :param n_points: The number of data points that should be simultaneously displayed in the stream viewer. A large number can impact performance. Note that the number of points that are displayed are irrelevant of the downsampling factor (see below), i.e. the viewer will always display n_points points.
    :type n_points: int, optional
    :param downsample_factor: This many samples are skipped in the data when plotting it. A higher number increases performance but lowers the level of detail in the data.
    :type downsample_factor: int, optional
    :param kwargs: Keyword arguments (see below)
    :type kwargs: Any

    `Keyword Arguments` are passed to class:`InspectBaseClass` and can be either of the following:
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl'], i.e. plotly or matplotlib, defaults to 'plotly'.
    :type backend: str, optional
    :param template: Valid backend theme. E.g. for `plotly` backend either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], defaults to 'ggplot2'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: float, optional
    :param width: Figure width, defaults to 700
    :type width: float, optional
    """
    def __init__(self, hardware: str, file: str, n_points: int = 10000, downsample_factor: int = 100, **kwargs):
        super().__init__(data=None, show_controls=True, **kwargs)

        # Adding buttons for navigating back and forth in the stream
        self._button_move_left = widgets.Button(description="←", tooltip="Move backwards in time.")
        self._button_move_right = widgets.Button(description="→", tooltip="Move forward in time.")
        self._info_button = widgets.Button(description="Data Info", tooltip="Display mean/std/min/max/diff for all (visible) data on screen. If you want the std of a single trace for example, zoom in such that only the data you want to consider is cropped and hit this button.")

        self._button_move_left.on_click(self._move_left)
        self._button_move_right.on_click(self._move_right)
        self._info_button.on_click(self._get_info)

        self.buttons.append(self._button_move_left)
        self.buttons.append(self._button_move_right) 
        self.buttons.append(self._info_button) 

        # Output for printing data info
        self._output = widgets.Output()
        self.buttons.append(self._output)

        # Linking the stream-file (note that the file extension is handled by the StreamFile object as
        # it could be different for different hardware)
        self.stream = Stream(src=file, hardware=hardware)

        # Adding lines for all the channels present in the hardware file
        for name in self.stream.keys:
            self.add_line(x=None, y=None, name=name)

        # Adding labels
        self.set_xlabel("time")
        self.set_ylabel("trace (V)")

        # Initializing plot
        self.n_points = n_points
        self.current_start = 0
        self.downsample_factor = downsample_factor

        self.update()
        self.show()

    def update(self):
        # Create slice for data access
        where = slice(self.current_start, self.current_start + self.n_points*self.downsample_factor, self.downsample_factor)
        
        # Time array is the same for all channels
        t = self.stream.time[where]
        # Convert to datetime
        t_datetime = self.stream.time.timestamp_to_datetime(t)
        
        for name in self.stream.keys:
            self.update_line(name=name, x=t_datetime, y=self.stream[name, where, "as_voltage"])

    def _move_right(self, b):
        # ATTENTION: should be restricted to file size at some point (and the end point should be provided by stream)
        self.current_start += int(self.n_points*self.downsample_factor/2)
        self.update()

    def _move_left(self, b):
        self.current_start = max(0, self.current_start - int(self.n_points*self.downsample_factor/2))
        self.update()

    def _get_info(self, b):
        xmin, xmax = self.get_figure().layout.xaxis.range
        if type(xmin) is str: # then we assume it to be datetime
            xmin = np.datetime64(xmin)
            xmax = np.datetime64(xmax)

        ymin, ymax = self.get_figure().layout.yaxis.range

        y_vals = list()
        for trace in self.get_figure().select_traces():
            if trace.visible is True:
                x = trace.x
                y = trace.y
                x_mask = np.logical_and(x > xmin, x < xmax)
                y_mask = np.logical_and(y > ymin, y < ymax)
                y_vals.append(y[np.logical_and(x_mask, y_mask)])

        y_vals = np.concatenate(y_vals, axis=0)

        if len(y_vals) == 0:
            out_str = f"mean_y:  None, min_y: None, delta_y: None\nsigma_y: None, max_y: None"
        else:
            dat_min, dat_max = np.min(y_vals), np.max(y_vals)
            dat_diff = dat_max - dat_min
            dat_mean, dat_std = np.mean(y_vals), np.std(y_vals)
            out_str = f"mean_y: {dat_mean:6.3f}, min_y: {dat_min:6.3f}, delta_y: {dat_diff:5.3f}\nsigma_y: {dat_std:5.3f}, max_y: {dat_max:6.3f}"
        
        with self._output:
            self._output.clear_output()
            print(out_str)
            
# Has no test case (yet)
class Preview(Viewer):
    """
    Class for inspecting the behavior of functions which were subclassed from :class:`._baseClasses.FncBase`.
    Can also be used to display single events.

    :param events: An iterable of events. Can be e.g. :class:`EventIterator`, a 2d :class:`numpy.ndarray` or a list of List[float].
    :type events: Iterable
    :param f: The function to be inspected, already initialized with the values that should stay fixed throughout the inspection. Default None (which means that just the events of the iterable will be displayed)
    :type f: :class:`._baseClasses.FncBaseClass`
    :param kwargs: Keyword arguments (see below)
    :type kwargs: Any

    `Keyword Arguments` are passed to class:`Viewer` and class:`InspectBaseClass` and can be either of the following:
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl'], i.e. plotly or matplotlib, defaults to 'plotly'
    :type backend: str, optional
    :param template: Valid backend theme. E.g. for `plotly` backend either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], defaults to 'ggplot2'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: float, optional
    :param width: Figure width, defaults to 700
    :type width: float, optional
    """
    def __init__(self, events: Iterable, f: Callable = None, **kwargs):
        #viewer_kwargs = {k:v for k,v in kwargs.items() if k in ["backend","template","width","height"]}
        #for k in ["backend","template","width","height"]: kwargs.pop(k, None)
        super().__init__(data=None, show_controls=True, **kwargs)

        self.f = f if f is not None else PreviewEvent()
        self.events = iter(events)

        self._button_next = widgets.Button(description="Next", tooltip="Show next event.")
        self._button_next.on_click(self._update_plot)
        self.buttons.append(self._button_next)
        
        self.start()
    
    def _update_plot(self, b=None):
        try: 
            self.plot(self.f.preview(next(self.events)))
        except StopIteration: 
            self.close()
        except:
            self.close()
            raise
    
    def start(self):
        """
        Show the plot and start iterating over the events.
        """
        self._update_plot()
        self.show()
 
##### EXPERIMENTAL START ######    
class Heatmap(Viewer): # not yet finished (TODO)
    """
    Plot a Heatmap. (DO NOT USE, UNFINISHED)

    :param xdata: The xdata to bin and plot.
    :type xdata: List[float]
    :param ydata: The ydata to bin and plot.
    :type ydata: List[float]
    :param xbins: The x-binning data to use. If None, the binning is done automatically. An integer is interpreted as the desired total number of bins. You can also parse a tuple of the form `(start, end, nbins)` to bin the data between `start` and `end` into a total of `nbins` bins
    :type xbins: Union[None, int, tuple], optional
    :param ybins: The y-binning data to use. If None, the binning is done automatically. An integer is interpreted as the desired total number of bins. You can also parse a tuple of the form `(start, end, nbins)` to bin the data between `start` and `end` into a total of `nbins` bins
    :type ybins: Union[None, int, tuple], optional
    :param xlabel: x-label for the heatmap.
    :type xlabel: str, optional
    :param ylabel: y-label for the heatmap.
    :type ylabel: str, optional
    :param zscale: z-scale for the heatmap. Either of ['linear', 'log'], defaults to 'linear'.
    :type zscale: str, optional
    :param kwargs: Keyword arguments (see below)
    :type kwargs: Any

    `Keyword Arguments` are passed to class:`InspectBaseClass` and can be either of the following:
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl'], i.e. plotly or matplotlib, defaults to 'plotly'.
    :type backend: str, optional
    :param template: Valid backend theme. E.g. for `plotly` backend either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], defaults to 'ggplot2'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: float, optional
    :param width: Figure width, defaults to 700
    :type width: float, optional
    """
    def __init__(self, xdata: List[float], ydata: List[float], xbins: Union[tuple, int] = None, ybins: Union[tuple, int] = None, xlabel: str = None, ylabel: str = None, zscale: str = 'linear', **kwargs):


        heatmap = dict()
        heatmap["density"] = [(xbins, ybins), (xdata, ydata)]

        plot_data = dict(
            heatmap = heatmap,
            axes = { "xaxis": { "label": xlabel}, "yaxis": { "label": ylabel} }
        )

        super().__init__(data=plot_data, **kwargs)