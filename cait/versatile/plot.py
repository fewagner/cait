import numpy as np
from ipywidgets import widgets
from typing import List, Union, Callable

from .stream import StreamFile
from ._baseClasses import InspectBaseClass, Viewer
   
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
        self.stream = StreamFile(file=file, hardware=hardware)

        # Adding lines for all the channels present in the hardware file
        for name in self.stream.available_channel_names:
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
        # Retrieve data from stream object. (tuple of time array and dictionary 
        # with keys=trace-names, values=voltage-trace)
        self.current_data = self.stream[self.current_start:(self.current_start + self.n_points*self.downsample_factor)]
        for trace in self.current_data[1].keys():
            self.update_line(name=trace, x=self.current_data[0][::self.downsample_factor], y=self.current_data[1][trace][::self.downsample_factor])

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

        x = self.current_data[0]
        x_mask = np.logical_and(x > xmin, x < xmax)

        y_vals = list()
        for trace in self.current_data[1].keys():
            # Check if data from file is still in plot (there's no reason to believe it is not
            # but the next check - whether it is hidden or not - is important!)
            line = list(self.get_figure().select_traces(selector=dict(name=trace)))
            if len(line) > 0 and line[0].visible is True:
                # Only include in calculation if visible (not deactivated in legend)
                y = self.current_data[1][trace]
                y_mask = np.logical_and(y > ymin, y < ymax)
                y_vals.append(y[np.logical_and(x_mask, y_mask)])

        y_vals = np.concatenate(y_vals, axis=0)

        if len(y_vals) == 0:
            out_str = f"mean_y:  None, min_y: None, delta_y: None\nsigma_y: None, max_y: None"
        else:
            dat_min, dat_max = np.min(y_vals), np.max(y_vals)
            dat_diff = dat_max - dat_min
            dat_mean, dat_std = np.mean(y_vals), np.std(y_vals)
            out_str = f"mean_y:  {dat_mean:.3f}, min_y: {dat_min:.3f}, delta_y: {dat_diff:.3f}\nsigma_y: {dat_std:.3f}, max_y: {dat_max:.3f}"
        
        with self._output:
            self._output.clear_output()
            print(out_str)
            
##### EXPERIMENTAL START ######
class Preview(InspectBaseClass):
    """
    Class for inspecting the behavior of functions which were subclassed from :class:`._baseClasses.FncBase`.

    :param fnc: The function to be inspected, already initialized with the values that should stay fixed throughout the inspection.
    :type fnc: :class:`._baseClasses.FncBase`
    :param events: An iterable of events. Can be e.g. :class:`.file.EventIterator`, a 2d :class:`numpy.ndarray` or a list of List[float].
    :type events: iterable
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
    def __init__(self, fnc: Callable, events, **kwargs):
        super().__init__(events, **kwargs)
        self.fnc = fnc
        
        self.start()
        
    def _calc_next(self, ev):
        return self.fnc.preview(ev)
 
class Heatmap(Viewer): # not yet finished (TODO)
    """
    Plot a Heatmap. 

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