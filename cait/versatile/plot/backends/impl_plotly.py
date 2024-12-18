from typing import Callable
#from itertools import cycle

import numpy as np
from ipywidgets import widgets
from IPython.display import display
import plotly.graph_objects as go
#import plotly.express as px

from .backendbase import BackendBaseClass
from .helper import EmptyRep

class BaseClassPlotly(BackendBaseClass):
    """
    Base Class for plots using the `plotly` library. Not meant for standalone use but rather to be called through :class:`Viewer`. 

    This class produces plots given a dictionary of instructions of the following form:
    
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
    
    Line/scatter plots are created for each key of the line/scatter dictionaries. The respective values have to be tuples/lists of length 2 including x and y data.
    The axes dictionary (as well as 'label' and 'scale') are optional and only intended to be used in case one wants to put axes labels or change to a log scale.

    :param template: Valid plotly theme. Either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], defaults to 'seaborn'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: int, optional
    :param width: Figure width, defaults to 700
    :type width: int, optional
    """
    def __init__(self, 
                 template: str = "seaborn", 
                 height: int = 500, 
                 width: int = 700, 
                 show_controls: bool = False):
     
        self._line_names = list()
        self._scatter_names = list()
        self._histogram_names = list()
        self._heatmap_names = list()
        
        self._color_log = False

        self.show_controls = show_controls
        self.buttons_initialized = False
    
        self._init_fig(template, height, width)

    def _init_UI(self):
        if self.show_controls:
            if not self.buttons_initialized:
                self._output = widgets.Output()
                self._add_button("Exit", self._close, "Close plot widget.", 0)
                self._add_button("Data Info", self._get_info, "Display mean/std/min/max/diff for all (visible) data on screen. If you want the std of a single trace for example, zoom in such that only the data you want to consider is cropped and hit this button.")
                self.buttons_initialized = True
            self.UI = widgets.VBox([widgets.HBox(self.buttons + [self._output]), self.fig])
        else:
            self.UI = widgets.VBox([self.fig])

    def _init_fig (self, template, height, width):
        self.fig = go.FigureWidget()
        self.fig.update_layout(
                        autosize = True,
                        xaxis = dict(
                            zeroline = False,
                            showgrid = True
                        ),
                        yaxis = dict(
                            zeroline = False,
                            showgrid = True
                        ),
                        height = height,
                        width = width,
                        legend = dict(x=0.99,
                                      y=0.95,
                                      xanchor="right",
                                      yanchor="top",
                                     ),
                        margin = dict(l=0, r=0, t=40, b=80),
                        xaxis_hoverformat ='0.3g',
                        yaxis_hoverformat ='0.3g',
                        hovermode = "x unified",
                        showlegend = True,
                        template = template,
                        bargap=0,
                        barmode='overlay'
                        #modebar_add = ['drawline', 'eraseshape'],
                        #newshape=dict(line_color='yellow',
                        #        fillcolor='turquoise',
                        #        opacity=0.5)
                    )
        
    def _add_button(self, text: str, callback: Callable, tooltip: str = None, where: int = -1, key: int = None):
        # This gives inherited classes the possibility to add buttons themselves before 
        # calling super().__init__
        if not hasattr(self, "buttons"): self.buttons = []

        if where == -1: self.buttons.append(widgets.Button(description=text, tooltip=tooltip))
        else: self.buttons.insert(where, widgets.Button(description=text, tooltip=tooltip))

        self.buttons[where].on_click(callback) 
        
    def _show_legend(self, show=True):
        self.fig.update_layout(showlegend = show)

    def _add_line(self, x, y, name=None):
        self.fig.add_trace(go.Scatter(x=x,
                                      y=y,
                                      name=name,
                                      mode="lines",
                                      line={#'color': next(self.colors),
                                            'width': 3},
                                      showlegend= True if name is not None else False) 
                                      )
        if name is not None: self._line_names.append(name)

    def _add_scatter(self, x, y, name=None):
        self.fig.add_trace(go.Scatter(x=x,
                                      y=y,
                                      name=name,
                                      mode="markers",
                                      line={#'color': next(self.colors),
                                            'width': 3},
                                      showlegend= True if name is not None else False) 
                                      )
        if name is not None: self._scatter_names.append(name)

    def _add_histogram(self, bins, data, name=None):
        if bins is None:
            arg = dict()
        elif isinstance(bins, int):
            arg = dict(nbinsx=bins)
        elif isinstance(bins, tuple) and len(bins) == 3:
            arg = dict(xbins=dict(start=bins[0], end=bins[1], size=(bins[1]-bins[0])/bins[2]) )
        elif isinstance(bins, (list, np.ndarray)):
            bins = np.array(bins)
            if not np.all(np.diff(np.unique(np.diff(bins))) < 1e-10):
                raise ValueError("If bin edges are provided as a list/numpy array, the spacing has to be uniform and increasing for backend 'plotly'.")
            arg = dict(xbins=dict(start=bins[0], end=bins[-1], size=np.unique(np.diff(bins))[0]) )
        else:
            raise TypeError("Bin info has to be either None, an integer (number of bins), or a tuple of length 3 (start, end, number of bins)")
        
        self.fig.add_trace(go.Histogram(name=name, 
                                        x=data,
                                        **arg,
                                        showlegend = True if name is not None else False)
                            )
        if name is not None: self._histogram_names.append(name)

        # lower opacity of histograms if more than one is plotted
        if len([k for k in self.fig.select_traces(selector="histogram")]) > 1:
            self.fig.update_traces(selector="histogram", patch=dict(opacity=0.8))

    def _add_heatmap(self, x, y, bins, name=None):
        # use numpy default bins
        if bins is None:
            arg = dict()
        # use single integer for number of bins on both axes
        elif isinstance(bins, int):
            arg = dict(bins=[bins, bins])
        # use tuple of length 3 as linspace for both axes
        elif isinstance(bins, tuple) and len(bins) == 3:
            arg = dict(bins=[np.linspace(*bins), np.linspace(*bins)])
        # use tuple of length two with default numpy behaviour
        elif (isinstance(bins, tuple) 
              and len(bins) == 2 
              and all([isinstance(x, (int, list, np.ndarray)) for x in bins])):
            arg = dict(bins=bins)
        # use tuple of length two, which contains tuples of length 3,
        # as start/end/N to create bins from linspace
        elif (isinstance(bins, tuple) 
              and len(bins) == 2
              and all([isinstance(x, tuple) for x in bins])
              and all([len(x)==3 for x in bins])):
            arg = dict(bins=[np.linspace(*bins[0]), np.linspace(*bins[1])])
        # use single np.array or list as bins for both axes
        elif isinstance(bins, (list, np.ndarray)):
            arg = dict(bins=[np.array(bins), np.array(bins)])
        else:
            raise TypeError("Bin info has to be either None, an integer (number of bins), a tuple of length 3 (start, end, number of bins), or a numpy array of bin edges. To pass information for both axes separately, use tuples of length 2 whose elements are integers, tuples, numpy arrays, as mentioned before.")
        
        counts, x_edges, y_edges = np.histogram2d(x, y, **arg)
        x_centers = (x_edges[:-1] + x_edges[1:])/2
        y_centers = (y_edges[:-1] + y_edges[1:])/2

        counts_new = counts.T.copy()
        mask = counts_new == 0

        if self._color_log:
            z = np.log10(counts_new, where=~mask)
        else:
            z = counts_new

        # json doesn't support np.nan which is why we have to convert to object dtype
        z = np.asarray(z, dtype=object)
        z[mask] = None

        hm = go.Heatmap()
        hm.update({"x": x_centers, 
                   "y": y_centers, 
                   "z": z, 
                   "customdata": counts_new, 
                   "name": name,
                   "showlegend": name is not None,
                   "hovertemplate": '(%{x}, %{y})<br>counts: %{customdata}',
                   "coloraxis": "coloraxis"})
        self.fig.add_trace(hm)

        if name is not None: self._heatmap_names.append(name)

    def _add_vmarker(self, marker_pos, y_int, name=None):
        if marker_pos is None or y_int is None: 
            x, y = None, None
        else:
            # Stack three of those arrays on top of each other and flatten in column-major order
            # This results in an array where each ts_datetime point comes three times in a row
            x = np.array([marker_pos, marker_pos, marker_pos]).flatten(order="F")

            # y-values will jump between y_min and y_max and are separated by a None
            y = [np.min(y_int), np.max(y_int), None]*np.array(marker_pos).shape[-1]

        self.fig.add_trace(go.Scatter(x=x,
                                      y=y,
                                      name=name,
                                      mode="lines",
                                      line={#'color': next(self.colors),
                                            'width': 1.5,
                                            'dash': 'dash'},
                                      showlegend= True if name is not None else False) 
                                      )
        if name is not None: self._line_names.append(name)

    def _update_line(self, name, x, y):
        self.fig.update_traces(dict(x=x, y=y), selector=dict(name=name))

    def _update_scatter(self, name, x, y):
        self.fig.update_traces(dict(x=x, y=y), selector=dict(name=name))

    def _update_histogram(self, name, bins, data):
        #n_histograms = len([k for k in self.fig.select_traces(selector="histogram")])
        opacity = 0.8 if len(self._histogram_names)>1 else 1

        if bins is None:
            arg = dict()
        elif isinstance(bins, int):
            arg = dict(nbinsx=bins)
        elif isinstance(bins, tuple) and len(bins) == 3:
            arg = dict(xbins=dict(start=bins[0], end=bins[1], size=(bins[1]-bins[0])/bins[2]) )
        elif isinstance(bins, (list, np.ndarray)):
            bins = np.array(bins)
            if not np.all(np.diff(np.unique(np.diff(bins))) < 1e-10):
                raise ValueError("If bin edges are provided as a list/numpy array, the spacing has to be uniform and increasing for backend 'plotly'.")
            arg = dict(xbins=dict(start=bins[0], end=bins[-1], size=np.unique(np.diff(bins))[0]) )
        else:
            raise TypeError("Bin info has to be either None, an integer (number of bins), or a tuple of length 3 (start, end, number of bins)")
        
        self.fig.update_traces(dict(x=data, opacity=opacity, **arg), selector=dict(name=name))

    def _update_heatmap(self, name, x, y, bins):
        # use numpy default bins
        if bins is None:
            arg = dict()
        # use single integer for number of bins on both axes
        elif isinstance(bins, int):
            arg = dict(bins=[bins, bins])
        # use tuple of length 3 as linspace for both axes
        elif isinstance(bins, tuple) and len(bins) == 3:
            arg = dict(bins=[np.linspace(*bins), np.linspace(*bins)])
        # use tuple of length two with default numpy behaviour
        elif (isinstance(bins, tuple) 
              and len(bins) == 2 
              and all([isinstance(x, (int, list, np.ndarray)) for x in bins])):
            arg = dict(bins=bins)
        # use tuple of length two, which contains tuples of length 3,
        # as start/end/N to create bins from linspace
        elif (isinstance(bins, tuple) 
              and len(bins) == 2
              and all([isinstance(x, tuple) for x in bins])
              and all([len(x)==3 for x in bins])):
            arg = dict(bins=[np.linspace(*bins[0]), np.linspace(*bins[1])])
        # use single np.array or list as bins for both axes
        elif isinstance(bins, (list, np.ndarray)):
            arg = dict(bins=[np.array(bins), np.array(bins)])
        else:
            raise TypeError("Bin info has to be either None, an integer (number of bins), a tuple of length 3 (start, end, number of bins), or a numpy array of bin edges. To pass information for both axes separately, use tuples of length 2 whose elements are integers, tuples, numpy arrays, as mentioned before.")
        
        counts, x_edges, y_edges = np.histogram2d(x, y, **arg)
        x_centers = (x_edges[:-1] + x_edges[1:])/2
        y_centers = (y_edges[:-1] + y_edges[1:])/2

        counts_new = counts.T.copy()
        mask = counts_new == 0

        if self._color_log:
            z = np.log10(counts_new, where=~mask)
        else:
            z = counts_new

        # json doesn't support np.nan which is why we have to convert to object dtype
        z = np.asarray(z, dtype=object)
        z[mask] = None

        self.fig.update_traces({ "x": x_centers, "y": y_centers, "z": z, "customdata": counts_new}, 
                               selector=dict(name=name))

    def _update_vmarker(self, name, marker_pos, y_int):
        if marker_pos is None or y_int is None: 
            x, y = None, None
        else:
            # Stack three of those arrays on top of each other and flatten in column-major order
            # This results in an array where each ts_datetime point comes three times in a row
            x = np.array([marker_pos, marker_pos, marker_pos]).flatten(order="F")

            # y-values will jump between y_min and y_max and are separated by a None
            y = [np.min(y_int), np.max(y_int), None]*np.array(marker_pos).shape[-1]

        self.fig.update_traces(dict(x=x, y=y), selector=dict(name=name))
    
    def _get_artist(self, name: str):
        return list(self.fig.select_traces(selector=dict(name=name)))[0]

    def _set_axes(self, data):
        if "xaxis" in data.keys():
            if "label" in data["xaxis"].keys():
                self.fig.layout.xaxis.title = data["xaxis"]["label"]
            if "scale" in data["xaxis"].keys():
                self.fig.layout.xaxis.type = data["xaxis"]["scale"]
            if "range" in data["xaxis"].keys():
                self.fig.update_xaxes(range=data["xaxis"]["range"])

        if "yaxis" in data.keys():
            if "label" in data["yaxis"].keys():
                self.fig.layout.yaxis.title = data["yaxis"]["label"]
            if "scale" in data["yaxis"].keys():
                self.fig.layout.yaxis.type = data["yaxis"]["scale"]
            if "range" in data["yaxis"].keys():
                self.fig.update_yaxes(range=data["yaxis"]["range"])

        if "caxis" in data.keys():
            if "label" in data["caxis"].keys():
                self.fig.update_coloraxes(
                    showscale=True,
                    colorbar_title_text=data["caxis"]["label"],
                    colorbar_title_side="right",
                )
            if "scale" in data["caxis"].keys():
                if data["caxis"]["scale"] == "log":
                    self._color_log = True
                    self.fig.update_coloraxes(
                        showscale=True,
                        colorbar_tickprefix="1e",
                        colorbar_tickformat=",d",
                        colorbar_dtick=1
                    )
                else:
                    self._color_log = False
                    self.fig.update_coloraxes(
                        showscale=True,
                        colorbar_tickprefix="",
                        colorbar_tickformat="",
                        colorbar_dtick=None
                    )
                
            if "range" in data["caxis"].keys():
                self.fig.update_coloraxes(
                    showscale=True,
                    cmin=data["caxis"]["range"][0],
                    cmax=data["caxis"]["range"][1]
                )
            if "cmap" in data["caxis"].keys():
                if data["caxis"]["cmap"] is not None:
                    self.fig.update_coloraxes(
                        showscale=True,
                        colorscale=data["caxis"]["cmap"],
                    )

    def _get_info(self, b):
        xmin, xmax = self.fig.layout.xaxis.range

        if type(xmin) is str: # then we assume it to be datetime
            xmin = np.datetime64(xmin)
            xmax = np.datetime64(xmax)

        ymin, ymax = self.fig.layout.yaxis.range

        y_vals = list()
        for trace in self.fig.select_traces():
            if trace.visible is True:
                y = np.array(trace.y, dtype=float)
                x = np.array(trace.x, dtype=float) if trace.x is not None else np.arange(len(y))
                x_mask = ~np.isnan(x)
                y_mask = ~np.isnan(y)
                x_mask[x_mask] = np.logical_and(x[x_mask] > xmin, x[x_mask] < xmax)
                y_mask[y_mask] = np.logical_and(y[y_mask] > ymin, y[y_mask] < ymax)
                y_vals.append(y[np.logical_and(x_mask, y_mask)])

        if not y_vals or len(np.concatenate(y_vals, axis=0))==0:
            out_str = f"ȳ:  None, min_y: None, Δᵧ: None\nσᵧ: None, max_y: None"
        else:
            y_vals = np.concatenate(y_vals, axis=0)
            dat_min, dat_max = np.min(y_vals), np.max(y_vals)
            dat_diff = dat_max - dat_min
            dat_mean, dat_std = np.mean(y_vals), np.std(y_vals)
            out_str = f"ȳ: {dat_mean:6.3f}, yₘᵢₙ: {dat_min:6.3f}, Δᵧ: {dat_diff:5.3f}\nσᵧ: {dat_std:5.3f}, yₘₐₓ: {dat_max:6.3f}"
        
        with self._output:
            self._output.clear_output()
            print(out_str)

    def _show(self):
        self._init_UI()
        if not hasattr(self, "_display"):
            self._display = display(self.UI, display_id=True)
        else:
            self._display.update(self.UI)

    def _update(self):
        # Plotly automatically draws changes
        ...

    def _close(self, b=None):
        self._display.update(EmptyRep())

    def _get_figure(self):
        return self.fig
    
    @property
    def line_names(self):
        return self._line_names

    @property
    def scatter_names(self):
        return self._scatter_names

    @property
    def histogram_names(self):
        return self._histogram_names
    
    @property
    def heatmap_names(self):
        return self._heatmap_names