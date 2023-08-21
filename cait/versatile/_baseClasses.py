import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt

import numpy as np
from ipywidgets import widgets
from IPython.display import display
from itertools import cycle
from abc import ABC, abstractmethod
from typing import List, Union

#########################
# Function base classes #
#########################

class FncBase:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cache = dict() # all variables that can be cached are stored here
        self.data = dict() # all variables which could be relevant for plotting are stored here
        
        assert hasattr(self, "isvectorizable"), "Objects inherited from FncBase need to define self.isvectorizable: bool in their __init__ function."
        assert hasattr(self, "n_outputs"), "Objects inherited from FncBase need to define self.n_outputs: int in their __init__ function."
        
    def _f(self, event, **kwargs):
        pass
        
    def _partial(self, event):
        return self._f(event, **self.kwargs)
        
    def __call__(self, event):
        if event.ndim > 1 and not self.isvectorizable:
            out = list()
            for i, ev in enumerate(event):
                out.append(self._partial(ev))      
            return out
        elif event.ndim > 1:
            return self._partial(event)
        else:
            return self._partial(event)
        
    def update_kwargs(self, **kwargs):
        for k,v in kwargs.items():
            self.kwargs[k] = v
        
    def preview(self, event):
        raise NotImplementedError(f"{self.__class__.__name__} does not support preview.")
        
class FitFncBase(FncBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def model(self, x, *pars):
        # TODO: check for number of arguments (signature alone is insufficient because of possible *par)
        return self._model_fnc(x, *pars)
        
    def _model_fnc(self, x, *pars):
        raise NotImplementedError(f"No model function defined for {self.__class__.__name__}.")
    
#####################
# Plot base classes #
#####################

class BackendBaseClass(ABC):

    @abstractmethod
    def _init_fig(*args, **kwargs):
        ...

    @abstractmethod
    def _show_legend(show: bool = True):
        ...

    @abstractmethod
    def _add_line(x: List[float], y: List[float], name: str = None):
        ...

    @abstractmethod
    def _add_scatter(x: List[float], y: List[float], name: str = None):
        ...

    @abstractmethod
    def _add_histogram(bins: Union[int, tuple], data: List[float], name: str = None):
        ...

    @abstractmethod
    def _update_line(name: str, x: List[float], y: List[float]):
        ...

    @abstractmethod
    def _update_scatter(name: str, x: List[float], y: List[float]):
        ...

    @abstractmethod
    def _update_histogram(name: str, bins: Union[int, tuple], data: List[float]):
        ...

    @abstractmethod
    def _set_axes(data: dict):
        ...

class BaseClassPlotly(BackendBaseClass):
    """
    Base Class for plots using the `plotly` library. Not meant for standalone use but rather to be called through :class:`Viewer`. 

    This class produces plots given a dictionary of instructions of the following form:
    ```
    { line: { line1: [x_data1, y_data1],
               line2: [x_data2, y_data2]
              },
      scatter: { scatter1: [x_data1, y_data1],
                 scatter2: [x_data2, y_data2]
                },
      histogram: { hist1: [bin_data1, hist_data1],
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
    Line/scatter plots are created for each key of the line/scatter dictionaries. The respective values have to be tuples/lists of length 2 including x and y data.
    The axes dictionary (as well as 'label' and 'scale') are optional and only intended to be used in case one wants to put axes labels or change to a log scale.

    :param template: Valid plotly theme. Either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], defaults to 'ggplot2'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: float, optional
    :param width: Figure width, defaults to 700
    :type width: float, optional
    """
    def __init__(self, template="ggplot2", height=500, width=700):

        self.line_names = list()
        self.scatter_names = list()
        self.histogram_names = list()
        self.heatmap_names = list()
        #self.x_marker_names = list()
        #self.y_marker_names = list()

        self.colors = cycle(px.colors.qualitative.Plotly)
    
        self._init_fig(template, height, width)

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
        
    def _show_legend(self, show=True):
        self.fig.update_layout(showlegend = show)

    def _add_line(self, x, y, name=None):
        self.fig.add_trace(go.Scatter(x=x,
                                      y=y,
                                      name=name,
                                      mode="lines",
                                      line={'color': next(self.colors),
                                            'width': 3},
                                      showlegend= True if name is not None else False) 
                                      )
        if name is not None: self.line_names.append(name)

    def _add_scatter(self, x, y, name=None):
        self.fig.add_trace(go.Scatter(x=x,
                                      y=y,
                                      name=name,
                                      mode="markers",
                                      line={'color': next(self.colors),
                                            'width': 3},
                                      showlegend= True if name is not None else False) 
                                      )
        if name is not None: self.scatter_names.append(name)

    def _add_histogram(self, bins, data, name=None):
        if bins is None:
            arg = dict()
        elif isinstance(bins, int):
            arg = dict(nbinsx=bins)
        elif isinstance(bins, tuple) and len(bins) == 3:
            arg = dict(xbins=dict(start=bins[0], end=bins[1], size=(bins[1]-bins[0])/bins[2]) )
        else:
            raise TypeError("Bin info has to be either None, an integer (number of bins), or a tuple of length 3 (start, end, number of bins)")
        
        self.fig.add_trace(go.Histogram(name=name, 
                                        x=data,
                                        **arg,
                                        showlegend = True if name is not None else False)
                            )
        if name is not None: self.histogram_names.append(name)

        # lower opacity of histograms if more than one is plotted
        if len([k for k in self.fig.select_traces(selector="histogram")]) > 1:
            self.fig.update_traces(selector="histogram", patch=dict(opacity=0.8))

    def _update_line(self, name, x, y):
        self.fig.update_traces(dict(x=x, y=y), selector=dict(name=name))

    def _update_scatter(self, name, x, y):
        self.fig.update_traces(dict(x=x, y=y), selector=dict(name=name))

    def _update_histogram(self, name, bins, data):
        #n_histograms = len([k for k in self.fig.select_traces(selector="histogram")])
        opacity = 0.8 if len(self.histogram_names)>1 else 1

        if bins is None:
            arg = dict()
        elif isinstance(bins, int):
            arg = dict(nbinsx=bins)
        elif isinstance(bins, tuple) and len(bins) == 3:
            arg = dict(xbins=dict(start=bins[0], end=bins[1], size=(bins[1]-bins[0])/bins[2]) )
        else:
            raise TypeError("Bin info has to be either None, an integer (number of bins), or a tuple of length 3 (start, end, number of bins)")
        
        self.fig.update_traces(dict(x=data, opacity=opacity, **arg), selector=dict(name=name))
    
    def _set_axes(self, data):
        if "xaxis" in data.keys():
            if "label" in data["xaxis"].keys():
                self.fig.layout.xaxis.title = data["xaxis"]["label"]
            if "scale" in data["xaxis"].keys():
                self.fig.layout.xaxis.type = data["xaxis"]["scale"]

        if "yaxis" in data.keys():
            if "label" in data["yaxis"].keys():
                self.fig.layout.yaxis.title = data["yaxis"]["label"]
            if "scale" in data["yaxis"].keys():
                self.fig.layout.yaxis.type = data["yaxis"]["scale"]

    # def _add_xmarker(self, x_marker_names):
    #     for n in x_marker_names: 
    #         self.fig.add_vline(x=0, name=n, line_dash='dash', line_color="Black", line_width=2)
    #     self.x_marker_names = x_marker_names

    # def _add_ymarker(self, y_marker_names):
    #     for n in y_marker_names: 
    #         self.fig.add_hline(y=0, name=n, line_dash='dash', line_color="Black", line_width=2)
    #     self.y_marker_names = y_marker_names

    # def __add_linemarker(self):
    #     ...

    # def _add_annotation(self):
    #     ...

    
        
        # update heatmaps # not yet finished (TODO)
        # for k in self.heatmap_names:
        #     opacity = 0.8 if len(self.heatmap_names)>1 else 1

        #     # x binning
        #     if data["heatmap"][k][0][0] is None:
        #         argx = dict()
        #     elif isinstance(data["heatmap"][k][0][0], int):
        #         argx = dict(nbinsx=data["heatmap"][k][0][0])
        #     elif isinstance(data["heatmap"][k][0][0], tuple):
        #         argx = dict(xbins=dict(start=data["heatmap"][k][0][0][0],
        #                               end=data["heatmap"][k][0][0][1],
        #                               size=(data["heatmap"][k][0][0][1]-data["heatmap"][k][0][0][0])/data["heatmap"][k][0][0][2])
        #                   )
        #     else:
        #         raise TypeError("Bin info has to be either None, an integer (number of bins), or a tuple of length 3 (start, end, number of bins)")
                
        #     # y binning
        #     if data["heatmap"][k][0][1] is None:
        #         argy = dict()
        #     elif isinstance(data["heatmap"][k][0][1], int):
        #         argy = dict(nbinsy=data["heatmap"][k][0][1])
        #     elif isinstance(data["heatmap"][k][0][1], tuple):
        #         argy = dict(ybins=dict(start=data["heatmap"][k][0][1][0],
        #                               end=data["heatmap"][k][0][1][1],
        #                               size=(data["heatmap"][k][0][1][1]-data["heatmap"][k][0][1][0])/data["heatmap"][k][0][1][2])
        #                   )
        #     else:
        #         raise TypeError("Bin info has to be either None, an integer (number of bins), or a tuple of length 3 (start, end, number of bins)")

        #     H, xedges, yedges = np.histogram2d(x=data["heatmap"][k][1][0],
        #                                        y=data["heatmap"][k][1][1])
            
        #     self.fig.update_traces(dict(z=H,
        #                                 y=data["heatmap"][k][1][1],
        #                                 opacity=opacity,
        #                                 **argx, **argy),
        #                                 selector=dict(name=k))
        # for k in self.y_marker_names:
        #     self.fig.update_shapes(dict(y0=data["y_marker"][k], y1=data["y_marker"][k]), selector=dict(name=k))

        # for k in self.x_marker_names:
        #     self.fig.update_shapes(dict(x0=data["x_marker"][k], x1=data["x_marker"][k]), selector=dict(name=k))

class BaseClassMPL(BackendBaseClass):
    """
    Base Class for plots using the `matplotlib` library. Not meant for standalone use but rather to be called through :class:`Viewer`. 

    This class produces plots given a dictionary of instructions of the following form:
    ```
    { line: { line1: [x_data1, y_data1],
               line2: [x_data2, y_data2]
              },
      scatter: { scatter1: [x_data1, y_data1],
                 scatter2: [x_data2, y_data2]
                },
      histogram: { hist1: [bin_data1, hist_data1],
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
    Line/scatter plots are created for each key of the line/scatter dictionaries. The respective values have to be tuples/lists of length 2 including x and y data.
    The axes dictionary (as well as 'label' and 'scale') are optional and only intended to be used in case one wants to put axes labels or change to a log scale.

    :param template: Valid matplotlib theme. Either of ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'], defaults to 'ggplot'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: float, optional
    :param width: Figure width, defaults to 700
    :type width: float, optional
    """
    def __init__(self, template="ggplot", height=500, width=700):
        self.line_names = list()
        self.scatter_names = list()
        self.histogram_names = list()
        self.heatmap_names = list()
    
        self._init_fig(template, height, width)

    def _init_fig (self, template, height, width):
        self.fig = plt.figure()
        self.fig.add_axes((0,0,1,1))

    # def _show_legend(self, show=True):
    #     self.fig.update_layout(showlegend = show)

    def _add_line(self, x, y, name=None):
        self.fig.axes[0].plot(x, y, "-", linewidth=2, label=name)
        if name is not None: self.line_names.append(name)

    def _add_scatter(self, x, y, name=None):
        self.fig.axes[0].plot(x, y, "o", markersize=2, label=name)
        if name is not None: self.scatter_names.append(name)

    # def _add_histogram(self, bins, data, name=None):
    #     if bins is None:
    #         arg = dict()
    #     elif isinstance(bins, int):
    #         arg = dict(nbinsx=bins)
    #     elif isinstance(bins, tuple) and len(bins) == 3:
    #         arg = dict(xbins=dict(start=bins[0], end=bins[1], size=(bins[1]-bins[0])/bins[2]) )
    #     else:
    #         raise TypeError("Bin info has to be either None, an integer (number of bins), or a tuple of length 3 (start, end, number of bins)")
        
    #     self.fig.add_trace(go.Histogram(name=name, 
    #                                     x=data,
    #                                     **arg,
    #                                     showlegend = True if name is not None else False)
    #                         )
    #     if name is not None: self.histogram_names.append(name)

    #     # lower opacity of histograms if more than one is plotted
    #     if len([k for k in self.fig.select_traces(selector="histogram")]) > 1:
    #         self.fig.update_traces(selector="histogram", patch=dict(opacity=0.8))

    # def _update_line(self, name, x, y):
    #     self.fig.update_traces(dict(x=x, y=y), selector=dict(name=name))

    # def _update_scatter(self, name, x, y):
    #     self.fig.update_traces(dict(x=x, y=y), selector=dict(name=name))

    # def _update_histogram(self, name, bins, data):
    #     #n_histograms = len([k for k in self.fig.select_traces(selector="histogram")])
    #     opacity = 0.8 if len(self.histogram_names)>1 else 1

    #     if bins is None:
    #         arg = dict()
    #     elif isinstance(bins, int):
    #         arg = dict(nbinsx=bins)
    #     elif isinstance(bins, tuple) and len(bins) == 3:
    #         arg = dict(xbins=dict(start=bins[0], end=bins[1], size=(bins[1]-bins[0])/bins[2]) )
    #     else:
    #         raise TypeError("Bin info has to be either None, an integer (number of bins), or a tuple of length 3 (start, end, number of bins)")
        
    #     self.fig.update_traces(dict(x=data, opacity=opacity, **arg), selector=dict(name=name))
    
    # def _set_axes(self, data):
    #     ...

class Viewer():
    """Class for plotting data given a dictionary of instructions (see below).

    :param data: Data dictionary containing line/scatter/axes informations (see below), defaults to None
    :type data: dict, optional
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl'], i.e. plotly or matplotlib, defaults to 'plotly'
    :type backend: str, optional
    :param show_controls: Set to True if plot controls should be shown. For the Viewer alone, this is just an "Exit" button which closes the plot, but inhereted objects can add more buttons with arbitrary functionality. Defaults to False
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
            
class InspectBaseClass(Viewer):
    """
    Base Class for inspecting the behavior of functions. Not meant for standalone use (see below).

    :param events: An iterable of events. Can be e.g. :class:`.file.EventIterator`, a 2d :class:`numpy.ndarray` or a list of List[float].
    :type events: iterable

    To use this class' functionality, subclass `:class:InspectBaseClass` and override the `:func:InspectBaseClass._calc_next` method. See its docstring for more information or `:class:Preview` for an example.

    `Keyword Arguments` are passed to class:`Viewer` and can be either of the following:
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl'], i.e. plotly or matplotlib, defaults to 'plotly'
    :type backend: str, optional
    :param template: Valid backend theme. E.g. for `plotly` backend either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], defaults to 'ggplot2'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: float, optional
    :param width: Figure width, defaults to 700
    :type width: float, optional
    """
    def __init__(self, events, **kwargs):

        viewer_kwargs = {k:v for k,v in kwargs.items() if k in ["backend","template","width","height"]}
        for k in ["backend","template","width","height"]: kwargs.pop(k, None)
        super().__init__(data=None, **viewer_kwargs)

        self.kwargs = kwargs
        self.events = iter(events)

        self._button_next = widgets.Button(description="Next")
        self._button_next.on_click(self._update_plot)
        self.buttons.append(self._button_next)  

    def _calc_next(self, ev, **kwargs):
        """
        Function to calculate the plot dictionary for a given event.

        :param ev: The event as contained in the iterable object used to construct this class.
        :type ev: List[float]

        :return: Dictionary of the form (see below) with instructions for how to plot the data calculated for `ev`.
        :rtype: dict

        ```
        { line: { line1: [x_data1, y_data1],
                line2: [x_data2, y_data2]
                },
        scatter: { scatter1: [x_data1, y_data1],
                    scatter2: [x_data2, y_data2]
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
        ...

    def _update_plot(self, b=None):
        try: 
            self.plot(self._calc_next(next(self.events)))
        except StopIteration: 
            self.close()
            print("No more events to inspect.")
        except:
            self.close()
            raise
    
    def start(self):
        """
        Show the plot and start iterating over the events.
        """
        self._update_plot()
        self.show()