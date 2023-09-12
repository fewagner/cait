from itertools import cycle
from abc import ABC, abstractmethod
from typing import List, Union

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

#########################
# Function Base Classes #
#########################

class FncBaseClass(ABC):
    def __init__(self, **kwargs):
        # Use kwargs as variables in __call__ function. This way, the inputs are also documented in the function class (via the __init__ method) and __call__ does not need to specify the inputs again
        self.kw = kwargs
        # all variables that can be cached are stored here
        self.cache = dict() 
        # all variables which could be relevant for plotting are stored here
        self.data = dict() 
        
    @abstractmethod
    def __call__(self, event):
        ...
        
    def update_kwargs(self, **kwargs):
        for k,v in kwargs.items():
            self.kw[k] = v
        
    def preview(self, event):
        raise NotImplementedError(f"{self.__class__.__name__} does not support preview.")
        
class FitFncBaseClass(FncBaseClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def model(self, x, *pars):
        # TODO: check for number of arguments (signature alone is insufficient because of possible *par)
        return self._model_fnc(x, *pars)
        
    def _model_fnc(self, x, *pars):
        raise NotImplementedError(f"No model function defined for {self.__class__.__name__}.")
    
#####################
# Plot Base Classes #
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

##### EXPERIMENTAL START ######  
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