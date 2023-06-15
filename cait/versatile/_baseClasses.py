import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import widgets
from IPython.display import display
from itertools import cycle

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

class BaseClassPlotly:
    """Base Class for plots using the `plotly` library. Not meant for standalone use but rather to be called through :class:`Viewer`. 

    This class produces plots given a dictionary of instructions of the following form:
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
        self.plot_is_initialized = False

        self.line_names = list()
        self.scatter_names = list()
        self.x_marker_names = list()
        self.y_marker_names = list()

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
                        #modebar_add = ['drawline', 'eraseshape'],
                        #newshape=dict(line_color='yellow',
                        #        fillcolor='turquoise',
                        #        opacity=0.5)
                    )

    def _add_lines(self, names):
        traces = [go.Scatter(name=n,
                             mode="lines",
                             line={'color': next(self.colors),
                                   'width': 3},
                             showlegend=True) 
                            for n in names]
        self.fig.add_traces(traces)
        self.line_names = names

    def _add_scatter(self, names):
        traces = [go.Scatter(name=n,
                             mode="markers",
                             marker={'color': next(self.colors),
                                   'size': 5},
                             showlegend=True) 
                            for n in names]
        self.fig.add_traces(traces)
        self.scatter_names = names
    
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

    def _add_xmarker(self, x_marker_names):
        for n in x_marker_names: 
            self.fig.add_vline(x=0, name=n, line_dash='dash', line_color="Black", line_width=2)
        self.x_marker_names = x_marker_names

    def _add_ymarker(self, y_marker_names):
        for n in y_marker_names: 
            self.fig.add_hline(y=0, name=n, line_dash='dash', line_color="Black", line_width=2)
        self.y_marker_names = y_marker_names

    def _add_linemarker(self):
        ...

    def _add_annotation(self):
        ...

    def _init_plot (self, data):
        if "line" in data.keys():
            for v in data["line"].values():
                assert 2==len(v), "Line data has to be a tuple/list of length 2 containing x/y data respectively"
            self._add_lines(data["line"].keys())
        if "scatter" in data.keys(): 
            for v in data["scatter"].values():
                assert 2==len(v), "Scatter data has to be a tuple/list of length 2 containing x/y data respectively"
            self._add_scatter(data["scatter"].keys())

        if "axes" in data.keys(): self._set_axes(data["axes"])
        if "x_marker" in data.keys(): self._add_xmarker(data["x_marker"].keys())
        if "y_marker" in data.keys(): self._add_ymarker(data["y_marker"].keys())
        if "line_marker" in data.keys(): self._add_linemarker()
        if "annotation" in data.keys(): self._add_annotation(data["y_marker"])

        self.plot_is_initialized = True

    def _plot (self, data):
        if not self.plot_is_initialized: self._init_plot(data)

        for k in self.line_names:
                self.fig.update_traces(dict(x=data["line"][k][0],
                                            y=data["line"][k][1]),
                                       selector=dict(name=k))
        for k in self.scatter_names:
                self.fig.update_traces(dict(x=data["scatter"][k][0],
                                            y=data["scatter"][k][1]),
                                       selector=dict(name=k))
                
        for k in self.y_marker_names:
            self.fig.update_shapes(dict(y0=data["y_marker"][k], y1=data["y_marker"][k]), selector=dict(name=k))

        for k in self.x_marker_names:
            self.fig.update_shapes(dict(x0=data["x_marker"][k], x1=data["x_marker"][k]), selector=dict(name=k))

class BaseClassMPL:
    def __init__(self, template="seaborn", height=500, width=700):
        ##
        raise NotImplementedError("Matplotlib backend is not yet implemented.")
        ##
        self.plot_is_initialized = False

        self.xdata = None
        self.y_names = list()
        self.x_marker_names = list()
        self.y_marker_names = list()
    
        self._init_fig(template)

    def _init_fig (self, template):
        ...

    def _set_xdata(self, data):
        self.xdata = data

    def _add_ydata(self, y_names):
        # create lines
        self.y_names = y_names
    
    def _add_xmarker(self, x_marker_names):
        for n in x_marker_names: 
            ... # add markers
        self.x_marker_names = x_marker_names

    def _add_ymarker(self, y_marker_names):
        for n in y_marker_names: 
            ... # add markers
        self.y_marker_names = y_marker_names

    def _add_linemarker(self):
        ...

    def _add_annotation(self):
        ...

    def _init_plot (self, data):
        if "y_data" in data.keys(): self._add_ydata(data["y_data"].keys())
        if "x_marker" in data.keys(): self._add_xmarker(data["x_marker"].keys())
        if "y_marker" in data.keys(): self._add_ymarker(data["y_marker"].keys())
        if "line_marker" in data.keys(): self._add_linemarker()
        if "annotation" in data.keys(): self._add_annotation(data["y_marker"])

        self.plot_is_initialized = True

    def _plot (self, data):
        if not self.plot_is_initialized: self._init_plot(data)

        for k in self.y_names:
                ... 
                
        for k in self.y_marker_names:
            ...

        for k in self.x_marker_names:
            ...

class Viewer():
    """Class for plotting data given a dictionary of instructions (see below).

    :param data: Data dictionary containing line/scatter/axes informations (see below), defaults to None
    :type data: dict, optional
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl'], i.e. plotly or matplotlib, defaults to 'plotly'
    :type backend: str, optional

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
    def __init__(self, data=None, backend="plotly", **kwargs):
        if backend=="plotly":
            self.fig_widget = BaseClassPlotly(**kwargs)
        elif backend=="mpl":
            self.fig_widget = BaseClassMPL(**kwargs)
        else:
            raise NotImplementedError('Only backend "plotly" and "mpl" are supported.')
        
        self.buttons = list()

        self._button_exit = widgets.Button(description="Exit")
        self._button_exit.on_click(self.close)   
        self.buttons.append(self._button_exit)   

        self.visible = False

        if data is not None: 
            self.plot(data)  
            self.show()

    def _init_UI(self):
        self.UI = widgets.VBox([widgets.HBox(self.buttons), self.fig_widget.fig])

    def plot(self, data):
        """Plot data stored in dictionary.

        :param data: The data dictionary to plot (see :class:`Viewer` for details on how the dictionary needs to be structured)
        :type data: dict
        """
        self.fig_widget._plot(data)

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
    """Base Class for inspecting the behavior of functions. Not meant for standalone use (see below).

    :param events: An iterable of events. Can be e.g. :class:`.file.EventIterator`, a 2d :class:`numpy.ndarray` or a list of List[float].
    :type events: iterable

    To use this class' functionality, subclass :class:`InspectBaseClass` and override the :func:`InspectBaseClass._calc_next(self, ev, **kwargs)` method. See its docstring for more information or :class:`Preview` for an example.

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

        viewer_kwargs = {k:v for k,v in kwargs.items() if k in ["backend","template"]}
        for k in ["backend","template"]: kwargs.pop(k, None)
        super().__init__(data=None, **viewer_kwargs)

        self.kwargs = kwargs
        self.events = iter(events)

        self._button_next = widgets.Button(description="Next")
        self._button_next.on_click(self._update_plot)
        self.buttons.append(self._button_next)  

    def _calc_next(self, ev, **kwargs):
        """Function to calculate the plot dictionary for a given event.

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
        """Show the plot and start iterating over the events.
        """
        self._update_plot()
        self.show()