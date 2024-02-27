from itertools import cycle
from abc import ABC, abstractmethod
from typing import List, Union, Callable, Tuple
import io
import base64

from ipywidgets import widgets
from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

try:
    import uniplot
except ImportError:
    uniplot = None

#########################
##### Helper Classes ####
#########################

class EmptyRep:
    # Helper Class that generates empty cell output
    def __repr__(self):
        return ""

#####################
# Plot Base Classes #
#####################

class BackendBaseClass(ABC):

    @abstractmethod
    def __init__(template: str, height: int, width: int, show_controls: bool):
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
    def _add_vmarker(marker_pos: List[float], y_int: Tuple[float], name: str = None):
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
    def _update_vmarker(name: str, marker_pos: List[float], y_int: Tuple[float]):
        ...

    @abstractmethod
    def _set_axes(data: dict):
        ...

    @abstractmethod
    def _add_button(text: str, callback: Callable, tooltip: str = None, where: int = -1, key: str = None):
        ...

    @abstractmethod
    def _get_figure():
        ...

    @abstractmethod
    def _show():
        ...

    @abstractmethod
    def _update():
        ...

    @abstractmethod
    def _close(b=None):
        ...

    @property
    @abstractmethod
    def line_names(self):
        ...

    @property
    @abstractmethod
    def scatter_names(self):
        ...

    @property
    @abstractmethod
    def histogram_names(self):
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
    :type height: int, optional
    :param width: Figure width, defaults to 700
    :type width: int, optional
    """
    def __init__(self, 
                 template: str = "ggplot2", 
                 height: int = 500, 
                 width: int = 700, 
                 show_controls: bool = False):
     
        self._line_names = list()
        self._scatter_names = list()
        self._histogram_names = list()
        self.heatmap_names = list()
        #self.x_marker_names = list()
        #self.y_marker_names = list()

        self.colors = cycle(px.colors.qualitative.Plotly)

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
                                      line={'color': next(self.colors),
                                            'width': 3},
                                      showlegend= True if name is not None else False) 
                                      )
        if name is not None: self._line_names.append(name)

    def _add_scatter(self, x, y, name=None):
        self.fig.add_trace(go.Scatter(x=x,
                                      y=y,
                                      name=name,
                                      mode="markers",
                                      line={'color': next(self.colors),
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
                                      line={'color': next(self.colors),
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
        else:
            raise TypeError("Bin info has to be either None, an integer (number of bins), or a tuple of length 3 (start, end, number of bins)")
        
        self.fig.update_traces(dict(x=data, opacity=opacity, **arg), selector=dict(name=name))

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

    def _get_info(self, b):
        xmin, xmax = self.fig.layout.xaxis.range

        if type(xmin) is str: # then we assume it to be datetime
            xmin = np.datetime64(xmin)
            xmax = np.datetime64(xmax)

        ymin, ymax = self.fig.layout.yaxis.range

        y_vals = list()
        for trace in self.fig.select_traces():
            if trace.visible is True:
                y = trace.y
                x = trace.x if trace.x is not None else np.arange(len(y))
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

    :param template: Custom style 'cait', 'science' or any valid matplotlib theme, i.e. either of ['default', 'classic', 'Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'], defaults to 'cait'
    :type template: str, optional
    :param height: Figure height, defaults to 3
    :type height: float, optional
    :param width: Figure width, defaults to 5
    :type width: float, optional
    """
    def __init__(self, 
                 template: str = "cait", 
                 height: float = 3, 
                 width: float = 5, 
                 show_controls: bool = False):
        
        self._line_names = list()
        self._scatter_names = list()
        self._histogram_names = list()
        self.heatmap_names = list()

        if type(template) is str and template == "science":
            template = "cait.styles.science"
        elif type(template) is str and template == "cait":
            template = "cait.styles.cait_style"
        elif type(template) is list and "science" in template:
            template[template.index("science")] = "cait.styles.science"
        elif type(template) is list and "cait" in template:
            template[template.index("cait")] = "cait.styles.cait_style"

        self.template = template
        self.show_controls = show_controls
        self.buttons_initialized = False
        self.is_visible = False
    
        self._init_fig(height, width)

    def _init_fig (self, height, width):
        with plt.style.context(self.template):
            self.fig = plt.figure(figsize=(width, height))
            self.fig.add_axes((0,0,1,1))

    def _init_UI(self):
        if self.show_controls:
            if not self.buttons_initialized:
                self._add_button("Exit", self._close, "Close plot widget.", 0)
                self._add_button("Save .png", lambda b: self._save_figure("png"), "Download this figure as PNG.")
                self._add_button("Save .pdf", lambda b: self._save_figure("pdf"), "Download this figure as PDF.")
                self.buttons_initialized = True
            self.UI = widgets.HBox(self.buttons)
        else:
            self.UI = None

    def _add_button(self, text: str, callback: Callable, tooltip: str = None, where: int = -1, key: str = None):
        # This gives inherited classes the possibility to add buttons themselves before 
        # calling super().__init__
        if not hasattr(self, "buttons"): self.buttons = []

        if where == -1: self.buttons.append(widgets.Button(description=text, tooltip=tooltip))
        else: self.buttons.insert(where, widgets.Button(description=text, tooltip=tooltip))

        self.buttons[where].on_click(callback) 

    def _show_legend(self, show: bool = True):
        if not show:
            raise NotImplementedError("'show=False' not implemented for 'backend=mpl'")
        
        with plt.style.context(self.template):
            self.fig.legend()
            
        self._draw()

    def _add_line(self, x, y, name=None):
        # Plotly supports x=None, matplotlib has to handle it. We set it
        # here to get consistent results between the two backends
        if x is None and y is not None: x = np.arange(len(y))
        if x is None and y is None: x, y = np.nan, np.nan

        if name is not None: self._line_names.append(name)

        with plt.style.context(self.template):
            self.fig.axes[0].plot(x, y, marker='none', label=name)

        self._draw()

    def _add_scatter(self, x, y, name=None):
        # Plotly supports x=None, matplotlib has to handle it. We set it
        # here to get consistent results between the two backends
        if x is None and y is not None: x = np.arange(len(y))
        if x is None and y is None: x, y = np.nan, np.nan

        if name is not None: self._scatter_names.append(name)

        with plt.style.context(self.template):
            self.fig.axes[0].plot(x, y, linestyle='none', marker=".", label=name)

        self._draw()

    def _add_histogram(self, bins, data, name=None):
        if bins is None:
            arg = dict()
        elif isinstance(bins, int):
            arg = dict(bins=bins)
        elif isinstance(bins, tuple) and len(bins) == 3:
            arg = dict(bins=np.arange(bins[0], bins[1], (bins[1]-bins[0])/bins[2]))
        else:
            raise TypeError("Bin info has to be either None, an integer (number of bins), or a tuple of length 3 (start, end, number of bins)")
        
        if name is not None: self._histogram_names.append(name)

        with plt.style.context(self.template):
            self.fig.axes[0].hist(x=data, **arg, label=name)

        self._draw()

        # lower opacity of histograms if more than one is plotted
        #if len([k for k in self.fig.select_traces(selector="histogram")]) > 1:
        #    self.fig.update_traces(selector="histogram", patch=dict(opacity=0.8))

    def _add_vmarker(self, marker_pos, y_int, name=None):
        if marker_pos is None or y_int is None: 
            x, y = np.nan, np.nan
        else:
            # Stack three of those arrays on top of each other and flatten in column-major order
            # This results in an array where each ts_datetime point comes three times in a row
            x = np.array([marker_pos, marker_pos, marker_pos]).flatten(order="F")

            # y-values will jump between y_min and y_max and are separated by a None
            y = [np.min(y_int), np.max(y_int), None]*np.array(marker_pos).shape[-1]

        if name is not None: self._line_names.append(name)

        with plt.style.context(self.template):
            self.fig.axes[0].plot(x, y, marker='none', linestyle='--', label=name)

        self._draw()

    def _update_line(self, name: str, x: List[float], y: List[float]):
        # Plotly supports x=None, matplotlib has to handle it. We set it
        # here to get consistent results between the two backends
        if x is None and y is not None: x = np.arange(len(y))
        if x is None and y is None: x, y = np.nan, np.nan

        ind = [l.get_label() for l in self.fig.axes[0].lines].index(name)

        with plt.style.context(self.template):
            self.fig.axes[0].lines[ind].set_xdata(x)
            self.fig.axes[0].lines[ind].set_ydata(y)

        #self._draw()

    def _update_scatter(self, name: str, x: List[float], y: List[float]):
        # Plotly supports x=None, matplotlib has to handle it. We set it
        # here to get consistent results between the two backends
        if x is None and y is not None: x = np.arange(len(y))
        if x is None and y is None: x, y = np.nan, np.nan

        ind = [l.get_label() for l in self.fig.axes[0].lines].index(name)

        with plt.style.context(self.template):
            self.fig.axes[0].lines[ind].set_xdata(x)
            self.fig.axes[0].lines[ind].set_ydata(y)
        
        #self._draw()

    def _update_histogram(self, name: str, bins: Union[int, tuple], data: List[float]):
        ...

    def _update_vmarker(self, name, marker_pos, y_int):
        if marker_pos is None or y_int is None: 
            x, y = np.nan, np.nan
        else:
            # Stack three of those arrays on top of each other and flatten in column-major order
            # This results in an array where each ts_datetime point comes three times in a row
            x = np.array([marker_pos, marker_pos, marker_pos]).flatten(order="F")

            # y-values will jump between y_min and y_max and are separated by a None
            y = [np.min(y_int), np.max(y_int), None]*np.array(marker_pos).shape[-1]

        ind = [l.get_label() for l in self.fig.axes[0].lines].index(name)

        with plt.style.context(self.template):
            self.fig.axes[0].lines[ind].set_xdata(x)
            self.fig.axes[0].lines[ind].set_ydata(y)

        #self._draw()

    def _set_axes(self, data: dict):
        with plt.style.context(self.template):
            if "xaxis" in data.keys():
                if "label" in data["xaxis"].keys():
                    self.fig.axes[0].set_xlabel(data["xaxis"]["label"])
                if "scale" in data["xaxis"].keys():
                    self.fig.axes[0].set_xscale(data["xaxis"]["scale"])

            if "yaxis" in data.keys():
                if "label" in data["yaxis"].keys():
                    self.fig.axes[0].set_ylabel(data["yaxis"]["label"])
                if "scale" in data["yaxis"].keys():
                    self.fig.axes[0].set_yscale(data["yaxis"]["scale"])
        self._draw()

    def _save_figure(self, fmt: str):
        buffer = io.BytesIO()
        self.fig.savefig(buffer, format=fmt, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        payload = graphic.decode('utf-8')
        
        html = HTML(f"""
                    <html>
                    <body>
                    <a download="figure_from_{np.datetime64('now')}.{fmt}" id="dl_mpl_figure" href="data:image/{fmt};base64,{payload}"></a>

                    <script>
                    (function download() {{
                    document.getElementById('dl_mpl_figure').click();
                    console.log("clicked")
                    }})()
                    </script>

                    </body>
                    </html>
                    """)
        
        self._save_html.update(html)

    def _show(self):
        self.is_visible = True

        self._init_UI()
        if self.UI is not None:
            if not hasattr(self, "_controls"):
                self._controls = display(self.UI, display_id=True)
            else:
                self._controls.update(self.UI)

            # Placeholder for download functionality
            if not hasattr(self, "_save_html"):
                self._save_html = display(EmptyRep(), display_id=True)
            else: 
                self._save_html.update(EmptyRep())

        self._draw()

        # Needed to suppress automatic jupyter output (leads to figure showing twice)
        plt.close()

    def _update(self):
        self._draw()

    def _close(self, b=None):
        self.is_visible = False

        if self.UI is not None:
            self._controls.update(EmptyRep())

        self._display.update(EmptyRep())

    def _draw(self):
        # Rescale axis limits (plotly does this automatically)
        self.fig.axes[0].relim()
        self.fig.axes[0].autoscale_view()

        if self.is_visible:
            if not hasattr(self, "_display"): 
                self._display = display(self.fig, display_id=True)
            else:
                self._display.update(self.fig)

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
    
class BaseClassUniplot(BackendBaseClass):
    """
    Base Class for plots using the `uniplot` library (has to be installed). Not meant for standalone use but rather to be called through :class:`Viewer`. 

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

    :param height: Figure height, defaults to 17
    :type height: int, optional
    :param width: Figure width, defaults to 60
    :type width: int, optional
    """
    def __init__(self, 
                 template: str = None, # For consistency with other backends. Has no effect.
                 height: int = 17, 
                 width: int = 60, 
                 show_controls: bool = True):
        
        if uniplot is None: 
            raise RuntimeError("Install 'uniplot>=0.12.2' to use this feature.")

        # Height/width in characters
        self.height = int(height)
        self.width = int(width)
        
        # Dictionaries holding x/y values to be drawn on the command line every time
        # the plot is updated
        self.lines = dict()
        self.scatters = dict()
        self.histograms = dict()

        self.show_controls = show_controls
        self.buttons = list()

        self._is_visible = False

        # Configurations for the plot (are passed on to plot routine)
        self.plt_opt = uniplot.options.Options(
            width=self.width,
            height=self.height
            )
        
        # Uniplot does not support x- and y-labels. Therefore we have to do a workaround
        self.axis_labels = {"x": None, "y": None}

        self._add_button("exit", self._close, "exit", None, "x")

    def _add_button(self, text: str, callback: Callable, tooltip: str = None, where: int = None, key: str = None):
        if key is not None:
            self.buttons.append( 
                {
                    "key": key,
                    "text": f"{key}: {text}",
                    "callback": callback
                })

    def _show_legend(self, show: bool = True):
        ...

    def _add_line(self, x, y, name=None):
        if x is None and y is not None: x = np.arange(len(y))
        if x is None and y is None: x, y = np.nan, np.nan

        if name is None: name = f"line {len(self.lines)+1}"
        self.lines[name] = [x, y]

    def _add_scatter(self, x, y, name=None):
        if x is None and y is not None: x = np.arange(len(y))
        if x is None and y is None: x, y = np.nan, np.nan

        if name is None: name = f"scatter {len(self.scatters)+1}"
        self.scatters[name] = [x, y]

    def _add_histogram(self, bins, data, name=None):
        if bins is None:
            arg = dict()
        elif isinstance(bins, int):
            arg = dict(bins=bins)
        elif isinstance(bins, tuple) and len(bins) == 3:
            arg = dict(bins=np.arange(bins[0], bins[1], (bins[1]-bins[0])/bins[2]))
        else:
            raise TypeError("Bin info has to be either None, an integer (number of bins), or a tuple of length 3 (start, end, number of bins)")

        hist, bin_edges = np.histogram(data, **arg)

        x = np.zeros(2*len(bin_edges))
        y = np.zeros(2*len(bin_edges))
        x[0] = bin_edges[0]
        x[1::2] = bin_edges
        x[2::2] = bin_edges[1:]
        y[1:-1:2] = hist
        y[2:-1:2] = hist
        
        if name is None: name = f"histogram {len(self.histograms)+1}"
        self.histograms[name] = [x, y]
        
    def _add_vmarker(self, marker_pos, y_int, name=None):
        raise NotImplementedError("vmarker not implemented for backend 'uniplot'")

    def _update_line(self, name: str, x: List[float], y: List[float]):
        if x is None and y is not None: x = np.arange(len(y))
        if x is None and y is None: x, y = np.nan, np.nan

        self.lines[name] = [x, y]

    def _update_scatter(self, name: str, x: List[float], y: List[float]):
        if x is None and y is not None: x = np.arange(len(y))
        if x is None and y is None: x, y = np.nan, np.nan

        self.scatters[name] = [x, y]

    def _update_histogram(self, name: str, bins: Union[int, tuple], data: List[float]):
        ...

    def _update_vmarker(self, name, marker_pos, y_int):
        raise NotImplementedError("vmarker not implemented for backend 'uniplot'")

    def _set_axes(self, data: dict):
        if "xaxis" in data.keys():
            if "label" in data["xaxis"].keys() and data["xaxis"]["label"] is not None:
                self.axis_labels["x"] = data["xaxis"]["label"]
            if "scale" in data["xaxis"].keys():
                self.plt_opt.x_as_log = data["xaxis"]["scale"] == "log"

        if "yaxis" in data.keys():
            if "label" in data["yaxis"].keys() and data["yaxis"]["label"] is not None:
                self.axis_labels["y"] = data["yaxis"]["label"]
            if "scale" in data["yaxis"].keys():
                self.plt_opt.y_as_log = data["yaxis"]["scale"] == "log"

        title = ""
        if self.axis_labels["x"] is not None: 
            title += "x: " + self.axis_labels["x"]
        if self.axis_labels["y"] is not None:
            if self.axis_labels["x"] is not None:
                title += ", "
            title += "y: " + self.axis_labels["y"]

        self.plt_opt.title = "\n" + title if title else None

    def _clear(self):
        # Delete previous plot (if existent) from command line
        if self._is_visible:
            nr_lines_to_erase = self.plt_opt.height + 4
            if self.plt_opt.legend_labels is not None:
                nr_lines_to_erase += len(self.plt_opt.legend_labels)
            if self.buttons:
                nr_lines_to_erase += 1
            if self.plt_opt.title is not None:
                nr_lines_to_erase += 2
            uniplot.plot_elements.erase_previous_lines(nr_lines_to_erase)

    def _show(self):
        self._is_visible = True
        self._draw()

    def _update(self):
        self._clear()
        self._draw()

    def _close(self, b=None):
        self._clear()
        self._is_visible = False

    def _draw(self):
        if not self._is_visible: return 

        # All plots are drawn identically (using uniplot.plot)
        # The distinction between line/scatter/histogram arises from the option 'lines'
        # We collect all x- and y-data in a list as well as the corresponding 'lines' 
        # options and legend entries
        xs = [v[0] for v in (list(self.lines.values())
                             +list(self.scatters.values())
                             +list(self.histograms.values())
                             )]
        ys = [v[1] for v in (list(self.lines.values())
                             +list(self.scatters.values())
                             +list(self.histograms.values())
                             )]
        
        self.plt_opt.legend_labels = list(self.lines.keys()) + list(self.scatters.keys()) + list(self.histograms.keys())
        self.plt_opt.lines = [True]*len(self.lines) + [False]*len(self.scatters) + [True]*len(self.histograms)

        if not xs: return

        # Calculate axis limits and save in options object (this can later be modified
        # interactively using keys to zoom/pan)
        minx, maxx = np.min(np.concatenate(xs)), np.max(np.concatenate(xs))
        miny, maxy = np.min(np.concatenate(ys)), np.max(np.concatenate(ys))
        xran, yran = maxx-minx, maxy-miny
        self.plt_opt.x_min = minx - 0.01*xran
        self.plt_opt.x_max = maxx + 0.01*xran
        self.plt_opt.y_min = miny - 0.01*yran
        self.plt_opt.y_max = maxy + 0.01*yran

        n, key = 0, ''

        # Buttons that end the loop (pan/zoom buttons stay in the loop)
        hot_keys = [b["key"].lower() for b in self.buttons]

        while True:
            if n > 0: self._clear()

            kwargs = {k: getattr(self.plt_opt, k) for k in ["legend_labels", "x_min", "x_max", "y_min", "y_max", "width", "height", "x_as_log", "y_as_log", "lines", "title"] if getattr(self.plt_opt, k) is not None}

            uniplot.plot(ys, xs, color=True, **kwargs)
    
            # If controls are diabled, no looping is necessary (no awaiting inputs)
            if not self.show_controls: break

            print("Move with S/D/F/E, zoom with I/O, R to reset. ESC/Q to quit.")
            if self.buttons:
                print(f"other actions: {', '.join([b['text'] for b in self.buttons])}")
                
            # Get key input
            key = uniplot.getch.getch().lower()

            if key in ["q", "\x1b"] + hot_keys: 
                # Break out of loop (below, we distinguish between hot
                # keys and q/ESC)
                break
            elif key == "i":
                self.plt_opt.zoom_in()
            elif key == "o":
                self.plt_opt.zoom_out()
            elif key == "r":
                self.plt_opt.x_min = minx - 0.01*xran
                self.plt_opt.x_max = maxx + 0.01*xran
                self.plt_opt.y_min = miny - 0.01*yran
                self.plt_opt.y_max = maxy + 0.01*yran
            elif key == "s":
                self.plt_opt.shift_view_left()
            elif key == "f":
                self.plt_opt.shift_view_right()
            elif key == "e":
                self.plt_opt.shift_view_up()
            elif key == "d":
                self.plt_opt.shift_view_down()

            n += 1

        # Once we exit the loop, we check if a hot key was responsible for exiting and
        # call the respective callback if applicable
        if key in hot_keys: 
            self.buttons[hot_keys.index(key)]["callback"]()

    def _get_figure(self):
        return None
    
    @property
    def line_names(self):
        return list(self.lines.keys())
    
    @property
    def scatter_names(self):
        return list(self.scatters.keys())
    
    @property
    def histogram_names(self):
        return list(self.histograms.keys())