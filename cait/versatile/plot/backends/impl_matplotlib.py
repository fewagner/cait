from typing import Callable, List, Union
import io
import base64

from ipywidgets import widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np

from .backendbase import BackendBaseClass
from .helper import EmptyRep

# Helper function to make seaborn style consistent also
# in newer matplotlib versions
def _correct_seaborn_styles(style: str):
    if not style.startswith("seaborn"): 
        return style
    elif style.startswith("seaborn-v0_8"): 
        return style
    else:
        return "seaborn-v0_8" + style.split("seaborn")[-1]

class BaseClassMPL(BackendBaseClass):
    """
    Base Class for plots using the `matplotlib` library. Not meant for standalone use but rather to be called through :class:`Viewer`. 

    This class produces plots given a dictionary of instructions of the following form:
    ::
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
                        }
                    }
                }

    Line/scatter plots are created for each key of the line/scatter dictionaries. The respective values have to be tuples/lists of length 2 including x and y data.
    The axes dictionary (as well as 'label' and 'scale') are optional and only intended to be used in case one wants to put axes labels or change to a log scale.

    :param template: Custom style 'cait', 'science' or any valid matplotlib theme, i.e. either of ['default', 'classic', 'Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'], defaults to 'seaborn'
    :type template: str, optional
    :param height: Figure height, defaults to 3
    :type height: float, optional
    :param width: Figure width, defaults to 5
    :type width: float, optional
    """
    def __init__(self, 
                 template: str = "seaborn", 
                 height: float = 3, 
                 width: float = 5, 
                 show_controls: bool = False):
        
        self._line_names = list()
        self._scatter_names = list()
        self._histogram_names = list()
        self.heatmap_names = list()

        # To catch the missing seaborn styles in newer matplotlib versions
        if type(template) is str:
            template = _correct_seaborn_styles(template)
        if type(template) is list:
            template = [_correct_seaborn_styles(t) for t in template]
            
        # Special styles
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

        # Used for auto scaling of axes. 
        # If user sets xlim or ylim, they are not auto-scaled
        self._x_lim_auto = True
        self._y_lim_auto = True
    
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

    def _get_artist(self, name: str):
        ind = [l.get_label() for l in self.fig.axes[0].lines].index(name)
        return self.fig.axes[0].lines[ind]

    def _set_axes(self, data: dict):
        with plt.style.context(self.template):
            if "xaxis" in data.keys():
                if "label" in data["xaxis"].keys():
                    self.fig.axes[0].set_xlabel(data["xaxis"]["label"])
                if "scale" in data["xaxis"].keys():
                    self.fig.axes[0].set_xscale(data["xaxis"]["scale"])
                if "range" in data["xaxis"].keys():
                    r = data["xaxis"]["range"]
                    self.fig.axes[0].set_xlim(r)
                    self._x_lim_auto = r is None

            if "yaxis" in data.keys():
                if "label" in data["yaxis"].keys():
                    self.fig.axes[0].set_ylabel(data["yaxis"]["label"])
                if "scale" in data["yaxis"].keys():
                    self.fig.axes[0].set_yscale(data["yaxis"]["scale"])
                if "range" in data["yaxis"].keys():
                    r = data["yaxis"]["range"]
                    self.fig.axes[0].set_ylim(r)
                    self._y_lim_auto = r is None

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
        self._save_html.update(EmptyRep())

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
        self.fig.axes[0].autoscale(enable=self._x_lim_auto, axis="x", tight=True)
        self.fig.axes[0].autoscale(enable=self._y_lim_auto, axis="y")

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