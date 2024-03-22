from typing import Callable, Union, List

import numpy as np

from .backendbase import BackendBaseClass

try:
    import uniplot
except ImportError:
    uniplot = None

class BaseClassUniplot(BackendBaseClass):
    """
    Base Class for plots using the `uniplot` library (has to be installed). Not meant for standalone use but rather to be called through :class:`Viewer`. 

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

        # For handling fixed axis ranges
        self._xrange = None
        self._yrange = None

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
            if "range" in data["xaxis"].keys():
                    self._xrange = data["xaxis"]["range"]

        if "yaxis" in data.keys():
            if "label" in data["yaxis"].keys() and data["yaxis"]["label"] is not None:
                self.axis_labels["y"] = data["yaxis"]["label"]
            if "scale" in data["yaxis"].keys():
                self.plt_opt.y_as_log = data["yaxis"]["scale"] == "log"
            if "range" in data["yaxis"].keys():
                    self.yrange = data["yaxis"]["range"]

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

        # Calculate axis limits (if not set) 
        # and save in options object (this can later be modified
        # interactively using keys to zoom/pan)
        if self._xrange:
            self.plt_opt.x_min, self.plt_opt.x_max = self._xrange
        else:
            minx, maxx = np.min(np.concatenate(xs)), np.max(np.concatenate(xs))
            xran = maxx - minx
            self.plt_opt.x_min = minx - 0.01*xran
            self.plt_opt.x_max = maxx + 0.01*xran

        if self._yrange:
            self.plt_opt.y_min, self.plt_opt.y_max = self._yrange
        else:
            miny, maxy = np.min(np.concatenate(ys)), np.max(np.concatenate(ys))
            yran = maxy - miny
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