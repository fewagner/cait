from typing import List, Union

from ._baseClasses import InspectBaseClass, Viewer

class Preview(InspectBaseClass):
    """Class for inspecting the behavior of functions which were subclassed from :class:`._baseClasses.FncBase`.

    :param fnc: The function to be inspected, already initialized with the values that should stay fixed throughout the inspection.
    :type fnc: :class:`._baseClasses.FncBase`
    :param events: An iterable of events. Can be e.g. :class:`.file.EventIterator`, a 2d :class:`numpy.ndarray` or a list of List[float].
    :type events: iterable
    :param kwargs: Keyword arguments (see below)
    :type kwargs: Any

    `Keyword Arguments` are passed to class:`InspectBaseClass` and can be either of the following:
    :param backend: The backend to use for the plot. Either of ['plotly', 'mpl'], i.e. plotly or matplotlib, defaults to 'plotly'
    :type backend: str, optional
    :param template: Valid backend theme. E.g. for `plotly` backend either of ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'], defaults to 'ggplot2'
    :type template: str, optional
    :param height: Figure height, defaults to 500
    :type height: float, optional
    :param width: Figure width, defaults to 700
    :type width: float, optional
    """
    def __init__(self, fnc, events, **kwargs):
        super().__init__(events, **kwargs)
        self.fnc = fnc
        
        self.start()
        
    def _calc_next(self, ev):
        return self.fnc.preview(ev)
    
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


        lines = dict()
        if isinstance(y, dict):
            for key, value in y.items():
                lines[key] = [x, value]
        else:
            lines["line1"] = [x, y]

        data = dict(
            line = lines,
            axes = { "xaxis": { "label": xlabel,
                                "scale": xscale
                              },
                     "yaxis": { "label": ylabel,
                               "scale": yscale
                              }
                    }
        )

        super().__init__(data=data, **kwargs)

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


        lines = dict()
        if isinstance(y, dict):
            for key, value in y.items():
                lines[key] = [x, value]
        else:
            lines["line1"] = [x, y]

        data = dict(
            scatter = lines,
            axes = { "xaxis": { "label": xlabel,
                                "scale": xscale
                              },
                     "yaxis": { "label": ylabel,
                               "scale": yscale
                              }
                    }
        )

        super().__init__(data=data, **kwargs)