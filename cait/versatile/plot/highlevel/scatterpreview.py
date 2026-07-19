import datetime
from typing import Callable, List

import numpy as np
from IPython.display import display
from ipywidgets import widgets

from ...eventfunctions.processing.helper import Unity
from ...iterators.iteratorbase import IteratorBaseClass
from ..viewer import Viewer


class ScatterPreview:
    """
    Scatter plot with event preview. 
    
    Clicking on scatter data displays the corresponding event in a separate figure (and the effect a function has on it, if provided). You can select scatter points and delete them from the plot using the "cut selected" button. The currently selected data indices and events are furthermore accessible through the respective methods.

    :param x: x-data for the scatter plot.
    :type x: List[float]
    :param y: y-data for the scatter plot.
    :type y: List[float]
    :param ev_it: Event iterator corresponding to the x-y data (if a point (x,y) is clicked, the corresponding event from the iterator is displayed).
    :type ev_it: IteratorBaseClass
    :param f: The function to be inspected, already initialized with the values that should stay fixed throughout the inspection. Defaults to Unity (which means that just the events of the iterable will be displayed)
    :type f: :class:`cait.versatile.eventfunctions.functionbase.FncBaseClass`
    :param kwargs: Keyword arguments for the cait.versatile.Viewer class.
    :type kwargs: dict, optional
    """
    def __init__(self, x: List[float], y: List[float], ev_it: IteratorBaseClass, f: Callable = None, **kwargs):
        # due to explicit usage of plotly attributes, only this backend can be supported
        if "backend" in kwargs and kwargs["backend"]!="plotly":
            raise NotImplementedError(f"{self.__class__.__name__} only supports the 'plotly' backend.")

        if not len(set([a:=len(x), b:=len(y), c:=len(ev_it)]))==1:
            raise ValueError(f"The lengths of x, y and ev_it have to be the same. Got {a}, {b} and {c}.")

        # default function is Unity    
        self._f = f if f is not None else Unity(ev_it.t)

        # used to restore original data after performing cuts
        self._original_data = (np.array(x), np.array(y), ev_it)
        self._xdata, self._ydata, self._ev_it = self._original_data
        
        # used to keep track of indices wrt original data (even after cuts)
        self._orig_inds = np.arange(len(ev_it))
        self._selection = []

        # create a Viewer for scatter and event preview plots
        self.scatter = Viewer(**kwargs)
        self.preview = Viewer(**{k:kwargs[k] for k in ["width", "height", "template", "backend"] if k in kwargs.keys()})
        
        # setup functionality of scatter plot
        self.scatter.add_scatter(x=self._xdata, y=self._ydata, name="scatter")
        self.scatter.fig_widget.fig.data[0].on_click(self._click_fnc)
        self.scatter.fig_widget.fig.data[0].on_selection(self._select_fnc)
        self.scatter.fig_widget.fig.data[0].on_deselect(self._deselect_fnc)
        self.scatter.fig_widget.fig.update_layout(hovermode = "closest")
        self.scatter.show_legend(False)
        
        # add buttons (note that we cannot use the default buttons of Viewer here because
        # it would not work)
        self._cut_button = widgets.Button(description="cut selected", tooltip="cut selected")
        self._cut_button.on_click(self._cut)
        self._undo_button = widgets.Button(description="undo all cuts", tooltip="undo all cuts")
        self._undo_button.on_click(self._undo) 

        # setup layout and display
        display(
            widgets.VBox(
                [
                    widgets.HBox([self._cut_button, self._undo_button]),
                    widgets.HBox([self.scatter.fig_widget.fig, self.preview.fig_widget.fig])
                ]
            )
        )

    def _click_fnc(self, trace, points, state):
        ind = points.point_inds[0]
        # get event and timestamp from event iterator and the preview details
        ev = self._ev_it.grab(ind)
        ts = self._ev_it.timestamps[ind]
        d = self._f.preview(ev)

        # build string from timestamp and update xlabel
        tsstr = np.array(ts, dtype="datetime64[us]").astype(datetime.datetime)[()].strftime('%d-%b-%Y, %H:%M:%S')

        # remove y-axis label
        if "axes" in d:
            if "yaxis" in d["axes"]:
                if "label" in d["axes"]["yaxis"]:
                    d["axes"]["yaxis"]["label"] = ""
        # add x-axis info
        if "axes" not in d: 
            d["axes"] = {}
        if "xaxis" not in d["axes"]: 
            d["axes"]["xaxis"] = {}
        if "label" not in d["axes"]["xaxis"]:
            d["axes"]["xaxis"]["label"] = f"Event {ind}, {tsstr} ({ts})"
        elif d["axes"]["xaxis"]["label"]:
            d["axes"]["xaxis"]["label"] += f"<br>Event {ind}, {tsstr} ({ts})"

        self.preview.plot(d)
        self.preview.show_legend()
            
    def _select_fnc(self, trace, points, selector):
        self._selection = points.point_inds
        
    def _deselect_fnc(self, trace, points):
        self._selection = []
        
    def _cut(self, b=None):
        if self._selection:
            # get indices which are NOT selected
            complement = np.ones(len(self._xdata), dtype=bool)
            complement[self._selection] = False
            
            # set xdata, ydata and ev_it to contain only the ones NOT selected
            self._xdata = self._xdata[complement]
            self._ydata = self._ydata[complement]
            self._ev_it = self._ev_it[:, complement]
            
            # update scatter plot
            self.scatter.update_scatter(name="scatter", x=self._xdata, y=self._ydata)
            
            # save surviving indices in terms of original indices
            self._orig_inds = self._orig_inds[complement]
    
    def _undo(self, b=None):
        # revert xdata, ydata and ev_it back to their original data and draw the changes
        self._xdata, self._ydata, self._ev_it = self._original_data
        self.scatter.update_scatter(name="scatter", x=self._xdata, y=self._ydata)
        for i in range(self._ev_it.n_channels):
            self.preview.update_line(name=f"channel {i}", x=[], y=[])
        
    @property
    def selected_inds(self):
        """Returns the indices (of the original x-y data) currently selected in the scatter plot."""
        return self._orig_inds[self._selection].tolist()
    
    @property
    def selected_events(self):
        """Returs an iterator of events currently selected in the scatter plot."""
        return self._ev_it[:, self._selection]