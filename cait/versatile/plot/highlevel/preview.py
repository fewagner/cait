import datetime
from typing import Callable

import numpy as np

from ...eventfunctions.processing.helper import Unity
from ...iterators.iteratorbase import IteratorBaseClass
from ..viewer import Viewer


# Has no test case (yet)
class Preview(Viewer):
    """
    Class for inspecting the behavior of functions which were subclassed from :class:`abstract_functions.FncBaseClass`.
    Can also be used to display single events if no function is specified.

    :param events: An iterable of events. Can be e.g. :class:`IteratorBaseClass`, a 2d :class:`numpy.ndarray` or a list of List[float].
    :type events: IteratorBaseClass
    :param f: The function to be inspected, already initialized with the values that should stay fixed throughout the inspection. Defaults to Unity (which means that just the events of the iterable will be displayed)
    :type f: :class:`cait.versatile.eventfunctions.functionbase.FncBaseClass`
    :param show_ev_time: If True, event number and time are shown in the y-axis label. Defaults to True.
    :type show_ev_time: bool, optional
    :param kwargs: Keyword arguments for `Viewer`.
    :type kwargs: Any

    **Example Preview:**

    .. code-block:: python
    
        import cait.versatile as vai

        # Get events from mock data (and remove baseline)
        it = vai.MockData().get_event_iterator().with_processing(vai.RemoveBaseline())[0]

        # View pulses
        vai.Preview(it)

        # View pulses starting from index 37
        vai.Preview(it[:, 37:])
    """
    def __init__(self, events: IteratorBaseClass, f: Callable = None, show_ev_time = True, **kwargs):
        #viewer_kwargs = {k:v for k,v in kwargs.items() if k in ["backend","template","width","height"]}
        #for k in ["backend","template","width","height"]: kwargs.pop(k, None)
        super().__init__(data=None, show_controls=True, **kwargs)

        if isinstance(events, IteratorBaseClass) and events.uses_batches:
            raise NotImplementedError("Iterators that return batches are not supported by Preview.")

        self._add_button("❮", self._prev, "Show previous event.", key="b")
        self._add_button("❯", self._next, "Show next event.", key="n")

        self._f = f if f is not None else Unity(events.t)
        self._current_ind = 0
        self._events = events
        self._show_ev_time = show_ev_time
        
        self.start()

    def _next(self, b=None):
        if self._current_ind > len(self._events)-2:
            self.close()
        else:
            self._current_ind += 1
            self._update_plot()

    def _prev(self, b=None):
        if self._current_ind > 0:
            self._current_ind -= 1
            self._update_plot()
    
    def _update_plot(self):
        try:
            ev = self._events.grab(self._current_ind)
            ts = self._events.timestamps[self._current_ind]

            d = self._f.preview(ev)
            tsstr = np.array(ts, dtype="datetime64[us]").astype(datetime.datetime)[()].strftime('%d-%b-%Y, %H:%M:%S')

            # Add event index to y-label
            if d.get("axes") is None: d["axes"] = dict()
            if d["axes"].get("yaxis") is None: d["axes"]["yaxis"] = dict()
            if d["axes"]["yaxis"].get("label") is None: 
                d["axes"]["yaxis"]["label"] = ""
            elif self._show_ev_time:
                d["axes"]["yaxis"]["label"] += ", "
            if self._show_ev_time:
                d["axes"]["yaxis"]["label"] += f"Event {self._current_ind}, {tsstr}"

            # Plot
            self.plot(d)
            self.show_legend()
        except:
            self.close()
            raise
        
    def start(self):
        """
        Show the plot and start iterating over the events.
        """
        self._update_plot()
        self.show()