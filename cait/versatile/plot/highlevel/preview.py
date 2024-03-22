from typing import Callable

from ..viewer import Viewer
from ...iterators.iteratorbase import IteratorBaseClass
from ...eventfunctions.processing.helper import Unity

# Has no test case (yet)
class Preview(Viewer):
    """
    Class for inspecting the behavior of functions which were subclassed from :class:`abstract_functions.FncBaseClass`.
    Can also be used to display single events if no function is specified.

    :param events: An iterable of events. Can be e.g. :class:`IteratorBaseClass`, a 2d :class:`numpy.ndarray` or a list of List[float].
    :type events: IteratorBaseClass
    :param f: The function to be inspected, already initialized with the values that should stay fixed throughout the inspection. Defaults to Unity (which means that just the events of the iterable will be displayed)
    :type f: :class:`abstract_functions.FncBaseClass`
    :param kwargs: Keyword arguments for `Viewer`.
    :type kwargs: Any
    """
    def __init__(self, events: IteratorBaseClass, f: Callable = Unity(), **kwargs):
        #viewer_kwargs = {k:v for k,v in kwargs.items() if k in ["backend","template","width","height"]}
        #for k in ["backend","template","width","height"]: kwargs.pop(k, None)
        super().__init__(data=None, show_controls=True, **kwargs)

        if isinstance(events, IteratorBaseClass) and events.uses_batches:
            raise NotImplementedError("Iterators that return batches are not supported by Preview.")

        self._add_button("Next", self._update_plot, "Show next event.", key="n")

        self.f = f
        self.events = iter(enumerate(events))
        
        self.start()
    
    def _update_plot(self, b=None):
        try:
            ind, ev = next(self.events)
            d =  self.f.preview(ev)

            # Add event index to y-label
            if d.get("axes") is None: d["axes"] = dict()
            if d["axes"].get("yaxis") is None: d["axes"]["yaxis"] = dict()
            if d["axes"]["yaxis"].get("label") is None: 
                d["axes"]["yaxis"]["label"] = ""
            else:
                d["axes"]["yaxis"]["label"] += ", "
            
            d["axes"]["yaxis"]["label"] += f"event {ind}"

            # Plot
            self.plot(d)

        except StopIteration: 
            self.close()
        except:
            self.close()
            raise
    
    def start(self):
        """
        Show the plot and start iterating over the events.
        """
        self._update_plot()
        self.show()