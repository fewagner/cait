import numpy as np

from .abstract_functions import FncBaseClass
from .event_functions import BoxCarSmoothing, RemoveBaseline
from .fit_functions import FitBaseline

#### NOT YET FINISHED ####
class CalcMP(FncBaseClass):
    def __init__(self, 
                 peak_bounds: tuple = None,
                 box_car_smoothing: dict = {'length': 50}, 
                 fit_baseline: dict = {'model': 0, 'where': 8, 'xdata': None}):
        
        if peak_bounds is None:
            self._peak_bounds = slice(None, None, None)
        elif type(peak_bounds) is tuple and len(peak_bounds) == 2:
            self._peak_bounds = slice(peak_bounds[0], peak_bounds[1])
        else:
            raise ValueError(f"'{peak_bounds}' is not a valid interval.")
        
        self._box_car_smoothing = BoxCarSmoothing(**box_car_smoothing)
        self._remove_baseline = RemoveBaseline(**fit_baseline)
        self._get_offset = FitBaseline(model=0, where=8)
    
    def __call__(self, event):
        self._offset_MP = self._get_offset(event)
    
#### NOT YET FINISHED ####
class CalcAddMP(FncBaseClass):
    def __init__(self):
        ...

    def __call__(self, event):
        ...