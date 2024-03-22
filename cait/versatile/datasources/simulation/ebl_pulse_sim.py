import numpy as np

from ..datasourcebase import DataSourceBaseClass
from ...iterators import IteratorBaseClass, PulseSimIterator

class EBPulseSim(DataSourceBaseClass):

    def __init__(self, baselines: IteratorBaseClass, sev: np.ndarray, pulse_heights: np.ndarray):
        if not isinstance(baselines, IteratorBaseClass):
            raise TypeError(f"Input argument 'baselines' has to be of type 'IteratorBaseClass', not '{type(baselines)}'.")
        
        if baselines.uses_batches:
            raise Exception
        
        if not len(baselines) == pulse_heights.shape[-1]:
            raise Exception
        
        self._baselines = baselines
        self._sev = sev
        self._phs = pulse_heights

    def get_event_iterator(self):
        return PulseSimIterator(self)

    @property
    def baselines(self):
        return self._baselines
    
    @property
    def sev(self):
        return self._sev
    
    @property
    def pulse_heights(self):
        return self._phs
    
