from typing import Union, List

import numpy as np

from .iteratorbase import IteratorBaseClass

# TODO: implement
class PulseSimIterator(IteratorBaseClass):
    def __init__(self, sim, channels: Union[int, List[int]] = None, inds: Union[int, List[int]] = None):
        super().__init__()

        self._sim = sim
        
        if channels is None: channels = list(range(sim.baselines.n_channels))
        self._channels = [channels] if isinstance(channels, int) else channels

        if inds is None: inds = np.arange(len(sim.baselines))
        self._inds = [inds] if isinstance(inds, int) else [int(i) for i in inds]

        # Save values to reconstruct iterator:
        self._params = {'sim': sim, 'channels': self._channels, 'inds': self._inds}
    
    def __len__(self):
        return len(self._inds)

    def __enter__(self):
        self._sim.baselines.__enter__()
        return self
    
    def __exit__(self, typ, val, tb):
        self._sim.baselines.__exit__(typ, val, tb)
    
    def __iter__(self):
        self._current_ind = 0
        self._sim.baselines.__iter__()
        return self

    def _next_raw(self):
        if self._current_ind < len(self._inds):
            out = next(self._sim.baselines) + self._sim.pulse_heights[self._current_ind]*self._sim.sev
            self._current_ind += 1

            return out
        else:
            raise StopIteration
        
    @property
    def timestamps(self):
        return self._sim.baselines.timestamps[self._params["inds"]]

    @property
    def uses_batches(self):
        return False # Just to be consistent with H5Iterator
    
    @property
    def n_batches(self):
        return len(self)
    
    @property
    def n_channels(self):
        return len(self._channels)
    
    @property
    def _slice_info(self):
        return (self._params, ('channels', 'inds'))