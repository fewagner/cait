from functools import partial
import numpy as np

from ..functionbase import FncBaseClass
from ...functions.trigger.trigger_of import filter_chunk, trigger_of
from ...functions.trigger.trigger_zscore import zscore_chunk, trigger_zscore

def is_same_function(f1, f2):
    return (f1.__module__ == f2.__module__) and (f1.__name__ == f2.__name__)

class TriggerSurvival(FncBaseClass):
    """
    Function that checks whether or not a given event would have survived triggering.

    :param trigger_fnc: The trigger function to use. Has to have function signature ``f(event: np.ndarray) -> (trigger_inds: list, trigger_vals: list)``.
    :type trigger_fnc: callable
    :param target_ind: The index on the voltage trace where the event (maximum) was placed, i.e. where the trigger is expected to be found.
    :type target_ind: int
    :param tolerance_samples: Maximum number of samples that a trigger can deviate from ``target_ind`` such that it is still considered a trigger.
    :type tolerance_samples: int

    :return: Tuple ``(did_trigger, trigger_value, trigger_index)``. If no trigger was found (including tolerance), the tuple ``(False, 0, 0)`` is returned.
    :rtype: Tuple[bool, float, int]
    """
    def __init__(self,
                 trigger_fnc: callable,
                 target_ind: int,
                 tolerance_samples: int = 10):
        
        self._f = trigger_fnc
        self._ind = target_ind
        self._tol = tolerance_samples
        
    def __call__(self, event: np.ndarray):
        self._inds, vals = self._f(event)
        
        flag = [np.abs(ind - self._ind) <= self._tol for ind in self._inds]
        if any(flag):
            which = np.argmax(flag)
            return True, vals[which], self._inds[which]
        else:
            return False, 0, 0

    @property
    def batch_support(self) -> str:
        return 'none'

    def preview(self, event: np.ndarray) -> dict:
        survived = self(event)
        event = event - np.mean(event)
        x = np.arange(len(event))
        mine, maxe = np.min(event), np.max(event)
        
        l = {
            "event": [x, event],
            "target index": [ [self._ind]*2, [mine, maxe] ]
        }
        
        s = { "triggers": [self._inds, event[self._inds] if len(self._inds)>0 else []] }
        
        if isinstance(self._f, partial):
            if is_same_function(self._f.func, trigger_of):
                if all([kw in self._f.keywords.keys() for kw in ["of", "threshold"]]):
                    of = self._f.keywords["of"]
                    rl = 2*(len(of)-1) # record_length
                    threshold = self._f.keywords["threshold"]
                    
                    N = len(x)
                    filtered_event = filter_chunk(event, of, rl)
                    x_filtered = x[rl:-rl]

                    l = l | {
                        "filtered event": [x_filtered, filtered_event],
                        "search_window": [
                            [rl]*2+[None]+[N-2*rl]*2+[None]+[N-rl]*2,
                            [mine, maxe, None, mine, maxe, None, mine, maxe]],
                        "threshold": [ [rl, N-rl], [threshold]*2 ],
                    }
            if is_same_function(self._f.func, trigger_zscore):
                if all([kw in self._f.keywords.keys() for kw in ["record_length", "threshold"]]):
                    rl = self._f.keywords["record_length"]
                    threshold = self._f.keywords["threshold"]
                    
                    N = len(x)
                    filtered_event = zscore_chunk(event, rl)
                    x_filtered = x[rl:-rl]
                    
                    mine, maxe = np.min(filtered_event), np.max(filtered_event)
                    
                    l["target index"] = [ [self._ind]*2, [mine, maxe] ]
                    l = l | {
                        "filtered event": [x_filtered, filtered_event],
                        "search_window": [
                            [rl]*2+[None]+[N-2*rl]*2+[None]+[N-rl]*2,
                            [mine, maxe, None, mine, maxe, None, mine, maxe]],
                        "threshold": [ [rl, N-rl], [threshold]*2 ],
                    }

        return dict(line=l, scatter=s)