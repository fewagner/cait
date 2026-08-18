from functools import partial

import numpy as np

import cait.versatile as vai

from ..functionbase import ScalarFncBaseclass


def is_same_function(f1, f2):
    return (f1.__module__ == f2.__module__) and (f1.__name__ == f2.__name__)

class TriggerSurvival(ScalarFncBaseclass):
    """
    Function that checks whether or not a given event would have survived triggering.

    In the preview, the 'search_window'-box marks the area which is searched for samples exceeding the threshold. Parts outside the box are not searched because they can in general not be filtered correctly (c. f. :func:`cait.versatile.functions.trigger.triggerbase.trigger_base`). However, if a sample is found above threshold close to the right edge of the box, the maximum search is continued outside. 

    :param trigger_fnc: The trigger function to use. Has to have function signature ``f(event: np.ndarray) -> (trigger_inds: list, trigger_vals: list)``.
    :type trigger_fnc: callable
    :param target_ind: The index on the voltage trace where the event (maximum) was placed, i.e. where the trigger is expected to be found.
    :type target_ind: int
    :param tolerance_samples: Maximum number of samples that a trigger can deviate from ``target_ind`` such that it is still considered a trigger.
    :type tolerance_samples: int

    :return: Tuple ``(did_trigger, trigger_value, trigger_index)``. If no trigger was found (including tolerance), the tuple ``(False, 0, 0)`` is returned.
    :rtype: Tuple[bool, float, int]

    .. code-block:: python

        from functools import partial
        import scipy as sp
        import numpy as np

        import cait.versatile as vai

        # This example is not very meaningful. Usually, one would
        # superimpose pulses onto empty baselines and check if they
        # are triggered. To make this example self-contained, we
        # use the pulses ov MockData (not empty baselines).
        md = vai.MockData()
        rl = md.record_length
        sev, of = md.sev[0], md.of[0]
        it = vai.MockData(record_length=6*rl).get_event_iterator()[0]

        # Make a 'long' version of the SEV to be superimposed
        padded_sev = np.zeros(6*rl)
        padded_sev[3*rl:4*rl] = np.array(sev)

        # Simulate random pulse heights and superimpose traces
        sim_phs = sp.stats.uniform.rvs(size=len(it))
        chunk_iterator = vai.iterators.PulseSimIterator(
                it,
                sev=padded_sev,
                pulse_heights=sim_phs,
        )

        # Trigger function is trigger_of with a given threshold.
        # If a trigger is found within 10 samples around target_ind 
        # (the maximum of the SEV), the trigger is considered a 'survivor'
        f = vai.TriggerSurvival(
                trigger_fnc=partial(vai.trigger_of, of=of, threshold=0.1),
                target_ind=np.argmax(padded_sev),
                tolerance_samples=10
            )

        # Preview
        vai.Preview(chunk_iterator, f)

        # Calculate survival for all pulses
        trigger_flag, trigger_val, trigger_ind = vai.apply(f, chunk_iterator)

    .. image:: media/TriggerSurvival_preview.png
    """
    _outputs = [
        ("triggered", bool),
        ("trigger_val", float),
        ("trigger_ind", int),
    ]

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
        survived, *_ = self(event)
        event = np.atleast_2d(event - np.mean(event, axis=-1, keepdims=True))
        x = np.arange(event.shape[-1])
        mine, maxe = np.min(event), np.max(event)
        
        l = {
            **{"event" + (f"Ch{i}" if event.shape[0]>1 else ""): [x, ev] for i, ev in enumerate(event)},
            "target index": [ [self._ind]*2, [mine, maxe] ]
        }

        # Placeholder, will be overwritten by filtered traces if a known trigger function is used
        s = { "triggers": [ self._inds, event[..., self._inds] if len(self._inds)>0 else []] }

        # Now squeeze the array again (was previously only used to standardize
        # arrays for plotting etc.)
        event = np.squeeze(event)
        
        if isinstance(self._f, partial):
            if is_same_function(self._f.func, vai.functions.trigger.trigger_of.trigger_of):
                if all([kw in self._f.keywords.keys() for kw in ["of", "threshold"]]):
                    of = self._f.keywords["of"]
                    rl = 2*(len(of)-1) # record_length
                    threshold = self._f.keywords["threshold"]
                    add_kwargs = {k: v for k, v in self._f.keywords.items() if k not in ["of", "threshold"]}
                    
                    N = len(x)
                    filtered_event = vai.functions.trigger.trigger_of.filter_chunk(event, of, rl, **add_kwargs)
                    x_filtered = x[rl:-rl]

                    l = {**l, **{
                        "filtered event": [x_filtered, filtered_event],
                        "search_window": [
                            [rl, rl, N-2*rl, N-2*rl, rl],
                            [mine, maxe, maxe, mine, mine]
                        ],
                            #[rl]*2+[None]+[N-2*rl]*2+[None]+[N-rl]*2,
                            #[mine, maxe, None, mine, maxe, None, mine, maxe]],
                        "threshold": [ [rl, N-2*rl], [threshold]*2 ],
                    }}
                    s = { "triggers": [
                        self._inds, 
                        filtered_event[np.array(self._inds)-rl] if len(self._inds)>0 else []
                        ] }
            elif is_same_function(self._f.func, vai.functions.trigger.trigger_of.trigger_of2d):
                if all([kw in self._f.keywords.keys() for kw in ["of", "threshold"]]):
                    of = self._f.keywords["of"]
                    rl = 2*(np.array(of).shape[-1]-1) # record_length
                    threshold = self._f.keywords["threshold"]
                    add_kwargs = {k: v for k, v in self._f.keywords.items() if k not in ["of", "threshold"]}
                    
                    N = len(x)
                    filtered_event = vai.functions.trigger.trigger_of.filter_chunk_2d(event, of, rl, **add_kwargs)
                    x_filtered = x[rl:-rl]

                    l = {**l, **{
                        "filtered event": [x_filtered, filtered_event],
                        "search_window": [
                            [rl, rl, N-2*rl, N-2*rl, rl],
                            [mine, maxe, maxe, mine, mine]
                        ],
                            #[rl]*2+[None]+[N-2*rl]*2+[None]+[N-rl]*2,
                            #[mine, maxe, None, mine, maxe, None, mine, maxe]],
                        "threshold": [ [rl, N-2*rl], [threshold]*2 ],
                    }}
                    s = { "triggers": [
                        self._inds, 
                        filtered_event[np.array(self._inds)-rl] if len(self._inds)>0 else []
                        ] }
            elif is_same_function(self._f.func, vai.functions.trigger.trigger_zscore.trigger_zscore):
                if all([kw in self._f.keywords.keys() for kw in ["record_length", "threshold"]]):
                    rl = self._f.keywords["record_length"]
                    threshold = self._f.keywords["threshold"]
                    
                    N = len(x)
                    filtered_event = vai.functions.trigger.trigger_zscore.zscore_chunk(event, rl)
                    x_filtered = x[rl:-rl]
                    
                    mine, maxe = np.min(filtered_event), np.max(filtered_event)
                    
                    l["target index"] = [ [self._ind]*2, [mine, maxe] ]
                    l = {**l, **{
                        "filtered event": [x_filtered, filtered_event],
                        "search_window": [
                            [rl, rl, N-2*rl, N-2*rl, rl],
                            [mine, maxe, maxe, mine, mine]
                        ],
                        "threshold": [ [rl, N-2*rl], [threshold]*2 ],
                    }}
                    s = { "triggers": [
                        self._inds, 
                        filtered_event[np.array(self._inds)-rl] if len(self._inds)>0 else []
                        ] }

        return dict(line=l, scatter=s, axes=dict(xaxis=dict(label=f"Survived trigger: {survived}")))