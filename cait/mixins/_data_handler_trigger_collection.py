from typing import List, Union, Tuple

import numpy as np

import cait.versatile as vai

class TriggerCollectionMixin:
    """
    A mixin clas ....
    """

    def trigger_multiTES_type1(self,
                             fpath: List[str],
                             path_par: str,
                             record_length: int,
                             sigma: Union[float, List[float]],
                             interval: Tuple[float],
                             path_tp: List[str, str, List[int]] = None,
                             reuse_triggers: bool = True,
                             copy_events: bool = False,
                             **kwargs
                             ):
        """
        doc strings will be here
        """
        stream_arg = fpath + [path_par] + (path_tp[:2] if path_tp is not None else [])
        sigmas = [sigma]*len(fpath) if isinstance(sigma, float) else sigma
        stream = vai.Stream(hardware="cresst", stream_arg)
        
        rec_window_coinc = (-stream.dt_us*record_length/4, stream.dt_us*record_length/4)
        trigger_ts, trigger_phs = [], []
        
        for i, (key, sigma) in enumerate(zip(stream.keys, sigmas)):
            if reuse_triggers and self.exists("triggers", f"trigger_ts_{key}") and self.exists("triggers", f"trigger_phs_{key}"):
                trigger_ts.append(list(self.get("triggers", f"trigger_ts_{key}")))
                phs.append(list(self.get("triggers", f"trigger_phs_{key}")))
            else:
                ind, ph = vai.trigger_zscore(stream[key],
                                             record_length=record_length,
                                             sigma=sigma,
                                             **kwargs)
                ts = stream.time[ind]
    
                self.set("triggers", f"trigger_ts_{key}"=np.array(ts, dtype=np.int64), f"trigger_phs_{key}"=np.array(ph), overwrite_existing=True)
                
            if path_tp is not None:
                tp_ts = stream.tp_timestamps[path_tp[2][i]]
                *_, outside = vai.timestamp_coincidence(tp_ts, ts, rec_window_coinc)
                ts = list(np.array(ts)[outside])
                ph = list(np.array(ph)[outside])
                
            trigger_ts.append(ts)
            trigger_phs.append(ph)

        # build events
        event_ts = trigger_ts[0]
        for ts in trigger_ts[1:]:
            inside, coinc_inds, outside = vai.timestamp_coincidence(event_ts, ts, interval)
            event_ts = sorted(np.array(event_ts)[coinc_inds])

        self.set("triggers", "timestamps"=np.array(event_ts, dtype=np.int64), overwrite_existing=True)

        
        
                                           
                                           
