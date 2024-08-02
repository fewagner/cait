import os

from typing import List, Union, Tuple

import numpy as np

import cait.versatile as vai

class TriggerCollectionMixin:
    """
    A mixin clas ....
    """

    def trigger_coincidence(self,
                            filedict: dict,
                            interval: Tuple[float] = None,
                            sigma: Union[float, List[float]] = 5,
                            reuse_triggers: bool = False,
                            copy_events: bool = False,
                            **kwargs
                           ):
            """
            Explain, what the meaning of the newly created datasets in the DataHandler is
            
            :param filedict: 
            :type filedict: dict
            :param interval: 
            :type interval: Tuple[float], optional
            :param sigma: 
            :type sigma: Union[float, List[float]], optional
            :param reuse_triggers: 
            :type reuse_triggers: bool, optional
            :param copy_events: 
            :type copy_events: bool, optional
            :param kwargs: Additional keyword arguments forwarded to ``cait.versatile.trigger_zscore``.
            :type kwargs: Any
            
            **Example:**
            ::
                filedict = {
                    "par": 'file.par',
                    "tp": ['file.test_stamps', 'file.dig_stamps'],
                    "ch4": ["file_ch4.csmpl", "5"],
                    "ch5": ["file_ch5.csmpl", "5"],
                    "ch6": ["file_ch6.csmpl", "7"],
                }
                
                # perform initial triggering and save triggers in DataHandler
                # In this step, the events are built according to the record length of the DataHandler
                dh.trigger_coincidence(filedict,
                                       # n_triggers=1000 # example of how to only trigger the first 1000 events
                                       )
                                       
                # Inspect coincidence histogram (coincidences of first two channels, here of a doubleTES module)
                double_tes_flag = np.prod(dh.get("event_building", "trigger_flag", [0,1]), axis=0, dtype=bool)
                ts = dh.get("event_building", "trigger_timestamps", [0,1])[:, double_tes_flag]
                diffs = np.diff(ts, axis=0)
                vai.Histogram(diffs/1000, bins=100, xlabel="delta (ms)")
                
                # Decide on a coincidence interval from the histogram, e.g. +- 5 ms
                # Do the event building step again (note that raw triggers from previous call can be used)
                dh.trigger_coincidence(filedict,
                                       interval=(-5000, 5000), # microseconds
                                       reuse_triggers=True,
                                       # n_triggers=1000 # example of how to only trigger the first 1000 events
                                       )
            """
            stream_arg = [filedict.get("par")] + filedict.get("tp", []) + [v[0] for k,v in filedict.items() if k.lower().startswith("ch")]
            stream = vai.Stream("cresst", stream_arg)
            tp_keys = dict()
            for ch_name in stream.keys:
                for k,v in filedict.items():
                    if k.lower().startswith("ch") and os.path.basename(os.path.splitext(v[0])[0]).endswith(ch_name):
                        tp_keys[ch_name] = v[1]
            
            sigmas = [sigma]*len(stream.keys) if isinstance(sigma, (int, float)) else sigma

            rec_window_coinc = (-stream.dt_us*self.record_length//4, stream.dt_us*self.record_length//4)
            if interval is None: 
                interval = rec_window_coinc

            trigger_ts, trigger_phs = [], []

            for i, (key, sigma) in enumerate(zip(stream.keys, sigmas)):
                if reuse_triggers:
                    if not (self.exists("triggers", f"ts_{key}") and self.exists("triggers", f"ph_{key}")):
                        raise KeyError(f"To reuse triggers, datasets 'ts_{key}' and 'ph_{key}' must exist in the 'triggers' group.")

                    ts = list(self.get("triggers", f"ts_{key}"))
                    ph = list(self.get("triggers", f"ph_{key}"))

                else:
                    ind, ph = vai.trigger_zscore(stream[key],
                                                 record_length=self.record_length,
                                                 threshold=sigma,
                                                 **kwargs)
                    ts = stream.time[ind]

                    # save trigger timestamps and trigger heights. Can be used in subsequent calls to avoid going through the trigger process again if just the interval argument for building events changes
                    self.set("triggers", 
                             **{f"ts_{key}": np.array(ts)}, 
                             dtype=np.int64, 
                             overwrite_existing=True)
                    self.set("triggers", 
                             **{f"ph_{key}": np.array(ph)}, 
                             dtype=np.float32, 
                             overwrite_existing=True)

                # if testpulse information is provided, trigger timestamps within one record window around testpulse timestamps are counted as such
                if "tp" in filedict.keys():
                    tp_ts = stream.tp_timestamps[tp_keys[key]]
                    *_, outside = vai.timestamp_coincidence(tp_ts, ts, rec_window_coinc)
                    ts = list(np.array(ts)[outside])
                    ph = list(np.array(ph)[outside])

                trigger_ts.append(ts)
                trigger_phs.append(ph)

            # build events
            event_ts = trigger_ts[0].copy()
            trigger_flag = [[True]*len(event_ts)]
            
            for i, ts in enumerate(trigger_ts[1:]):
                # determine coincidences
                inside, coinc_inds, outside = vai.timestamp_coincidence(event_ts, ts, interval)

                # all previous channels did not trigger for the newly found timestamps
                # Therefore, we add False in the end of their trigger flag (will be sorted later)
                for k in range(i+1):
                    trigger_flag[k].extend([False]*len(outside))

                # Build flag for the current channel. First initialize False in all existing spots.
                # The new ones all get True because they are definitely triggered (by construction)
                new_flag = np.array([False]*len(event_ts) + [True]*len(outside))
                # Add True at the correct spots (where already existing events are)
                new_flag[coinc_inds] = True
                trigger_flag.append(new_flag)

                # Merge new timestamps with existing ones
                new_ts = np.concatenate([event_ts, np.array(ts)[outside]])
                # Get the sort indices (needed to sort timestamps AND the flags)
                sortind = np.argsort(new_ts)
                # Sort timestamps
                event_ts = new_ts[sortind]
                
                # Sort flag (IMPORTANT: has to be done for ALL previous lists!)
                for j in range(i+2):
                    trigger_flag[j] = np.array(trigger_flag[j])[sortind].tolist()

            # save final timestamps and trigger flag after event building
            self.set("event_building", event_timestamps=np.array(event_ts), dtype=np.int64, overwrite_existing=True)
            self.set("event_building", trigger_flag=np.array(trigger_flag), dtype=bool, overwrite_existing=True)
            
            # also save the original trigger timestamps exactly like the trigger flag array
            # values of -1 indicate that the corresponding value does not exist
            # (because the channel didn't trigger separately for that event)
            original_ts = -1*np.ones(np.array(trigger_flag).shape, dtype=np.int64)
            original_ph = -1*np.ones(np.array(trigger_flag).shape, dtype=np.float32)
            for i, (t, p) in enumerate(zip(trigger_ts, trigger_phs)):
                original_ts[i, np.array(trigger_flag[i])] = np.array(t)
                original_ph[i, np.array(trigger_flag[i])] = np.array(p)
                
            self.set("event_building", trigger_timestamps=original_ts, dtype=np.int64, overwrite_existing=True)
            self.set("event_building", trigger_phs=original_ph, dtype=np.float32, overwrite_existing=True)

            if copy_events:
                # save events in events group
                self.include_event_iterator("events", stream.get_event_iterator(stream.keys, self.record_length, timestamps=event_ts))

                # do the same for testpulses if respective information is provided
                if "tp" in filedict.keys():
                    # save testpulses and tpas in separate groups (as they might be pulsed differently)
                    for key in stream.keys:
                        group_name = f"testpulses_{key}"
                        self.include_event_iterator(group_name, stream.get_event_iterator(key, self.record_length, timestamps=stream.tp_timestamps[tp_keys[key]]))
                        self.set(group_name, testpulseamplitude=stream.tpas[tp_keys[key]])