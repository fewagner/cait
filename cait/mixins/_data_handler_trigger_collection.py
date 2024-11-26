import os
from functools import partial
from typing import List, Union, Tuple

import numpy as np

import cait.versatile as vai
         
    
# Helper function that is used in both trigger_of and trigger_zscore
def _trigger_helper(dh, 
                    stream, 
                    trigger_channels,
                    slave_channels, 
                    testpulse_channels, 
                    copy_events, 
                    reuse_triggers,
                    interval,
                    trigger_fncs,
                    n_noise,
                    name_appendix
                    ):
    
    # Input validation
    if not all([x in stream.keys for x in trigger_channels]):
            raise KeyError(f"All 'trigger_channels' have to be valid channel names. Available: {stream.keys}")
            
    if ( slave_channels is not None ) and ( not all([x in stream.keys for x in slave_channels]) ):
        raise KeyError(f"All 'slave_channels' have to be valid channel names. Available: {stream.keys}")

    if ( testpulse_channels is not None ) and ( not all([x in stream.tp_keys for x in testpulse_channels]) ):
        raise KeyError(f"All 'testpulse_channels' have to be valid channel names. Available: {stream.tp_keys}")

    if testpulse_channels is not None:
        if len(trigger_channels) + (0 if slave_channels is None else len(slave_channels)) != len(testpulse_channels):
            raise ValueError(f"Testpulse channels are required for all channels (including slave channels). I.e. len(testpulse_channels)' must match 'len(trigger_channels)+len(slave_channels)'. Received {len(testpulse_channels)} and {len(trigger_channels)}+{0 if slave_channels is None else len(slave_channels)}")

    all_channels = trigger_channels + ([] if slave_channels is None else slave_channels)

    # Triggering channels
    for i, key in enumerate(trigger_channels):
        if reuse_triggers:
            if not (dh.exists(f"triggers-{name_appendix}", f"ts_{key}") and dh.exists(f"triggers-{name_appendix}", f"ph_{key}")):
                raise KeyError(f"To reuse triggers, datasets 'ts_{key}' and 'ph_{key}' must exist in the 'triggers-{name_appendix}' group.")
            print(f"Read existing triggers for channel {key}.")

        else:
            with stream: # this keeps the stream file opened (performance increase)
                print(f"Triggering channel {key} ...")
                ind, ph = trigger_fncs[i](stream[key])
                
            ts = stream.time[ind]

            # save trigger timestamps and trigger heights. 
            # Can be used in subsequent calls to avoid going through the trigger process 
            # again if just the interval argument for building events changes
            dh.set(f"triggers-{name_appendix}", 
                     **{f"ts_{key}": np.array(ts)}, 
                     dtype=np.int64, 
                     overwrite_existing=True)
            dh.set(f"triggers-{name_appendix}", 
                     **{f"ph_{key}": np.array(ph)}, 
                     dtype=np.float32, 
                     overwrite_existing=True)

    trigger_ts = [ list(dh.get(f"triggers-{name_appendix}", f"ts_{key}")) for key in trigger_channels]
    trigger_ph = [ list(dh.get(f"triggers-{name_appendix}", f"ph_{key}")) for key in trigger_channels]

    if testpulse_channels is not None:
        for key in testpulse_channels:
            if reuse_triggers and dh.exists(f"triggers-{name_appendix}", f"tp_ts_{key}"):
                print(f"Read existing testpulses for channel {key}.")
            else:
                dh.set(f"triggers-{name_appendix}", 
                         **{f"tp_ts_{key}": np.array(stream.tp_timestamps[key])},
                         dtype=np.int64,
                         overwrite_existing=True)
                dh.set(f"triggers-{name_appendix}", 
                         **{f"tpas_{key}": np.array(stream.tpas[key])},
                         dtype=np.float32,
                         overwrite_existing=True)

        tp_ts = [list(dh.get(f"triggers-{name_appendix}", f"tp_ts_{key}")) for key in testpulse_channels]
        tpas = [list(dh.get(f"triggers-{name_appendix}", f"tpas_{key}")) for key in testpulse_channels]

    else:
        tp_ts, tpas = None, None

    print("Building events ...")
    event_ts, trig_flag, orig_ts, orig_ph, all_tp_ts, final_tpas = vai.event_building(
                                                                trigger_ts=trigger_ts, 
                                                                trigger_phs=trigger_ph, 
                                                                record_length=dh.record_length, 
                                                                dt_us=stream.dt_us,
                                                                tp_ts=tp_ts,
                                                                tpas=tpas,
                                                                n_slave_ch=0 if slave_channels is None else len(slave_channels),
                                                                interval=interval)

    # save final timestamps and trigger flag after event building
    dh.set(f"event_building-{name_appendix}", event_timestamps=event_ts, dtype=np.int64, overwrite_existing=True)
    dh.set(f"event_building-{name_appendix}", trigger_flag=trig_flag, dtype=bool, overwrite_existing=True)

    dh.set(f"event_building-{name_appendix}", trigger_timestamps=orig_ts, dtype=np.int64, overwrite_existing=True)
    dh.set(f"event_building-{name_appendix}", trigger_phs=orig_ph, dtype=np.float32, overwrite_existing=True)

    if testpulse_channels is not None:
        dh.set(f"event_building-{name_appendix}", tp_ts=all_tp_ts, dtype=np.int64, overwrite_existing=True)
        dh.set(f"event_building-{name_appendix}", tpas=final_tpas, dtype=np.float32, overwrite_existing=True)
        
    if n_noise > 0:
        inds = stream.time.timestamp_to_ind(dh.get(f"event_building-{name_appendix}", "event_timestamps"))
        noise_inds = vai.sample_noise(inds.tolist(), dh.record_length, n_samples=n_noise)
        dh.set(f"event_building-{name_appendix}", noise_ts=stream.time[noise_inds], dtype=np.int64, overwrite_existing=True)

    if copy_events:
        # save events in events group
        if dh.exists("events"): 
            raise Exception("Could not copy events to DataHandler because the group 'events' already exists. To delete it, use 'dh.drop('events')'.")

        print("Writing events to DataHandler ...")
        if len(event_ts)>0:
            dh.include_event_iterator("events", 
                                      stream.get_event_iterator(
                                          all_channels, 
                                          dh.record_length, 
                                          timestamps=event_ts
                                      ))
        else:
            print("No events found to write to DataHandler.")

        # do the same for testpulses if respective information is provided
        if testpulse_channels is not None:
            if dh.exists("testpulses"): 
                raise Exception("Could not copy events to DataHandler because the group 'testpulses' already exists. To delete it, use 'dh.drop('testpulses')'.")
            # make sure all timestamps written in the tp file are actually within the stream file (and their voltage traces can be read completely)
            valid_tp_flag = all_tp_ts < stream.time[-3*dh.record_length//4]
            if not all(valid_tp_flag): 
                print("One or more testpulses could not be included because they fall (partially) outside the stream's range!!")

            # save testpulses and tpas
            print("Writing testpulses to DataHandler ...")
            tp_ts = all_tp_ts[valid_tp_flag]
            if len(tp_ts)>0:
                dh.include_event_iterator("testpulses", 
                                          stream.get_event_iterator(
                                              all_channels, 
                                              dh.record_length, 
                                              timestamps=tp_ts
                                          ))
                dh.set("testpulses", testpulseamplitude=final_tpas[..., valid_tp_flag])
            else:
                print("No testpulses found to write to DataHandler.")
            
        if n_noise>0:
            print("Writing noise to DataHandler ...")
            noise_ts = dh.get(f"event_building-{name_appendix}", "noise_ts")
            if len(noise_ts)>0:
                dh.include_event_iterator("noise", 
                                          stream.get_event_iterator(
                                              all_channels, 
                                              dh.record_length, 
                                              timestamps=noise_ts
                                          ))
            else:
                print("No noise found to write to DataHandler.")

class TriggerCollectionMixin:
    """
    A mixin class with convenience functions concerning triggering and event building, e.g. for CRESST doubleTES analysis.
    """
    def trigger_zscore(self,
                       stream: vai.datasources.stream.streambase.StreamBaseClass,
                       trigger_channels: List[str],
                       thresholds: Union[float, List[float]] = 5,
                       slave_channels: List[str] = None,
                       testpulse_channels: List[str] = None,
                       copy_events: bool = False,
                       reuse_triggers: bool = False,
                       interval: Tuple[float] = None,
                       n_noise: int = 0,
                       **kwargs
                      ):
        """
        Trigger stream channels from arbitrary hardware using a moving z-score trigger and build events from trigger timestamps (of multiple channels) and exclude testpulses if the respective information is provided.

        The stream channels specified by ``trigger_channels`` are triggered. If some channels are not triggered but read out in coincidence (i.e. as 'slave' channels), you can specify their channel names using ``slave_channels``. 

        Events are built as follows: Starting from the first channel's trigger timestamps, the remaining channels' triggers are checked to be in coincidence with already existing timestamps. The default coincidence window (if ``interval=None``), is ``-+dt_us*record_length//4`` but can be adapted as needed.

        If you provide ``testpulse_channels``, triggers within a record window of a testpulse are treated as testpulses. Note that you have to provide testpulse information for all channels INCLUDING 'slave' channels.

        :param stream: The stream object including the channels that you want to trigger.
        :type stream: vai.datasources.stream.streambase.StreamBaseClass
        :param trigger_channels: The list of channel names to be triggered. Have to be present in ``stream.keys``.
        :type trigger_channels: List[str]
        :param thresholds: A list of trigger thresholds (in sigmas) for each channel. If only a float is provided, it is used for all channels. Defaults to 5 sigmas
        :type thresholds: Union[float, List[float]], optional
        :param slave_channels: A list of channel names to be read out as 'slaves'. Have to be present in ``stream.keys``. Defaults to None
        :type slave_channels: List[str], optional
        :param testpulse_channels: A list of channel names to be used as testpulses. Have to be present in ``stream.tp_timestamps.keys``. Defaults to None
        :type testpulse_channels: List[str], optional
        :param copy_events: If true, the voltage traces of the events which were built are saved in the DataHandler (i.e. copied from the stream files). Defaults to False.
        :type copy_events: bool, optional
        :param reuse_triggers: If true, the triggers from a previous call of this function (which were saved in the DataHandler) are reused and only the event building is performed again (possibly with a different coincidence interval). Defaults to False.
        :type reuse_triggers: bool, optional
        :param interval: The coincidence interval for event building in microseconds, i.e. if a trigger lies within the specified interval around a trigger of another channel, they are collected to represent one event. Defaults to ``-+dt_us*record_length//4``.
        :type interval: Tuple[float], optional
        :param n_noise: The number of empty noise traces to include. Defaults to 0, i.e. no noise is included.
        :type n_noise: int, optional
        :param kwargs: Additional keyword arguments forwarded to :func:`cait.versatile.trigger_zscore`.
        :type kwargs: Any

        **Example:**

        .. code-block:: python
        
            import cait as ai
            import cait.versatile as vai

            # Construct stream object
            stream = vai.Stream(hardware="vdaq2", src="path/to/stream_file.bin")

            print(f"Available channels: {stream.keys}")
            print(f"Available TP channels: {stream.tp_timestamps}")

            # Construct DataHandler
            dh = ai.DataHandler(record_length=2**13, nmbr_channels=2, sample_frequency=stream.sample_frequency)
            dh.set_filepath(path_h5="folder_name/", fname="z-score-triggered", appendix=False)
            dh.init_empty()

            # Trigger
            dh.trigger_zscore(stream, 
                              trigger_channels=["ADC1", "ADC2"],
                              testpulse_channels=["DAC1", "DAC3"],
                              copy_events=True)
        """
        
        # Allow string input if only one channel
        trigger_channels = [trigger_channels] if isinstance(trigger_channels, str) else trigger_channels
        slave_channels = [slave_channels] if isinstance(slave_channels, str) else slave_channels
        testpulse_channels = [testpulse_channels] if isinstance(testpulse_channels, str) else testpulse_channels
        
        # Make sure that thresholds are a list (allow scalar input, to be used for all channels)
        thresholds = [thresholds]*len(trigger_channels) if isinstance(thresholds, (int, float)) else thresholds
        
        if len(thresholds) != len(trigger_channels):
            raise ValueError(f"You need to provide as many thresholds as trigger channels. Received {len(thresholds)} and {len(trigger_channels)}")
        
        trigger_fncs = [partial(vai.trigger_zscore, 
                               threshold=thresh, 
                               record_length=self.record_length, 
                               **kwargs) 
                       for thresh in thresholds]
        
        _trigger_helper(self, stream, trigger_channels, slave_channels, testpulse_channels, copy_events, reuse_triggers,
                        interval, trigger_fncs, n_noise, "z-score")
        
    def trigger_of(self,
                   stream: vai.datasources.stream.streambase.StreamBaseClass,
                   trigger_channels: List[str],
                   thresholds: List[float],
                   slave_channels: List[str] = None,
                   testpulse_channels: List[str] = None,
                   copy_events: bool = False,
                   reuse_triggers: bool = False,
                   interval: Tuple[float] = None,
                   of: np.ndarray = None,
                   n_noise: int = 0,
                   **kwargs
                   ):
        """
        Trigger stream channels from arbitrary hardware using a moving optimum filter trigger and build events from trigger timestamps (of multiple channels) and exclude testpulses if the respective information is provided.

        The stream channels specified by ``trigger_channels`` are triggered. If some channels are not triggered but read out in coincidence (i.e. as 'slave' channels), you can specify their channel names using ``slave_channels``. 

        Events are built as follows: Starting from the first channel's trigger timestamps, the remaining channels' triggers are checked to be in coincidence with already existing timestamps. The default coincidence window (if ``interval=None``), is ``-+dt_us*record_length//4`` but can be adapted as needed.

        If you provide ``testpulse_channels``, triggers within a record window of a testpulse are treated as testpulses. Note that you have to provide testpulse information for all channels INCLUDING 'slave' channels.

        :param stream: The stream object including the channels that you want to trigger.
        :type stream: vai.datasources.stream.streambase.StreamBaseClass
        :param trigger_channels: The list of channel names to be triggered. Have to be present in ``stream.keys``.
        :type trigger_channels: List[str]
        :param thresholds: A list of trigger thresholds (in V) for each channel.
        :type thresholds: Union[float, List[float]]
        :param slave_channels: A list of channel names to be read out as 'slaves'. Have to be present in ``stream.keys``. Defaults to None
        :type slave_channels: List[str], optional
        :param testpulse_channels: A list of channel names to be used as testpulses. Have to be present in ``stream.tp_timestamps.keys``. Defaults to None
        :type testpulse_channels: List[str], optional
        :param copy_events: If true, the voltage traces of the events which were built are saved in the DataHandler (i.e. copied from the stream files). Defaults to False.
        :type copy_events: bool, optional
        :param reuse_triggers: If true, the triggers from a previous call of this function (which were saved in the DataHandler) are reused and only the event building is performed again (possibly with a different coincidence interval). Defaults to False.
        :type reuse_triggers: bool, optional
        :param interval: The coincidence interval for event building in microseconds, i.e. if a trigger lies within the specified interval around a trigger of another channel, they are collected to represent one event. Defaults to ``-+dt_us*record_length//4``.
        :type interval: Tuple[float], optional
        :param of: The optimum filter to use for triggering (has to have one for each channel in 'trigger_channels'). If none is specified (default), the optimum filter is read from the DataHandler.
        :type of: np.ndarray, optional
        :param n_noise: The number of empty noise traces to include. Defaults to 0, i.e. no noise is included.
        :type n_noise: int, optional
        :param kwargs: Additional keyword arguments forwarded to :func:`cait.versatile.trigger_of`.
        :type kwargs: Any

        **Example:**

        .. code-block:: python
        
            import cait as ai
            import cait.versatile as vai

            # Construct stream object
            stream = vai.Stream(hardware="vdaq2", src="path/to/stream_file.bin")
            
            print(f"Available channels: {stream.keys}")
            print(f"Available TP channels: {stream.tp_timestamps}")

            # Construct DataHandler
            dh = ai.DataHandler(record_length=2**13, nmbr_channels=2, sample_frequency=stream.sample_frequency)
            dh.set_filepath(path_h5="folder_name/", fname="z-score-triggered", appendix=False)
            dh.init_empty()

            # Copy OF (a two-dimensional OF was previously saved)
            vai.OF.from_file("path/to/OF_file").to_dh(dh)

            # Trigger
            dh.trigger_zscore(stream, 
                              trigger_channels=["ADC1", "ADC2"],
                              testpulse_channels=["DAC1", "DAC3"],
                              thresholds=[1e-3, 1.5e-3],
                              copy_events=True)
        """
        # Allow string input if only one channel
        trigger_channels = [trigger_channels] if isinstance(trigger_channels, str) else trigger_channels
        slave_channels = [slave_channels] if isinstance(slave_channels, str) else slave_channels
        testpulse_channels = [testpulse_channels] if isinstance(testpulse_channels, str) else testpulse_channels
        
        # Make sure that thresholds are a list (allow scalar input, to be used for all channels)
        thresholds = [thresholds]*len(trigger_channels) if isinstance(thresholds, (int, float)) else thresholds
        
        if len(thresholds) != len(trigger_channels):
            raise ValueError(f"You need to provide as many thresholds as trigger channels. Received {len(thresholds)} and {len(trigger_channels)}")
            
        if of is not None:
            ofs = of
        elif self.exists("optimumfilter"):
            ofs = vai.OF.from_dh(self)
        else:
            raise ValueError("No OF provided when calling trigger function and no group 'optimumfilter' found in DataHandler. Provide an optimum filter!")
        
        if ofs.ndim == 1:
            ofs = np.array([ofs])
            
        if len(ofs) != len(trigger_channels):
            raise ValueError(f"Optimum filter has to have as many channels as channels to trigger, i.e. len(of) must be len(trigger_channels). Received {len(ofs)} and {len(trigger_channels)}.")
        
        trigger_fncs = [partial(vai.trigger_of, 
                               of=of,
                               threshold=thresh, 
                               **kwargs) 
                       for of, thresh in zip(ofs, thresholds)]
        
        _trigger_helper(self, stream, trigger_channels, slave_channels, testpulse_channels, copy_events, reuse_triggers,
                        interval, trigger_fncs, n_noise, "of")
        
    def trigger_coincidence(self,
                            filedict: dict,
                            interval: Tuple[float] = None,
                            sigma: Union[float, List[float]] = 5,
                            reuse_triggers: bool = False,
                            copy_events: bool = False,
                            **kwargs
                           ):
            """
            A trigger and event building convenience function developed with the needs of CRESST doubleTES analysis in mind. In the first step, a moving z-score trigger is applied to all provided channels. Afterwards, all timestamps found are grouped into events depending on the allowed coincidence interval which was specified. If testpulse information is provided, they are automatically recognized in the event building step and excluded from the events group. The function can be called multiple times with a different ``interval`` argument and as long as ``reuse_triggers==True``, the triggering itself does not have to be performed again. This allows the user to explore different intervals conveniently. Finally, the user can decide to copy the events which were built to the DataHandler by setting ``copy_events=True``. This will create an 'events' group (and a 'testpulses' group if testpulse information was provided).
            
            :param filedict: A dictionary containing the required files and timestamp assignment information (see structure in example below). The dictionary needs a 'par' key, whose value is the ``.par`` file of the recording, as well as at least two keys starting with 'ch' (e.g. 'ch4' and 'ch5'), whose values are lists of the ``.csmpl`` file containing the channel's stream and the testpulse channel (in the ``.test_stamps`` file) which pulses to this channel (if a testpulse channel sends pulses to multiple channels, the values can appear more than once). If you also provide a key 'tp', whose value is a list containing the ``.test_stamps`` and ``.dig_stamps`` files, testpulses are automatically recognized in the event building step (i.e. excluded from the particle events group and added to a separate 'testpulses' group).
            :type filedict: dict
            :param interval: The coincidence interval for event building in microseconds, i.e. if a trigger lies within the specified interval around a trigger of another channel, they are collected to represent one event. Defaults to ``-+dt_us*record_length//4``.
            :type interval: Tuple[float], optional
            :param sigma: The threshold of the moving z-score trigger in standard deviations, defaults to 5.
            :type sigma: Union[float, List[float]], optional
            :param reuse_triggers: If true, the triggers from a previous call of this function (which were saved in the DataHandler) are reused and only the event building is performed again (possibly with a different coincidence interval). Defaults to False.
            :type reuse_triggers: bool, optional
            :param copy_events: If true, the voltage traces of the events which were built are saved in the DataHandler (i.e. copied from the stream files). Defaults to False.
            :type copy_events: bool, optional
            :param kwargs: Additional keyword arguments forwarded to :func:`cait.versatile.trigger_zscore`.
            :type kwargs: Any
            
            **Example:**

            .. code-block:: python

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
            stream = vai.Stream("csmpl", stream_arg)
            tp_keys = dict()
            for ch_name in stream.keys:
                for k,v in filedict.items():
                    if k.lower().startswith("ch") and os.path.basename(os.path.splitext(v[0])[0]).endswith(ch_name):
                        tp_keys[ch_name] = v[1]
            
            sigmas = [sigma]*len(stream.keys) if isinstance(sigma, (int, float)) else sigma

            rec_window_coinc = (-stream.dt_us*self.record_length//4, stream.dt_us*self.record_length//4)
            if interval is None: 
                interval = rec_window_coinc
                
            # Collect all testpulses into a coherent dataset, i.e. collect TPs sent at the same times into one event
            # and if only one channel received the TP, the TPA in all other channels is set to 404
            # for a commented version of this part, see procedure below for events which is practically identical
            if "tp" in filedict.keys():
                all_tpas = [list(stream.tpas[tp_keys[stream.keys[0]]])]
                all_tp_ts = list(stream.tp_timestamps[tp_keys[stream.keys[0]]])
                
                tp_sent_flag = [[True]*len(all_tp_ts)]
                
                for i, key in enumerate(stream.keys[1:]):
                    all_tpas.append(list(stream.tpas[tp_keys[key]]))
                    
                    this_tp_ts = stream.tp_timestamps[tp_keys[key]]
                    inside, coinc_inds, outside = vai.timestamp_coincidence(all_tp_ts, this_tp_ts, rec_window_coinc)

                    for k in range(i+1):
                        tp_sent_flag[k].extend([False]*len(outside))

                    new_flag = np.array([False]*len(all_tp_ts) + [True]*len(outside))
                    new_flag[coinc_inds] = True
                    tp_sent_flag.append(new_flag)
                    
                    new_ts = np.concatenate([all_tp_ts, np.array(this_tp_ts)[outside]])
                    sortind = np.argsort(new_ts)
                    all_tp_ts = new_ts[sortind]

                    for j in range(i+2):
                        tp_sent_flag[j] = np.array(tp_sent_flag[j])[sortind].tolist()
                
                final_tpas = 404*np.ones(np.array(tp_sent_flag).shape, dtype=np.float32)
                print(len(all_tpas))
                for i, pa in enumerate(all_tpas):
                    final_tpas[i, np.array(tp_sent_flag[i])] = np.array(pa)

            trigger_ts, trigger_phs = [], []

            for i, (key, sigma) in enumerate(zip(stream.keys, sigmas)):
                if reuse_triggers:
                    if not (self.exists("triggers", f"ts_{key}") and self.exists("triggers", f"ph_{key}")):
                        raise KeyError(f"To reuse triggers, datasets 'ts_{key}' and 'ph_{key}' must exist in the 'triggers' group.")

                    ts = list(self.get("triggers", f"ts_{key}"))
                    ph = list(self.get("triggers", f"ph_{key}"))

                else:
                    with stream: # this keeps the stream file opened (performance increase)
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

                # if testpulse information is provided, trigger timestamps within a quater record window around 
                # testpulse timestamps (regardless of the channel) are counted as such
                if "tp" in filedict.keys():
                    *_, outside = vai.timestamp_coincidence(all_tp_ts, ts, rec_window_coinc)
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
                if self.exists("events"): raise Exception("Could not copy events to DataHandler because the group 'events' already exists. To delete it, use 'dh.drop('events')'.")

                print("Writing events to DataHandler...")
                self.include_event_iterator("events", stream.get_event_iterator(stream.keys, self.record_length, timestamps=event_ts))

                # do the same for testpulses if respective information is provided
                if "tp" in filedict.keys():
                    if self.exists("testpulses"): raise Exception("Could not copy events to DataHandler because the group 'testpulses' already exists. To delete it, use 'dh.drop('testpulses')'.")
                    # make sure all timestamps written in the tp file are actually within the stream file (and their voltage traces can be read completely)
                    valid_tp_flag = all_tp_ts < stream.time[-3*self.record_length//4]
                    if not all(valid_tp_flag): print("One or more testpulses could not be included because they fall (partially) outside the stream's range!!")
                    
                    # save testpulses and tpas
                    print("Writing testpulses to DataHandler...")
                    self.include_event_iterator("testpulses", stream.get_event_iterator(stream.keys, self.record_length, timestamps=all_tp_ts[valid_tp_flag]))
                    self.set("testpulses", testpulseamplitude=final_tpas[..., valid_tp_flag])