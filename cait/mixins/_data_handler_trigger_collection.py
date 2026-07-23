import warnings
from functools import partial
from typing import List, Tuple, Union

import numpy as np

import cait.versatile as vai
from cait.versatile.datasources.stream.streambase import StreamBaseClass


# Helper function that validates and sanitizes input and outputs
# correctly formatted versions of the inputs
def _sanitize_input(stream,
                    trigger_channels,
                    passive_channels,
                    testpulse_channels,
                    controlpulses_above,
                    calibration_channels,
                    thresholds):
    # Allow string input if only one channel
    trigger_channels = [trigger_channels] if isinstance(trigger_channels, str) else trigger_channels
    passive_channels = [passive_channels] if isinstance(passive_channels, str) else passive_channels
    testpulse_channels = [testpulse_channels] if isinstance(testpulse_channels, str) else testpulse_channels
    calibration_channels = [calibration_channels] if isinstance(calibration_channels, str) else calibration_channels
    
    # Make sure that thresholds are a list (allow scalar input, to be used for all channels)
    thresholds = [thresholds]*len(trigger_channels) if isinstance(thresholds, (int, float)) else thresholds

    if len(thresholds) != len(trigger_channels):
        raise ValueError(f"You need to provide as many thresholds as trigger channels. Received {len(thresholds)} and {len(trigger_channels)}")
    
    # All trigger channels (flattened such that when a 2d of is used, we can still check
    # all required channel names conveniently)
    trigger_channels_flat = np.hstack(trigger_channels).tolist()

    # Input validation
    if not all([x in stream.keys for x in trigger_channels_flat]):
        raise KeyError(f"All 'trigger_channels' have to be valid channel names. Available: {stream.keys}")
            
    if ( passive_channels is not None ) and ( not all([x in stream.keys for x in passive_channels]) ):
        raise KeyError(f"All 'passive_channels' have to be valid channel names. Available: {stream.keys}")

    if ( testpulse_channels is not None ) and ( not all([x in stream.tp_keys for x in testpulse_channels]) ):
        raise KeyError(f"All 'testpulse_channels' have to be valid channel names. Available: {stream.tp_keys}")
    
    if ( calibration_channels is not None ) and ( not all([x in stream.calp_keys for x in calibration_channels]) ):
        raise KeyError(f"All 'calibration_channels' have to be valid channel names. Available: {stream.calp_keys}")

    if testpulse_channels is not None:
        if len(trigger_channels_flat) + (0 if passive_channels is None else len(passive_channels)) != len(testpulse_channels):
            raise ValueError(f"Testpulse channels are required for all channels (including passive channels). I.e. 'len(testpulse_channels)' must match 'len(np.hstack(trigger_channels))+len(passive_channels)'. Received {len(testpulse_channels)} and {len(trigger_channels_flat)}+{0 if passive_channels is None else len(passive_channels)}")
    
    if calibration_channels is not None:
        if len(calibration_channels) != (len(trigger_channels_flat) + (0 if passive_channels is None else len(passive_channels))):
            raise ValueError(f"Calibration channels are required for all channels (including passive channels). I.e. 'len(calibration_channels)' must match 'len(np.hstack(trigger_channels))+len(passive_channels)'. Received {len(calibration_channels)} and {len(trigger_channels_flat)}+{0 if passive_channels is None else len(passive_channels)}")

    # Make sure that controlpulses_above is a list (allow scalar input, to be used for all channels)
    # Also check if it is specified for all testpulse channels
    if controlpulses_above is not None:
        if testpulse_channels is None:
            raise ValueError("If you specify 'controlpulses_above', you also have to specify 'testpulse_channels'.")
        else:
            controlpulses_above = [controlpulses_above]*len(testpulse_channels) if isinstance(controlpulses_above, (int, float, tuple)) else controlpulses_above

        if not (len(testpulse_channels) == len(controlpulses_above)):
            raise ValueError(f"The length of 'controlpulses_above' and 'testpulse_channels' has to match. Got {len(controlpulses_above)} and {len(testpulse_channels)}.")
        
    return trigger_channels, passive_channels, testpulse_channels, controlpulses_above, calibration_channels, thresholds

# Helper function that is used in both trigger_of and trigger_zscore
def _trigger_helper(dh, 
                    stream, 
                    trigger_channels,
                    passive_channels, 
                    testpulse_channels, 
                    controlpulses_above,
                    calibration_channels,
                    copy_events, 
                    reuse_triggers,
                    interval,
                    trigger_fncs,
                    f_noise,
                    name_appendix
                    ):

    # Convert noise sample frequency to number of noise traces to sample.
    # Print info if less than one sample is expected. Set number to at least
    # one because we want to prevent the case where some files don't produce
    # noise traces (and the noise group therefore not being in the HDF5 file)
    n_noise = int( f_noise * len(stream)*stream.dt_us/1e6/3600 )
    if f_noise>0 and n_noise==0:
        warnings.warn(f"The specified frequency to sample noise ({f_noise:.3f}/h) resulted in less than one expected event for the given stream of length {len(stream)*stream.dt_us/1e6/3600:.3f} h. Only a single event will be (attempted to be) sampled.")
        n_noise += 1

    # The entries of this list can either be tuples (results e.g. in OF2D) 
    # or strings (regular single channel trigger).
    all_channels = trigger_channels + ([] if passive_channels is None else passive_channels)
    # This is the same list but with only strings (will be used for including the raw
    # voltage traces later).
    all_channels_flat = np.hstack(all_channels).tolist()

    # Triggering channels
    for i, key in enumerate(trigger_channels):
        if reuse_triggers:
            if not (dh.exists(f"triggers-{name_appendix}", f"ts_{key}") and dh.exists(f"triggers-{name_appendix}", f"ph_{key}")):
                raise KeyError(f"To reuse triggers, datasets 'ts_{key}' and 'ph_{key}' must exist in the 'triggers-{name_appendix}' group.")
            print(f"Read existing triggers for channel {key}.")

        else:
            with stream: # this keeps the stream file opened (performance increase)
                print(f"Triggering channel {key} ...")
                # Multi-channel stream slicing requires 'key' to be a list (not tuple)
                # because otherwise the slicing would be ambiguous. At this point, 
                # however, 'key' is a tuple to prevent ambiguity when calling the trigger
                # function. Therefore, we convert tuples to lists.
                sanitized_key = list(key) if isinstance(key, tuple) else key
                ind, ph = trigger_fncs[i](stream[sanitized_key])
                
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

    trigger_ts_temp = [ list(dh.get(f"triggers-{name_appendix}", f"ts_{key}")) for key in trigger_channels]
    trigger_ph_temp = [ list(dh.get(f"triggers-{name_appendix}", f"ph_{key}")) for key in trigger_channels]

    # Account for potentially combined channels (e.g. in OF2D triggering) by filling
    # all channels' information with the trigger information (if they were triggered
    # together, i.e. if their channel name is a tuple).
    trigger_ts, trigger_ph = [], []
    for ch, ts, ph in zip(trigger_channels, trigger_ts_temp, trigger_ph_temp):
        if isinstance(ch, tuple):
            for _ in range(len(ch)):
                trigger_ts.append(ts)
                trigger_ph.append(ph)
        else:
            trigger_ts.append(ts)
            trigger_ph.append(ph)

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

    if calibration_channels is not None:
        for key in calibration_channels:
            if reuse_triggers and dh.exists(f"triggers-{name_appendix}", f"calp_ts_{key}"):
                print(f"Read existing calibration pulses for channel {key}.")
            else:
                dh.set(f"triggers-{name_appendix}", 
                         **{f"calp_ts_{key}": np.array(stream.calp_timestamps[key])},
                         dtype=np.int64,
                         overwrite_existing=True)
                dh.set(f"triggers-{name_appendix}", 
                         **{f"calpas_{key}": np.array(stream.calpas[key])},
                         dtype=np.float32,
                         overwrite_existing=True)

        calp_ts = [list(dh.get(f"triggers-{name_appendix}", f"calp_ts_{key}")) for key in calibration_channels]
        calpas = [list(dh.get(f"triggers-{name_appendix}", f"calpas_{key}")) for key in calibration_channels]
    
    else:
        calp_ts, calpas = None, None

    print("Building events ...")
    event_ts, trig_flag, orig_ts, orig_ph, all_tp_ts, final_tpas = vai.event_building(
                                                                trigger_ts=trigger_ts, 
                                                                trigger_phs=trigger_ph, 
                                                                record_length=dh.record_length, 
                                                                dt_us=stream.dt_us,
                                                                tp_ts=tp_ts,
                                                                tpas=tpas,
                                                                n_passive_ch=0 if passive_channels is None else len(passive_channels),
                                                                interval=interval)
    
    if (calp_ts is not None) and (calpas is not None):
        
        # After excluding events in coincidence with testpulses, we now exclude events in coincidence with calpulses.

        # If a channel did not trigger separately on an event, the ts and ph of such event were set to -1 while building the events.
        # Here, we remove those events before feeding them into the calpulses event_building
        trigger_ts_notp = [[ts for ts in timestamps if (ts != -1)] for timestamps in orig_ts]
        trigger_ph_notp = [[ph for ph in pulse_heights if (ph != -1)] for pulse_heights in orig_ph]

        event_ts, trig_flag, orig_ts, orig_ph, all_calp_ts, final_calpas = vai.event_building(
                                                                        trigger_ts=trigger_ts_notp, 
                                                                        trigger_phs=trigger_ph_notp, 
                                                                        record_length=dh.record_length, 
                                                                        dt_us=stream.dt_us,
                                                                        tp_ts=calp_ts,
                                                                        tpas=calpas,
                                                                        n_passive_ch=0 if passive_channels is None else len(passive_channels),
                                                                        interval=interval)

    if controlpulses_above is not None:
        # due to the input validation in the top-level functions 
        # we can be sure that all lists/arrays have consistent 
        # sizes for this loop
        cp_flag = np.zeros(all_tp_ts.shape[-1], dtype=bool)
        for i in range(len(controlpulses_above)):
            # Range specified as tuple (special use case, e.g. for double-TES analysis)
            if isinstance(controlpulses_above[i], tuple):
                cp_flag += (final_tpas[i] > controlpulses_above[i][0])*(final_tpas[i] < controlpulses_above[i][1])
            # Regular lower limit for counting testpulses as controlpulses
            else:
                cp_flag += final_tpas[i] > controlpulses_above[i]

        all_cp_ts = all_tp_ts[cp_flag]
        final_cpas = final_tpas[:, cp_flag]
        all_tp_ts = all_tp_ts[~cp_flag]
        final_tpas = final_tpas[:, ~cp_flag]
    else:
        all_cp_ts = final_cpas = None
        

    # save final timestamps and trigger flag after event building
    dh.set(f"event_building-{name_appendix}", event_timestamps=event_ts, dtype=np.int64, overwrite_existing=True)
    dh.set(f"event_building-{name_appendix}", trigger_flag=trig_flag, dtype=bool, overwrite_existing=True)

    dh.set(f"event_building-{name_appendix}", trigger_timestamps=orig_ts, dtype=np.int64, overwrite_existing=True)
    dh.set(f"event_building-{name_appendix}", trigger_phs=orig_ph, dtype=np.float32, overwrite_existing=True)

    if testpulse_channels is not None:
        dh.set(f"event_building-{name_appendix}", tp_ts=all_tp_ts, dtype=np.int64, overwrite_existing=True)
        dh.set(f"event_building-{name_appendix}", tpas=final_tpas, dtype=np.float32, overwrite_existing=True)

    if controlpulses_above is not None:
        dh.set(f"event_building-{name_appendix}", cp_ts=all_cp_ts, dtype=np.int64, overwrite_existing=True)
        dh.set(f"event_building-{name_appendix}", cpas=final_cpas, dtype=np.float32, overwrite_existing=True)
    
    if calibration_channels is not None:
        dh.set(f"event_building-{name_appendix}", calp_ts=all_calp_ts, dtype=np.int64, overwrite_existing=True)
        dh.set(f"event_building-{name_appendix}", calpas=final_calpas, dtype=np.float32, overwrite_existing=True)
        
    if n_noise > 0:
        temp_ts = dh.get(f"event_building-{name_appendix}", "event_timestamps")
        if dh.exists(f"event_building-{name_appendix}", "tp_ts"):
            temp_ts = np.hstack([temp_ts, dh.get(f"event_building-{name_appendix}", "tp_ts")])
        if dh.exists(f"event_building-{name_appendix}", "cp_ts"):
            temp_ts = np.hstack([temp_ts, dh.get(f"event_building-{name_appendix}", "cp_ts")])
        if dh.exists(f"event_building-{name_appendix}", "calp_ts"):
            temp_ts = np.hstack([temp_ts, dh.get(f"event_building-{name_appendix}", "calp_ts")])

        # Remove timestamps which would result in a trace extending outside of the stream's domain.
        # (We are a bit generous here and exclude a full record window in the beginning and in the end
        # of the stream even though 1/4 and 3/4 of a record window would be enough, but then we have to 
        # think about off-by-one errors and we are to lazy for that)
        inside_flag = (temp_ts > stream.time[dh.record_length]) * (temp_ts < stream.time[-dh.record_length])

        inds = stream.time.timestamp_to_ind(temp_ts[inside_flag])
        noise_inds = vai.sample_noise(inds.tolist(), dh.record_length, n_samples=n_noise)
        dh.set(f"event_building-{name_appendix}", noise_ts=stream.time[noise_inds], dtype=np.int64, overwrite_existing=True)

    # save events in events group
    if copy_events and dh.exists("events/event"): 
        warnings.warn("Could not copy events to DataHandler because dataset 'event' in group 'events' already exists. To delete it, use 'dh.drop('events', 'event')'.")
        copy_this = False
    else:
        copy_this = copy_events
    
    if len(event_ts)>0:
        dh.include_event_iterator("events", 
                                    stream.get_event_iterator(
                                        all_channels_flat, 
                                        dh.record_length, 
                                        timestamps=event_ts
                                    ),
                                    copy_events=copy_this)
        # Also copy the raw trigger information to the 'events' group
        dh.set("events", trigger_flag=trig_flag, dtype=bool, overwrite_existing=True)
        dh.set("events", trigger_timestamps=orig_ts, dtype=np.int64, overwrite_existing=True)
        dh.set("events", trigger_phs=orig_ph, dtype=np.float32, overwrite_existing=True)
    else:
        print("No events found to write to DataHandler.")

    # do the same for testpulses if respective information is provided
    if testpulse_channels is not None:
        if copy_events and dh.exists("testpulses/event"): 
            warnings.warn("Could not copy events to DataHandler because dataset 'event' in group 'testpulses' already exists. To delete it, use 'dh.drop('testpulses', 'event')'.")
            copy_this = False
        else:
            copy_this = copy_events

        # make sure all timestamps written in the tp file are actually within the stream file (and their voltage traces can be read completely)
        valid_tp_flag = all_tp_ts < stream.time[-3*dh.record_length//4]
        valid_tp_flag *= (all_tp_ts > stream.time[dh.record_length//4])
        if not all(valid_tp_flag): 
            print("One or more testpulses could not be included because they fall (partially) outside the stream's range!!")

        # save testpulses and tpas
        tp_ts = all_tp_ts[valid_tp_flag]
        if len(tp_ts)>0:
            dh.include_event_iterator("testpulses", 
                                        stream.get_event_iterator(
                                            all_channels_flat, 
                                            dh.record_length, 
                                            timestamps=tp_ts
                                        ),
                                        copy_events=copy_this)
            dh.set("testpulses", testpulseamplitude=final_tpas[..., valid_tp_flag], overwrite_existing=True)
        else:
            print("No testpulses found to write to DataHandler.")

    if controlpulses_above is not None:
        if copy_events and dh.exists("controlpulses/event"): 
            warnings.warn("Could not copy controlpulses to DataHandler because dataset 'event' in group 'controlpulses' already exists. To delete it, use 'dh.drop('controlpulses', 'event')'.")
            copy_this = False
        else:
            copy_this = copy_events

        if len(all_cp_ts)>0:
            # make sure all timestamps written in the tp file are actually within the stream file (and their voltage traces can be read completely)
            valid_cp_flag = all_cp_ts < stream.time[-3*dh.record_length//4]
            valid_tp_flag *= (all_tp_ts > stream.time[dh.record_length//4])
            if not all(valid_cp_flag): 
                print("One or more controlpulses could not be included because they fall (partially) outside the stream's range!!")

            # save controlpulses and cpas
            cp_ts = all_cp_ts[valid_cp_flag]
            if len(cp_ts)>0:
                dh.include_event_iterator("controlpulses", 
                                            stream.get_event_iterator(
                                                all_channels_flat, 
                                                dh.record_length, 
                                                timestamps=cp_ts
                                            ),
                                            copy_events=copy_this)
                dh.set("controlpulses", testpulseamplitude=final_cpas[..., valid_cp_flag], overwrite_existing=True)
            else:
                print("No controlpulses found to write to DataHandler.")
        else:
            print("No controlpulses found to write to DataHandler.")

    # do the same for calibration pulses if respective information is provided
    if calibration_channels is not None:
        if copy_events and dh.exists("calpulses/event"): 
            warnings.warn("Could not copy events to DataHandler because dataset 'event' in group 'calpulses' already exists. To delete it, use 'dh.drop('calpulses', 'event')'.")
            copy_this = False
        else:
            copy_this = copy_events

        # make sure all timestamps written in the calibration file are actually within the stream file (and their voltage traces can be read completely)
        valid_cal_flag = all_calp_ts < stream.time[-3*dh.record_length//4]
        if not all(valid_cal_flag): 
            print("One or more calpulses could not be included because they fall (partially) outside the stream's range!!")

        # save calpulses and calpas
        calp_ts = all_calp_ts[valid_cal_flag]
        if len(calp_ts)>0:
            dh.include_event_iterator("calpulses", 
                                        stream.get_event_iterator(
                                            all_channels_flat, 
                                            dh.record_length, 
                                            timestamps=calp_ts
                                        ),
                                        copy_events=copy_this)
            dh.set("calpulses", calpulseamplitude=final_calpas[..., valid_cal_flag], overwrite_existing=True)
        else:
            print("No calpulses found to write to DataHandler.")

    if n_noise>0:
        if copy_events and dh.exists("noise/event"): 
            warnings.warn("Could not copy noise to DataHandler because dataset 'event' in group 'noise' already exists. To delete it, use 'dh.drop('noise', 'event')'.")
            copy_this = False
        else:
            copy_this = copy_events
        
        noise_ts = dh.get(f"event_building-{name_appendix}", "noise_ts")

        if len(noise_ts)>0:
            dh.include_event_iterator("noise", 
                                        stream.get_event_iterator(
                                            all_channels_flat, 
                                            dh.record_length, 
                                            timestamps=noise_ts
                                        ),
                                        copy_events=copy_this)
        else:
            print("No noise found to write to DataHandler.")

class TriggerCollectionMixin:
    """
    A mixin class with convenience functions concerning triggering and event building.
    """
    def trigger_zscore(self,
                       stream: StreamBaseClass,
                       trigger_channels: List[str],
                       thresholds: Union[float, List[float]] = 5,
                       passive_channels: List[str] = None,
                       testpulse_channels: List[str] = None,
                       controlpulses_above: List[Union[float, Tuple[float]]] = None,
                       calibration_channels: List[str] = None,
                       copy_events: bool = False,
                       reuse_triggers: bool = False,
                       interval: Tuple[float] = None,
                       f_noise: float = 0,
                       **kwargs
                      ):
        """
        Trigger stream channels from arbitrary hardware using a moving z-score trigger and build events from trigger timestamps (of multiple channels) and exclude testpulses if the respective information is provided.

        The stream channels specified by ``trigger_channels`` are triggered. If some channels are not triggered but read out in coincidence (i.e. as 'passive' channels), you can specify their channel names using ``passive_channels``. 

        Events are built as follows: Starting from the first channel's trigger timestamps, the remaining channels' triggers are checked to be in coincidence with already existing timestamps. The default coincidence window (if ``interval=None``), is ``-+dt_us*record_length//4`` but can be adapted as needed.

        If you provide ``testpulse_channels``, triggers within a record window of a testpulse are treated as testpulses. Note that you have to provide testpulse information for all channels INCLUDING 'passive' channels.

        :param stream: The stream object including the channels that you want to trigger.
        :type stream: StreamBaseClass
        :param trigger_channels: The list of channel names to be triggered. Have to be present in ``stream.keys``.
        :type trigger_channels: List[str]
        :param thresholds: A list of trigger thresholds (in sigmas) for each channel. If only a float is provided, it is used for all channels. Defaults to 5 sigmas
        :type thresholds: Union[float, List[float]], optional
        :param passive_channels: A list of channel names to be read out as 'passives'. Have to be present in ``stream.keys``. Defaults to None
        :type passive_channels: List[str], optional
        :param testpulse_channels: A list of channel names to be used as testpulses. Have to be present in ``stream.tp_keys``. Defaults to None
        :type testpulse_channels: List[str], optional
        :param controlpulses_above: If specified, all testpulses with testpulse amplitudes above this value are considered to be controlpulses (i.e. they are saved in their own group in the DataHandler). You have to specify as many values as in 'testpulse_channels' (as a list). If you want to enable this feature for only one channel, just set the values for the other channels to some which cannot be exceeded, e.g. 1000. If you want to enforce an upper limit as well (i.e. count testpulses as controlpulses for testpulse amplitudes between ``a`` and ``b``), you can do so by passing a tuple ``(a, b)``. Defaults to None, i.e. no testpulse is counted as controlpulse.
        :type controlpulses_above: List[Union[float, Tuple[float]]], optional
        :param calibration_channels: A list of channel names to be used as calibration channels. Have to be present in ``stream.calp_keys``. Defaults to None
        :type calibration_channels: List[str], optional
        :param copy_events: If True, the voltage traces of the events which were built are saved in the DataHandler (i.e. copied from the stream files). If False, only a reference to the original data is saved. Defaults to False.
        :type copy_events: bool, optional
        :param reuse_triggers: If true, the triggers from a previous call of this function (which were saved in the DataHandler) are reused and only the event building is performed again (possibly with a different coincidence interval). Defaults to False.
        :type reuse_triggers: bool, optional
        :param interval: The coincidence interval for event building in microseconds, i.e. if a trigger lies within the specified interval around a trigger of another channel, they are collected to represent one event. Defaults to ``-+dt_us*record_length//4``.
        :type interval: Tuple[float], optional
        :param f_noise: The frequency (in events per hour) of empty noise traces to include. Defaults to 0, i.e. no noise is included.
        :type f_noise: float, optional
        :param kwargs: Additional keyword arguments forwarded to :func:`cait.versatile.trigger_zscore`.
        :type kwargs: Any

        **Example:**

        .. code-block:: python
        
            import cait as ai
            import cait.versatile as vai

            # Construct stream object
            stream = vai.Stream(hardware="vdaq2", src="path/to/stream_file.bin")

            print(f"Available channels: {stream.keys}")
            print(f"Available TP channels: {stream.tp_keys}")

            # Construct DataHandler
            dh = ai.DataHandler(record_length=2**13, nmbr_channels=2, sample_frequency=stream.sample_frequency)
            dh.set_filepath(path_h5="folder_name/", fname="z-score-triggered", appendix=False)
            dh.init_empty()

            # Trigger ADC1 (phonon channel) and read ADC2 in coincidence (light channel)
            dh.trigger_zscore(stream, 
                              trigger_channels=["ADC1"],
                              passive_channels=["ADC2"],
                              testpulse_channels=["DAC1", "DAC3"],
                              copy_events=True)
        """
        
        trigger_channels, passive_channels, testpulse_channels, controlpulses_above, calibration_channels, thresholds = _sanitize_input(
            stream=stream,
            trigger_channels=trigger_channels,
            passive_channels=passive_channels,
            testpulse_channels=testpulse_channels,
            controlpulses_above=controlpulses_above,
            calibration_channels=calibration_channels,
            thresholds=thresholds,
        )
        
        trigger_fncs = [partial(vai.trigger_zscore, 
                               threshold=thresh, 
                               record_length=self.record_length, 
                               **kwargs) 
                       for thresh in thresholds]
        
        _trigger_helper(self, stream, trigger_channels, passive_channels, testpulse_channels, controlpulses_above, calibration_channels,
                        copy_events, reuse_triggers, interval, trigger_fncs, f_noise, "z-score")
        
    def trigger_of(self,
                   stream: StreamBaseClass,
                   trigger_channels: List[Union[Tuple[str], str]],
                   of: np.ndarray,
                   thresholds: List[float],
                   passive_channels: List[str] = None,
                   testpulse_channels: List[str] = None,
                   controlpulses_above: List[Union[float, Tuple[float]]] = None,
                   calibration_channels: List[str] = None,
                   copy_events: bool = False,
                   reuse_triggers: bool = False,
                   interval: Tuple[float] = None,
                   f_noise: float = 0,
                   **kwargs
                   ):
        """
        Trigger stream channels from arbitrary hardware using a moving optimum filter trigger and build events from trigger timestamps (of multiple channels) and exclude testpulses if the respective information is provided.

        The stream channels specified by ``trigger_channels`` are triggered. If some channels are not triggered but read out in coincidence (i.e. as 'passive' channels), you can specify their channel names using ``passive_channels``. 

        If you want to perform a 2D optimum filter trigger, you specify the respective channels (which are filtered together) as tuples in the ``trigger_channels`` list (e.g. ``[('ch0', 'ch1'), 'ch2']`` will apply a 2D optimum filter to the first two channels. This will result in A SINGLE filtered trace which is triggered. The last channel is triggered regularly). Note that the ``of`` you pass to this function has to have as many channels as total trigger channels. I.e. in the example above, it would need to have 3 channels. Likewise, the ``testpulse_channels`` and ``controlpulses_above`` lists have to have length 3 here. HOWEVER, you only need 2 (!) trigger threshold (one for the combined channel, one for the single channel).

        Events are built as follows: Starting from the first channel's trigger timestamps, the remaining channels' triggers are checked to be in coincidence with already existing timestamps. The default coincidence window (if ``interval=None``), is ``-+dt_us*record_length//4`` but can be adapted as needed.

        If you provide ``testpulse_channels``, triggers within a record window of a testpulse are treated as testpulses. Note that you have to provide testpulse information for all channels INCLUDING 'passive' channels.

        :param stream: The stream object including the channels that you want to trigger.
        :type stream: StreamBaseClass
        :param trigger_channels: The list of channel names to be triggered. Have to be present in ``stream.keys``. If you pass a tuple of channel names, the 2D optimum filter is applied to this combination. See explanation above.
        :type trigger_channels: List[Union[Tuple[str], str]]
        :param of: The optimum filter to use for triggering (Has to have one for each channel in ``trigger_channels``. If either of the trigger channels is a tuple, i.e. treated using the 2D optimum filter, they count as multiple channels. E.g. if you 2D optimum filter the first two channels together and the third channel with the regular optimum filter, ``of`` would have to have shape ``(3, kernel_length)``).
        :type of: np.ndarray
        :param thresholds: A list of trigger thresholds (in V) for each entry in ``trigger_channels``. If only one is specified, the respective threshold is used for all channels. Note that a tuple in ``trigger_channels``, i.e. a channel combination treated by a 2D optimum filter, needs only one threshold.
        :type thresholds: Union[float, List[float]]
        :param passive_channels: A list of channel names to be read out as 'passives'. Have to be present in ``stream.keys``. Defaults to None
        :type passive_channels: List[str], optional
        :param testpulse_channels: A list of channel names to be used as testpulses. Have to be present in ``stream.tp_keys``. Defaults to None
        :type testpulse_channels: List[str], optional
        :param controlpulses_above: If specified, all testpulses with testpulse amplitudes above this value are considered to be controlpulses (i.e. they are saved in their own group in the DataHandler). You have to specify as many values as in 'testpulse_channels' (as a list). If you want to enable this feature for only one channel, just set the values for the other channels to some which cannot be exceeded, e.g. 1000. If you want to enforce an upper limit as well (i.e. count testpulses as controlpulses for testpulse amplitudes between ``a`` and ``b``), you can do so by passing a tuple ``(a, b)``. Defaults to None, i.e. no testpulse is counted as controlpulse.
        :type controlpulses_above: List[Union[float, Tuple[float]]], optional
        :param calibration_channels: A list of channel names to be used as calibration channels. Have to be present in ``stream.calp_keys``. Defaults to None
        :type calibration_channels: List[str], optional
        :param copy_events: If True, the voltage traces of the events which were built are saved in the DataHandler (i.e. copied from the stream files). If False, only a reference to the original data is saved. Defaults to False.
        :type copy_events: bool, optional
        :param reuse_triggers: If true, the triggers from a previous call of this function (which were saved in the DataHandler) are reused and only the event building is performed again (possibly with a different coincidence interval). Defaults to False.
        :type reuse_triggers: bool, optional
        :param interval: The coincidence interval for event building in microseconds, i.e. if a trigger lies within the specified interval around a trigger of another channel, they are collected to represent one event. Defaults to ``-+dt_us*record_length//4``.
        :type interval: Tuple[float], optional
        :param f_noise: The frequency (in events per hour) of empty noise traces to include. Defaults to 0, i.e. no noise is included.
        :type f_noise: float, optional
        :param kwargs: Additional keyword arguments forwarded to :func:`cait.versatile.trigger_of`.
        :type kwargs: Any

        **Example:**

        .. code-block:: python
        
            import cait as ai
            import cait.versatile as vai

            # Construct stream object
            stream = vai.Stream(hardware="vdaq2", src="path/to/stream_file.bin")
            
            print(f"Available channels: {stream.keys}")
            print(f"Available TP channels: {stream.tp_keys}")

            # Construct DataHandler
            dh = ai.DataHandler(record_length=2**13, nmbr_channels=2, sample_frequency=stream.sample_frequency)
            dh.set_filepath(path_h5="folder_name/", fname="z-score-triggered", appendix=False)
            dh.init_empty()

            # Load OF (saved in text file), has one channel
            # If you have some other DataHandler with the OF saved, you can use .from_dh instead
            of = vai.OF.from_file("path/to/OF_file")

            # Trigger ADC1 (phonon channel) and read ADC2 in coincidence (light channel)
            dh.trigger_zscore(stream, 
                              trigger_channels=["ADC1"],
                              of=of,
                              thresholds=[1e-3],
                              passive_channels=["ADC2"],
                              testpulse_channels=["DAC1", "DAC3"],
                              copy_events=True)
        """
        trigger_channels, passive_channels, testpulse_channels, controlpulses_above, calibration_channels, thresholds = _sanitize_input(
            stream=stream,
            trigger_channels=trigger_channels,
            passive_channels=passive_channels,
            testpulse_channels=testpulse_channels,
            controlpulses_above=controlpulses_above,
            calibration_channels=calibration_channels,
            thresholds=thresholds,
        )
            
        ofs = np.atleast_2d(of)
            
        if len(ofs) != len(np.hstack(trigger_channels)):
            raise ValueError(f"Optimum filter has to have as many channels as channels to trigger, i.e. len(of) must be len(np.hstack(trigger_channels)). Received {len(ofs)} and {len(np.hstack(trigger_channels))}.")
        
        # Used to split OF into single- or multi-channel components.
        # If channels were specified as tuples, the OF2D should be used
        # which needs a multi-channel filter.
        of_lens = [(len(x) if isinstance(x, tuple) else 1) for x in trigger_channels]

        trigger_fncs = [
            partial(
                vai.trigger_of2d if isinstance(ch, tuple) else vai.trigger_of, 
                # remove potential extra dimensions resulting from split
                of=np.squeeze(of),
                threshold=thresh, 
                **kwargs) 
            for of, thresh, ch in zip(
                # the split can produce empty arrays which are not iterated over in the zip
                np.split(ofs, np.cumsum(of_lens), axis=0), 
                thresholds, 
                trigger_channels
                )
        ]
        
        _trigger_helper(self, stream, trigger_channels, passive_channels, testpulse_channels, controlpulses_above, calibration_channels,
                        copy_events, reuse_triggers, interval, trigger_fncs, f_noise, "of")