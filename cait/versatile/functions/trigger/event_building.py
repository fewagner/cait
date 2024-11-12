from typing import List, Tuple

import numpy as np

from ..utils import timestamp_coincidence

def event_building(trigger_ts: List[List[int]],
                   trigger_phs: List[List[float]],
                   record_length: int,
                   dt_us: int,
                   n_slave_ch: int = 0,
                   tp_ts: List[List[int]] = None,
                   tpas: List[List[float]] = None,
                   interval: Tuple[float] = None,
                  ):
    """
    Build events from trigger timestamps (of multiple channels) and exclude testpulses if the respective information is provided.

    'trigger_ts' is a list of trigger timestamp lists for each channel that was triggered and 'trigger_phs' are the corresponding trigger pulse heights (with identical structure). If some channels are not triggered but read out in coincidence (i.e. as 'slave' channels), you can specify the number of such channels using 'n_slave_ch'. 

    Events are build as follows: Starting from the first list of timestamps in 'trigger_ts', the remaining lists are checked to be in coincidence with already existing timestamps. The default coincidence window (if ``interval=None``), is ``-+dt_us*record_length//4`` but can be adapted as needed.

    If you provide 'tp_ts' and 'tpas' (have to be both specified or left None), triggers within a record window of a testpulse are treated as testpulses. Note that you have to provide testpulse information for all channels INCLUDING 'slave' channels, i.e. 'tp_ts' has to have length ``len(trigger_ts) + n_slave_ch``.

    :param trigger_ts: A list of lists of timestamps of the channels that were triggered.
    :type trigger_ts: List[List[int]]
    :param trigger_phs: A list of lists of triggered pulse heights corresponding to the trigger timestamps above.
    :type trigger_phs: List[List[int]]
    :param record_length: The record length to use for building the events (usually a power of 2).
    :type record_length: int
    :param dt_us: The microsecond timebase that was used in the recording.
    :type dt_us: int
    :param n_slave_ch: The number of 'slave' channels, i.e. such that were not triggered but are merely read out in coincidence with other channels' triggers. Defaults to 0, i.e. all channels that are considered were triggered.
    :type n_slave_ch: int, optional
    :param tp_ts: A list of lists of timestamps of the testpulses corresponding to all channels. This includes 'slave' channels! I.e. you need a testpulse list for all triggered AND slave channels. If you don't have this information for the 'slave' channels, just put an empty list. Defaults to None, i.e. no pulses are counted as testpulses in the event building.
    :type tp_ts: List[List[int]], optional
    :param tpas: A list of lists of testpulse amplitudes of the testpulses corresponding to all channels. This includes 'slave' channels! I.e. you need a testpulse list for all triggered AND slave channels. If you don't have this information for the 'slave' channels, just put an empty list. Defaults to None, i.e. no pulses are counted as testpulses in the event building.
    :type tpas: List[List[float]], optional
    :param interval: The coincidence interval for event building in microseconds, i.e. if a trigger lies within the specified interval around a trigger of another channel, they are collected to represent one event. Defaults to ``-+dt_us*record_length//4``.
    :type interval: Tuple[float], optional

    :return:
        - ``event_ts`` - The timestamps of the events after event building, i.e. checking coincidences with other channels and excluding testpulses (if testpulse information was provided). Has shape ``(N,)``.
        - ``trigger_flag`` - A boolean flag of dimension ``(M, N)`` where ``M`` is the number of channels (including 'slave' channels). It is True for whichever event the respective channel triggered. For 'slave' channels, this is of course always False.
        - ``original_ts`` - Array of dimension ``(M, N)`` which stores the original trigger timestamps (before event building). It is set to ``-1`` wherever a channel did not trigger.
        - ``original_ph`` - Array of dimension ``(M, N)`` which stores the original trigger pulse heights (before event building). It is set to ``-1`` wherever a channel did not trigger.
        - ``all_tp_ts`` - The timestamps of all testpulses of all channels (after event building). Has shape ``(K,)``.
        - ``final_tpas`` - Array of dimension ``(M, K)`` which stores the testpulse amplitudes for all channels at times ``all_tp_ts``. It is set to ``404`` whenever a channel did not receive its own testpulse (but it was sent to another channel).
    :rtype: Tuple[numpy.ndarray]
    """
    
    # Make sure we have lists of lists (and no numpy arrays)
    if isinstance(trigger_ts, np.ndarray): trigger_ts.tolist()
    if isinstance(trigger_phs, np.ndarray): trigger_phs.tolist()
    if isinstance(tp_ts, np.ndarray): tp_ts.tolist()
    if isinstance(tpas, np.ndarray): tpas.tolist()

    if not len(trigger_ts) == len(trigger_phs):
        raise ValueError(f"'trigger_ts' and 'trigger_ph' must have the same length (number of triggered channels). Received {len(trigger_ts)} and {len(trigger_phs)}.")
    
    for i, ts in enumerate(trigger_ts):
        # Make sure we have lists of lists (and no numpy arrays)
        if isinstance(ts, np.ndarray):
            trigger_ts[i] = ts.tolist()
        if not isinstance(trigger_ts[i], list):
            raise TypeError(f"'trigger_ts' has to be a list of lists of integers.")
    
    for i, ph in enumerate(trigger_phs):
        # Make sure we have lists of lists (and no numpy arrays)
        if isinstance(ph, np.ndarray):
            trigger_phs[i] = ph.tolist()
        if not isinstance(trigger_phs[i], list):
            raise TypeError(f"'trigger_ph' has to be a list of lists of floats.")
    
    if not all([len(ts) == len(ph) for ts, ph in zip(trigger_ts, trigger_phs)]):
        raise ValueError(f"All lists in 'trigger_ts' and 'trigger_ph' must have the same length (number of triggers). Received {[len(x) for x in trigger_ts]} and {[len(x) for x in trigger_phs]}.")
    
    if ( not all([tp_ts is None, tpas is None]) ) and ( not all([tp_ts is not None, tpas is not None]) ):
        raise ValueError("'tp_ts' and 'tpas' have to be specified together.")
    
    if tp_ts is not None:
        if not len(tp_ts) == len(tpas):
            raise ValueError(f"'tp_ts' and 'tpas' must have the same length (number of testpulse channels). Received {len(tp_ts)} and {len(tpas)}.")
        
        for i, ts in enumerate(tp_ts):
            # Make sure we have lists of lists (and no numpy arrays)
            if isinstance(ts, np.ndarray):
                tp_ts[i] = ts.tolist()
            if not isinstance(tp_ts[i], list):
                raise TypeError(f"'tp_ts' has to be a list of lists of integers.")
    
        for i, tpa in enumerate(tpas):
            # Make sure we have lists of lists (and no numpy arrays)
            if isinstance(tpa, np.ndarray):
                tpas[i] = tpa.tolist()
            if not isinstance(tpas[i], list):
                raise TypeError(f"'tpas' has to be a list of lists of floats.")

        if not all([len(ts) == len(tpa) for ts, tpa in zip(tp_ts, tpas)]):
            raise ValueError(f"All lists in 'tp_ts' and 'tpas' must have the same length (number of testpulses). Received {[len(x) for x in tp_ts]} and {[len(x) for x in tpas]}.")
        
        if not len(tp_ts) == len(trigger_ts) + n_slave_ch:
            raise ValueError(f"The length of 'tp_ts' must match 'len(trigger_ts)+n_slave_ch', i.e. if specified, testpulse information is required for all triggered AND slave channels. Received {len(tp_ts)} and {len(trigger_ts)}+{n_slave_ch}.")

    # default coincidence window is just within the same record window (given by record_length and dt_us)
    rec_window_coinc = (-dt_us*record_length//4, dt_us*record_length//4)
    if interval is None: 
        interval = rec_window_coinc
        
    # Collect all testpulses into a coherent dataset, i.e. collect TPs sent at the same times into one event
    # and if only one channel received the TP, the TPA in all other channels is set to 404
    # for a commented version of this part, see procedure below for events which is practically identical
    if tp_ts is not None:
        all_tp_ts = tp_ts[0].copy()
        
        tp_sent_flag = [[True]*len(all_tp_ts)]
        
        for i, ts in enumerate(tp_ts[1:]):
            this_tp_ts = ts.copy()
            _, coinc_inds, outside = timestamp_coincidence(all_tp_ts, this_tp_ts, rec_window_coinc)

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

        for i, pa in enumerate(tpas):
            final_tpas[i, np.array(tp_sent_flag[i])] = np.array(pa)

    else:
        all_tp_ts = []
        final_tpas = []

    # These arrays will be identical to trigger_ts and trigger_phs, except when testpulse information
    # is provided. In that case, all triggers which are in coincidence with a testpulse are removed.
    trigger_ts_new, trigger_phs_new = [], []

    for ts, ph in zip(trigger_ts, trigger_phs):
        # if testpulse information is provided, trigger timestamps within a quarter record window around 
        # testpulse timestamps (regardless of the channel) are counted as such
        if tp_ts is not None:
            *_, outside = timestamp_coincidence(all_tp_ts, ts, rec_window_coinc)
            ts = list(np.array(ts)[outside])
            ph = list(np.array(ph)[outside])

        trigger_ts_new.append(ts)
        trigger_phs_new.append(ph)

    # build events
    event_ts = trigger_ts_new[0].copy()
    trigger_flag = [[True]*len(event_ts)]
    
    for i, ts in enumerate(trigger_ts_new[1:]):
        # determine coincidences
        _, coinc_inds, outside = timestamp_coincidence(event_ts, ts, interval)

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

    # Add False for slave channels to indicate that they did (of course) not trigger
    len_trigger_flags = len(trigger_flag[-1])
    for _ in range(n_slave_ch):
        trigger_flag.append([False]*len_trigger_flags)

    # also save the original trigger timestamps exactly like the trigger flag array
    # values of -1 indicate that the corresponding value does not exist
    # (because the channel didn't trigger separately for that event)
    original_ts = -1*np.ones(np.array(trigger_flag).shape, dtype=np.int64)
    original_ph = -1*np.ones(np.array(trigger_flag).shape, dtype=np.float32)
    for i, (t, p) in enumerate(zip(trigger_ts_new, trigger_phs_new)):
        original_ts[i, np.array(trigger_flag[i])] = np.array(t)
        original_ph[i, np.array(trigger_flag[i])] = np.array(p)

    return (np.array(event_ts), 
            np.array(trigger_flag), 
            np.array(original_ts), 
            np.array(original_ph), 
            np.array(all_tp_ts), 
            np.array(final_tpas))