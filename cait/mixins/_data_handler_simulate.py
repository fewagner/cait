import warnings
from functools import partial
from typing import List, Union

import h5py
import numpy as np

import cait.versatile as vai
from cait.versatile.datasources.stream.streambase import StreamBaseClass
from cait.versatile.iterators import PulseSimIterator

from ..data._baselines import calculate_mean_nps
from ..fit import pulse_template
from ..fit._saturation import scale_factor
from ..simulate._sim_pulses import simulate_events
from ._data_handler_trigger_collection import _sanitize_input


class SimulateMixin(object):
    """
    A Mixin Class for the DataHandler class with methods to simulate data sets.
    """

    def efficiency_sim_trigger_of(
        self,
        sim_ts: np.ndarray,
        sim_phs: np.ndarray,
        stream: StreamBaseClass,
        trigger_channels: Union[str, List[str]],
        of: np.ndarray,
        thresholds: Union[float, List[float]],
        sev: np.ndarray = None,
        sev_fitpars: List[List[float]] = None,
        shift_samples: List[List[int]] = None,
        shift_subsamples: List[List[float]] = None,
        passive_channels: Union[str, List[str]] = None,
        testpulse_channels: Union[str, List[str]] = None,
        tolerance_samples: int = 10,
        n_record_lens: int = 8,
        record_placement: int = 4,
        tag: str = "",
        preview: bool = False,
        **kwargs,
    ):
        """
        Perform a trigger efficiency simulation by superimposing a SEV onto random parts of a stream and running an optimum filter trigger to check whether they survive or not. See below for a description of the algorithm.

        The trigger-specific arguments to this function correspond to the ones of :func:`cait.mixins.TriggerCollectionMixin.trigger_of`, and the two are meant to be used in combination (i.e. the trigger efficiency function mimics the behavior of the trigger function). The arguments that the trigger function has but which are not featured here are irrelevant for the efficiency. Also refer to :func:`cait.mixins.TriggerCollectionMixin.trigger_of` for more information.

        :param sim_ts: The timestamps at which you want to simulate events. These cannot be within the first nor the last two record windows of the stream (because there, the filtering is unreliable). This is a 1d-array, i.e. even if you simulate pulses on multiple channels, they are simulated at identical timestamps. Events are simulated such that the SEV of the first trigger channel has its maximum at the given timestamp.
        :type sim_ts: np.ndarray
        :param sim_phs: The pulse heights of the pulses that you want to simulate. This has to have as many rows as there are trigger and passive channels (i.e. we simulate pulses on all channels which are of interest). 
        :type sim_phs: np.ndarray
        :param stream: The stream object including the channels that you want to trigger.
        :type stream: StreamBaseClass
        :param trigger_channels: The list of channel names to be triggered. Have to be present in ``stream.keys``. If you pass a tuple of channel names, the 2D optimum filter is applied to this combination.
        :type trigger_channels: List[Union[Tuple[str], str]]
        :param of: The optimum filter to use for triggering (Has to have one for each channel in ``trigger_channels``. If either of the trigger channels is a tuple, i.e. treated using the 2D optimum filter, they count as multiple channels. E.g. if you 2D optimum filter the first two channels together and the third channel with the regular optimum filter, ``of`` would have to have shape ``(3, kernel_length)``).
        :type of: np.ndarray
        :param thresholds: A list of trigger thresholds (in V) for each entry in ``trigger_channels``. If only one is specified, the respective threshold is used for all channels. Note that a tuple in ``trigger_channels``, i.e. a channel combination treated by a 2D optimum filter, needs only one threshold.
        :type thresholds: Union[float, List[float]]
        :param sev: The standard event to superimpose onto the stream (after it was scaled by ``sim_phs``). Has to have as many rows as there are trigger and passive channels. Cannot be set together with ``sev_fitpars``.
        :type sev: np.ndarray, optional
        :param sev_fitpars: The pulse shape fit parameters of the standard event to be superimposed onto the stream (after it was scaled by ``sim_phs``). The remaining parameters correspond to those of :func:`cait.fit.pulse_template`. Note that the time constants have to be given in milliseconds. Pulse shape parameters both of the 'traditional' 2-component model are supported as well as the n-component extension. Has to have as many rows as there are trigger and passive channels. The number of parameters may differ between channels (e.g. a 6-parameter 2-component model on one channel and an 8-parameter 3-component model on another); in that case the input is kept as a list of per-channel parameter arrays instead of a rectangular array. Cannot be set together with ``sev``.
        :type sev_fitpars: List[List[float]], optional
        :param shift_samples: An array of shift values (in samples) by which the ``sev`` or ``sev_fitpars`` should be offset from ``sim_ts`` before superimposing onto the stream chunks. This can be used to simulate slight onset variations between different channels. In such a case one would probably want to set all offsets for the first channel to zero and only vary the values for the remaining channels. Has to have as many rows as there are trigger and passive channels. Defaults to None, i.e. no shifts are applied and all simulated pulses are aligned such that the first trigger channel's SEV maximum sits on ``sim_ts``.
        :type shift_samples: List[List[int]], optional
        :param shift_subsamples: Simulates additional sub-sample shifts (as they would happen if the onset of the true pulse is not perfectly aligned with the sampling grid of the data acquisition). If you use a ``sev``, the array is linearly interpolated between samples. If you use ``sev_fitpars``, the model is evaluated at the intermediate points. Note that all values have to be in the interval [0, 1), corresponding to shifts between zero and one sample. The same shape requirements as for 'shift_samples' apply.
        :type shift_subsamples: List[List[float]]
        :param passive_channels: A list of channel names to be read out as 'passives'. Have to be present in ``stream.keys``. Defaults to None
        :type passive_channels: List[str], optional
        :param testpulse_channels: A list of channel names to be used as testpulses. Have to be present in ``stream.tp_keys``. Defaults to None
        :type testpulse_channels: List[str], optional
        :param tolerance_samples: Maximum number of samples that a trigger can deviate from ``sim_ts`` such that it is still considered a trigger. See also :class:`cait.versatile.TriggerSurvival`. Defaults to 10 samples.
        :type tolerance_samples: int, optional
        :param n_record_lens: The size of the stream chunks that are used for the simulation in multiples of the record window size. In principle, the higher this number, the better (because it more closely resembles the situation of placing it on the entire stream) but this comes at a performance cost. Has to be at least 4, but no noticeable differences are expected for any size larger than 6. Defaults to 8 record windows.
        :type n_record_lens: int, optional
        :param record_placement: You can choose where to place the ``sev`` or ``sev_fitpars`` onto the stream chunk (of length ``record_length*n_record_lens``) This parameter specifies after how many record windows into the chunk the SEV is placed. Defaults to 4 record windows into the stream chunk.
        :type record_placement: int, optional
        :param tag: An optional string to be added to the ``DataHandler`` groups that this function produces. E.g. if you want to run several simulations for different filters, you could run them with arguments ``tag='of1'`` etc. and the resulting groups would get a ``-of1`` suffix. Defaults to ``""``, i.e. no suffix.
        :type tag: str, optional
        :param preview: If True, a preview of the stream chunks superimposed with the scaled SEV, a filtered version thereof, and the trigger samples are shown. Meant for debugging purposes and/or finding appropriate values for ``tolerance_samples``, ``n_record_lens``, and ``record_placement``. Also see :class:`cait.versatile.TriggerSurvival`. Defaults to False.
        :type preview: bool, optional
        :param kwargs: Additional keyword arguments forwarded to :func:`cait.versatile.trigger_of`/:func:`cait.versatile.trigger_of2d`.
        :type kwargs: Any

        **Algorithm explanation:**

        In the trigger efficiency simulation one wants to study the trigger survival probability of an event (a pulse) randomly placed onto the stream with a given pulse height. The most direct way to do this is to take the entire stream and superimpose pulses at random times, then trigger the stream and check which of them survived the trigger and event building process. While conceptually straight forward to understand and implement, this procedure has a downside: There is an inherent limit on how many pulses one can simulate on a finite-length stream, because at some point, simulated pulses will pile-up with other simulated pulses. And since we want to study the efficiency for detecting *a single* event, artificially increasing the number of events on the stream drastically will impact the interpretation of this efficiency, i.e. the survival probability will be falsely decreased, and this effect has to be taken care of in the subsequent analysis. Note that as long as only few events are simulated, this method works fine. However, simulating as many events as possible is usually of interest to improve the statistical robustness of the simulation.

        The method employed here follows a slightly different approach. Note that if we just placed one event at a time onto the stream, the procedure above would be perfect. It would be highly inefficient though (because we would have to trigger the entire stream for each simulated event). As a tradeoff, here, we place single simulated events on smaller chunks of the stream and only trigger the chunks. Considering that the trigger's history is 'forgotten' after a few record windows anyways, this simplification will give identical results to placing single events on the full stream. The selected chunks **may overlap** and we may choose **infinitly many** such chunks (and hence events to simulate) because we do it **one simulated event at a time** which avoids any simulation pile-up issues by design. The algorithm proceeds as follows:

            1. Read stream chunks of size ``record_length*n_record_lens`` from the stream. The chunks are aligned such that ``sim_ts`` correspond to the maximum of the SEV if we place ``sev`` (or ``sev_fitpars``) ``record_placement`` record windows into the chunk (magenta pulse). If you handle multiple channels, the maximum of the first SEV (``sev[0]``) defines the position. This position also defines the **target index** for the trigger: If, later, a trigger is detected within at most ``tolerance_samples`` around this index, the simulated event is considered *triggered*.
            2. Shift the SEV by ``shift_samples``, multiply it by ``sim_phs`` and place it on the stream (dotted pulse). This happens for each of the channels separately, i.e. separate shifts and pulse heights are applied according to the values in the arrays ``shift_samples`` and ``sim_phs``. Usually, you will not apply a shift to the first channel but only relative shifts to the other channel(s). (Notice that the target index for triggering is *not necessarily* the maximum index of the *simulated event* if you simulate shifts to the first channel! It's always the maximum of the non-shifted SEV.) In the illustration below, the first channel was not shifted, but the second was.
            3. Run the OF trigger on the chunk. Note that the first record window is not searched for maximums exceeding the threshold (because the trace cannot be reliably filtered). The same holds for the last record window. The second to last record window *might* be searched for a maximum if a sample exceeds the threshold just before the beginning of the second to last record window (cf. :func:`cait.versatile.functions.trigger.triggerbase.trigger_base`). In the illustration below, the trigger found an already existing pulse (dark blue) because it is larger than the simulated (dashed dark blue) pulse, i.e. the real pulse *shadows* the simulated pulse. If there was no larger (real) pulse nearby, the simulated pulse would have been triggered (if it is larger than the threshold). This *shadowing* effect is a central aspect that the efficiency simulation should quantify.
            4. The maximum of the triggered pulse defines the timestamp of the event that is built: It is aligned with the 1/4th-mark of the record window as usual. Note that this event is only considered *triggered* by the efficiency simulation if the trigger timestamp is within ``tolerance_samples`` around the target index. In case of the illustration, this would most likely not have been the case (unless you set it to an unreasonably high number).

        .. image:: media/efficiency_sim_algorithm.png

        The algorithm described above applies to a single trigger channel. If you trigger multiple, the *target index* is individually defined for each channel as the maximum of the (non-shifted) SEV of that channel (Note, however, that the *simulation timestamp* is still defined by the maximum of the (non-shifted) SEV of the first channel). The triggering step is performed for all channels individually. A simulated pulse is considered *triggered* if **either** of the channels triggered (i.e. had a trigger within ``tolerance_samples`` of the *target index*). In case more than one channel triggers, the *event timestamp* is aligned at the maximum of the **first** channel that triggered (i.e. the first if it triggered, the second if the second triggered but the first one didn't, etc.). 

        If testpulse channels are specified, an additional step marks all simulated pulses **not triggered** if they are within half a record window of a testpulse, regardless of whether they triggered or not. 
        Calibration channel handling for the efficiency calculation is not yet implemented, and can be added in the future if necessary.

        **Example:**

        .. code-block:: python

            import numpy as np
            import scipy as sp

            import cait as ai
            import cait.versatile as vai

            record_length = 2**14
            N_sim = 1000

            # Set up stream (here just mock, in your case an actual stream)
            stream = vai.MockStream(seed=137, rate_Hz=2)

            # Set up DataHandler (in your case, you probably have one already)
            dh = ai.DataHandler(record_length=record_length, nmbr_channels=2, sample_frequency=stream.sample_frequency)
            dh.set_filepath(path_h5="", fname="eff-test", appendix=False)
            dh.init_empty()

            # You can either use a SEV (array) or fit parameters.
            # Here we use fit parameters.
            # [t0, An, At, tau_n, tau_in, tau_t]
            sev_fitpars = [
                [-2.56, 0.50, 0.50, 3.00, 1.00, 10.00], 
                [-2.56, 0.50, 0.50, 0.30, 0.10, 10.00]
            ]

            of = vai.MockData().of[0]

            # Random timestamps for simulation (always a good
            # idea to sort them so that we don't have to jump
            # back and forth in the simulation)
            sim_ts = np.sort(
                sp.stats.randint.rvs(
                    stream.time[0] + 6*stream.dt_us*record_length, 
                    stream.time[-1] - 4*stream.dt_us*record_length, 
                    size=N_sim
                )
            )
            # Random pulse heights between 0 and 1 (you probably got
            # some other range in a real world example)
            sim_phs = [
                sp.stats.uniform.rvs(size=N_sim),
                sp.stats.uniform.rvs(size=N_sim),
            ]
            # Random shifts (only for second channel)
            shifts = [
                np.zeros(N_sim),
                sp.stats.randint.rvs(-5, 5, size=N_sim)
            ]

            # Run the simulation.
            # Note that here, we trigger on Ch0 and Ch1 is read in coincidence.
            # Therefore, of must have one channel.
            # We still simulate for Ch1, though, meaning that we need sim_phs,
            # sev_fitpars, and shift_samples for both Ch0 AND Ch1. 
            dh.efficiency_sim_trigger_of(
                stream=stream,
                trigger_channels=["Ch0"],
                passive_channels=["Ch1"],
                testpulse_channels=["TP0", "TP1"],
                of=of,
                thresholds=[0.1],
                sim_ts=sim_ts,
                sim_phs=sim_phs,
                #preview=True, # Uncomment to see a preview of the trigger before you run the simulation
                sev_fitpars=sev_fitpars,
                shift_samples=shifts,
            )

            # Two groups have been created in the DataHandler. 
            # 'trig-eff-sim' contains trigger information and the 
            # simulation chunks, 'events-eff-sim' contains the particle
            # traces which survived the procedure. Check them out using
            dh.content()

            # You can look at the stream chunks that were simulated (only contains trigger channels):
            vai.Preview(dh.get_event_iterator("trig-eff-sim").with_processing(vai.RemoveBaseline()))

            # You can look at the events that survived (contains trigger and passive channels):
            vai.Preview(dh.get_event_iterator("events-eff-sim").with_processing(vai.RemoveBaseline()))

            # Basic histogram to show what was simulated and what survived.
            # You will probably make more sophisticated plots than this.
            simulated_phs = dh['trig-eff-sim/simulated_phs', 0]
            survived_trigger = dh['trig-eff-sim/flag_survived_trigger']
            survived_tp = dh['trig-eff-sim/flag_survived_tp']

            vai.Histogram(
                {
                    "simulated": simulated_phs, 
                    "triggered": simulated_phs[survived_trigger],
                    "survived tp": simulated_phs[survived_tp*survived_trigger],
                }, 
                bins=np.linspace(0, 1, 10),
                xlabel="Simulated pulse height (V)"
            )

            # Have a look at simulated vs. reconstructed pulse heights:
            # (Note: reconstructed pulse heights for channels that were not
            # triggered are set to -1)
            vai.Scatter(
                x=dh["events-eff-sim/simulated_phs", 0], 
                y=dh["events-eff-sim/reconstructed_phs", 0],
                xlabel="Simulated pulse heights (V)",
                ylabel="Reconstructed pulse heights (V)",
            )

            # ... Now you can run additional calculations/cuts on the
            # 'events-eff-sim' group and also calculate the cut efficiency.
        """
        # Ensures that all inputs are lists
        trigger_channels, passive_channels, testpulse_channels, *_, thresholds = _sanitize_input(
            stream=stream,
            trigger_channels=trigger_channels,
            passive_channels=passive_channels,
            testpulse_channels=testpulse_channels,
            controlpulses_above=None,
            calibration_channels=None,
            thresholds=thresholds,
        )

        if np.sum([x is None for x in [sev, sev_fitpars]]) != 1:
            raise ValueError(f"You have to specify EITHER a sev or its fit parameters. At least one. Not both.")

        # Save a variable to later check whether we are using
        # a SEV (array) or fit parameters of the pulse shape model
        using_fit = sev_fitpars is not None
        sev_or_pars = sev_fitpars if using_fit else sev

        # Want 1d timestamps but 2d pulse heights, sev, and of
        sim_ts = np.array(sim_ts).flatten()

        sim_phs, of = np.atleast_2d(sim_phs), np.atleast_2d(of)

        if using_fit:
            # Channels may use pulse shape models with different numbers of
            # parameters (e.g. a 6-parameter 2-component model on one channel
            # and an 8-parameter 3-component model on the other). Such input is
            # ragged and cannot be stored in a rectangular array, so we only
            # build one when all channels agree and otherwise keep a list of
            # per-channel 1d parameter arrays.
            if np.ndim(sev_or_pars[0]) == 0:  # single channel given as flat list
                sev_or_pars = [sev_or_pars]
            sev_or_pars = [np.asarray(p, dtype=float).ravel() for p in sev_or_pars]
            ragged_fitpars = len({p.size for p in sev_or_pars}) > 1
            if not ragged_fitpars:
                sev_or_pars = np.array(sev_or_pars)
        else:
            ragged_fitpars = False
            sev_or_pars = np.atleast_2d(sev_or_pars)

        # Needed for 2d-of support ... 
        # All 'channels' that are an entity as seen by the trigger.
        # (2d-of channels which are given by tuples are treated
        # as a single entity by the trigger).
        trigger_groups = trigger_channels.copy()
        # The flattened list of all channels (each of these
        # needs an OF!)
        trigger_channels = np.hstack(trigger_channels).tolist()

        
        # Also add passive channels because even though 
        # they are not triggered, they are considered during 
        # event building.
        if passive_channels is not None:
            all_groups = trigger_groups + passive_channels
        else:
            all_groups = trigger_groups

        # All channel names (with potential 2d-of tuples expanded).
        # These are the channel which are eventually read and saved
        # as particle events in the DataHandler.
        all_channels = np.hstack(all_groups).tolist()

        n_trig_ch = len(trigger_channels)
        n_total_groups = len(all_groups)
        n_total_ch = len(all_channels)
        
        # Used to split OF into single- or multi-channel components.
        # If channels were specified as tuples, the OF2D should be used
        # which needs a multi-channel filter.
        of_lens = [(len(x) if isinstance(x, tuple) else 1) for x in trigger_groups]

        if len(of) != n_trig_ch:
            raise ValueError(f"Optimum filter has to have as many channels as channels to trigger, i.e. len(of) must be len(np.hstack(trigger_channels)). Received {len(of)} and {n_trig_ch}.")

        if sim_phs.shape[0] != n_total_ch:
            raise ValueError(f"Pulse heights are needed for all channels (including passive ones). Got pulse heights of shape {sim_phs.shape} and in total {n_total_ch} channel(s).")
        if len(sev_or_pars) != n_total_ch:
            raise ValueError(f"SEVs are needed for all channels (including passive ones). Got sev/sev_fitpars for {len(sev_or_pars)} channel(s) and in total {n_total_ch} channel(s).")
        if sim_phs.shape[-1] != len(sim_ts):
            raise ValueError(f"The number of timestamps must agree with the number of pulse heights. Got {len(sim_ts)} and {sim_phs.shape[-1]}.")
        
        if shift_samples is not None:
            shift_samples = np.atleast_2d(shift_samples).astype(np.int32)
            if shift_samples.shape != sim_phs.shape:
                raise ValueError(f"The shapes of 'shift_samples' and 'sim_phs' have to be identical. Got {shift_samples.shape} and {sim_phs.shape}.")
            
        if shift_subsamples is not None:
            shift_subsamples = np.atleast_2d(shift_subsamples).astype(np.float32)
            if shift_subsamples.shape != sim_phs.shape:
                raise ValueError(f"The shapes of 'shift_subsamples' and 'sim_phs' have to be identical. Got {shift_subsamples.shape} and {sim_phs.shape}.")
            if not np.all((shift_subsamples>=0)*(shift_subsamples<1)):
                raise ValueError("All values in 'shift_subsamples' have to be in the interval [0, 1).")

        rl = self.record_length
            
        if n_record_lens < 4:
            raise ValueError(f"Input 'n_record_lens' has to be at least 4 (the more the better).")
        if record_placement > n_record_lens-2:
            raise ValueError(f"Input 'record_placement' has to be smaller than 'n_record_lens-2'.")
            
        if np.any(sim_ts<stream.time[0]+stream.dt_us*rl*(record_placement+2)):
            raise ValueError(f"The earliest possible timestamp for simulation is stream.dt_us*record_length*(record_placement+2). Make sure that all timestamps are well within the stream.")
        if np.any(sim_ts>stream.time[-1]-stream.dt_us*rl*(n_record_lens-record_placement)):
            raise ValueError(f"The latest possible timestamp for simulation is stream.time[-1]-stream.dt_us*record_length*(n_record_lens-record_placement). Make sure that all timestamps are well within the stream.")
        
        appendix = f"-{tag}" if tag else ""

        # The time array for the (small) record window.
        # We will evaluate the SEV's first channel (or the respective)
        # fitpars on this array and check if the maximum is at t=0.
        # This is usually the case but if not, we have to later account
        # for that because the simulation timestamp is aligned with the
        # maximum.
        _compensate_t = (np.arange(rl) - rl/4)*stream.dt_us/1000
        
        # PREPARING SEV TO BE USED IN PULSE_SIM_ITERATOR
        if using_fit:
            # We place the pulse in the (extended) record window as
            # defined by 'record_placement'.
            # The time=0 of the fitpars is adjusted to fall onto
            # pulse_sim_index
            # The (extended) iterator's time array is used to 
            # evaluate the fitpars.
            # (Kept as a list of per-channel parameter arrays if the models
            # have different numbers of parameters.)
            used_pars = [np.copy(p) for p in sev_or_pars[:n_trig_ch]]
            if not ragged_fitpars:
                used_pars = np.atleast_2d(used_pars)
            of_trigger = [np.squeeze(x) for x in np.split(of, np.cumsum(of_lens), axis=0)[:-1]]
            
            pulse_sim_index = record_placement * rl + np.argmax(
                pulse_template(
                    (np.arange(rl) - rl/4)*stream.dt_us/1000, 
                    *used_pars[0]
                    )
                )

            peak_t = _compensate_t[np.argmax(pulse_template(_compensate_t, *sev_or_pars[0]))]
            iterator_kwargs = dict(sev_fitpars=used_pars)
        else:
            # The SEV that is placed on the stream chunks for triggering
            # is first sanitized with a window function (this prevents
            # high frequency artifacts). Reducing the pulse's area means
            # that we also have to re-normalize the OF!
            # (This is only done for the channels which are actually triggered)
            window_sev = [vai.TukeyWindow()(sev_or_pars[i]) for i in range(n_trig_ch)]
            of_trigger = []
            for of_, sev_, ch_ in zip(
                # the split can produce empty arrays which are not iterated over in the zip because trigger_groups is shorter
                np.split(of, np.cumsum(of_lens), axis=0),
                np.split(window_sev, np.cumsum(of_lens), axis=0),
                trigger_groups,
                ):
                of_, sev_ = np.squeeze(of_), np.squeeze(sev_)

                if isinstance(ch_, tuple):
                    # This is for 2d-OF
                    scale = np.max(vai.OptimumFiltering2D(of_)(sev_))
                else:
                    # This is for regular OF
                    scale = np.max(vai.OptimumFiltering(of_)(sev_))
                    
                of_trigger.append(of_/scale)
              
            # We now place it on the stream chunk as defined by the record_placement
            padded_sev = np.zeros((n_trig_ch, n_record_lens*rl))
            for i, s in enumerate(window_sev):
                padded_sev[i, record_placement*rl:(record_placement+1)*rl] = np.array(s)
            
            pulse_sim_index = np.argmax(padded_sev[0])
            peak_t = _compensate_t[np.argmax(sev_or_pars[0])]
            iterator_kwargs = dict(sev=padded_sev)

        # Usually zero (see above and below when saving particle events to the DataHandler)
        argmax_offset = int(peak_t*1000/stream.dt_us)

        # CONSTRUCT PULSE_SIM_ITERATOR
        chunk_iterator = PulseSimIterator(
            stream.get_event_iterator(
                trigger_channels, 
                record_length=n_record_lens*rl, 
                timestamps=sim_ts, 
                # This alignment ensures that the peak positions
                # lie at the timestamps.
                alignment=pulse_sim_index/(n_record_lens*rl),
            ),
            **iterator_kwargs,
            pulse_heights=sim_phs[:n_trig_ch,:],
            shift_samples=shift_samples[:n_trig_ch,:] if shift_samples is not None else None,
            shift_subsamples=shift_subsamples[:n_trig_ch,:] if shift_subsamples is not None else None,
        )
        
        # DEFINE TARGET INDEX FOR TRIGGER SURVIVAL FUNCTION
        # For 1d-of channels, the target index is the maximum position
        # of the template/evaluated pulse shape.
        # For 2d-of channels, the target index is the mean of all 
        # the participating channel's max positions.
        # (Q: would it make more sense to use the filter max position
        # here? I doubt that it will make any difference)
        # This index is the same irrespective of the added shifts.
        if using_fit:
            _target_inds = [
                np.argmax(pulse_template(chunk_iterator.t, *p)) 
                for p in used_pars
            ]
        else:
            _target_inds = [np.argmax(ps) for ps in padded_sev]

        # (now has length 'len(trigger_groups)')
        target_inds = [int(np.mean(x)) for x in np.split(_target_inds, np.cumsum(of_lens))[:-1]]

        # INITIALIZE TRIGGER SURVIVAL FUNCTION
        fns = [
            vai.TriggerSurvival(
                trigger_fnc=partial(
                    vai.trigger_of2d if isinstance(g, tuple) else vai.trigger_of, 
                    of=ot, 
                    threshold=th,
                    **kwargs,
                    ),
                target_ind=tind,
                tolerance_samples=tolerance_samples
            )
            for tind, ot, th, g in zip(
                target_inds, 
                of_trigger, 
                thresholds,
                trigger_groups,
                )
        ]

        # This (helper) variable looks something like this: [[0, 1], 2, 3] etc.
        # It is needed to slice the correct parts of the iterator.
        chs = [
            (int(x[0]) if x.size==1 else x.tolist()) 
            for x in np.split(np.arange(n_trig_ch), np.cumsum(of_lens))[:-1]
        ]
        
        if preview:
            for ch, f in zip(chs, fns):
                vai.Preview(chunk_iterator[ch], f, backend="plotly")
            return
        
        # Initialize array with as many channels as total channels (including passive).
        # Triggers will be false for passive, values and inds will be -1.
        # In the next step, the rows of those arrays which correspond to trigger_channels
        # will be filled accordingly.
        separate_trigger = np.zeros((n_total_groups, len(chunk_iterator)), dtype=bool)
        trigger_val = -1*np.ones((n_total_groups, len(chunk_iterator)), dtype=np.float32)
        trigger_ind = -1*np.ones((n_total_groups, len(chunk_iterator)), dtype=np.int32)

        for i, (ch, f) in enumerate(zip(chs, fns)):
            separate_trigger[i, :], trigger_val[i, :], trigger_ind[i, :] = vai.apply(
                f, 
                chunk_iterator[ch], 
                pb_prefix=f"Triggering channel {ch}",
            )

        # NOTE: In the actual dh.trigger_of, we do sophisticate event building.
        # Of course, we want to stick as closely to this procedure here, to get
        # a reliable simulation of the trigger efficiency. 
        # Nevertheless, the idea of the simulation is to look at one event at a
        # time and put it on the stream (otherwise, there could be simulation pile-up).
        # Hence, the event building process would just be 'take this one event, build
        # an event if at least one of the channels triggered, remove it if it is in 
        # coincidence with a testpulse'. 
        # Here, we avoid looping through all events and calling vai.event_building on it.
        # Reasoning: The difference between here and the 'actual' triggering is, that we
        # already pre-defined the event. Hence, by just checking whether or not either of
        # the channels triggered the simulated pulse, and whether or not it would be 
        # shadowed by a testpulse is sufficient.

        # We consider an event to be triggered, if either of the channels triggered.
        # (the sum is a logical or)
        did_trigger = np.sum(separate_trigger, axis=0, dtype=bool)

        # Move event_ts to highest ranking trigger timestamp.
        # If events didn't trigger, use sim_ts.
        event_ts = np.copy(sim_ts)
        not_yet_built = np.ones(len(sim_ts), dtype=bool)
        for triggered, ind in zip(separate_trigger, trigger_ind):
            flag = triggered*not_yet_built
            event_ts[flag] += (ind[flag] - pulse_sim_index)*stream.dt_us
            not_yet_built[flag] = False
        
        # Exclude testpulses if 'testpulse_channels' are specified.
        # Otherwise, this will just be an array of Trues.
        # Important: use the actual event_ts here, not the sim_ts.
        survived_tp = np.ones(len(chunk_iterator), dtype=bool)
        
        if testpulse_channels is not None:
            tp_ts = list()
            for tp_ch in testpulse_channels:
                tp_ts.extend(np.array(stream.tp_timestamps[tp_ch]).tolist())
            
            inside, *_ = vai.timestamp_coincidence(
                # If a TP timestamp exists in more than one channel, we just
                # treat it once. Also, TP timestamps must be sorted.
                np.unique(np.sort(tp_ts)), 
                event_ts, 
                (-stream.dt_us*rl//4, stream.dt_us*rl//4)
            )
            survived_tp[inside] = False

        # SAVE TO DATAHANDLER
        set_kwargs = dict(change_existing=True, overwrite_existing=True)
        group = "trig-eff-sim"+appendix
        group_events = "events-eff-sim"+appendix

        # Save information about trigger efficiency simulation
        self.include_event_iterator(group, chunk_iterator, copy_events=False)
        self.set(
            group, 
            trigger_flag=separate_trigger,
            flag_survived_trigger=did_trigger,
            flag_survived_tp=survived_tp,
            dtype=bool,
            **set_kwargs,
        )
        self.set(
            group, 
            trigger_index=trigger_ind,
            event_timestamps=event_ts,
            **{
                **(
                    dict(simulated_shifts=np.atleast_2d(shift_samples)) 
                    if shift_samples is not None else dict()
                ),
                **(
                    dict(simulated_subshifts=np.atleast_2d(shift_subsamples)) 
                    if shift_subsamples is not None else dict()
                )
            },
            dtype=np.int32,
            **set_kwargs,
        )
        self.set(
            group, 
            simulated_phs=sim_phs,
            reconstructed_phs=trigger_val,
            dtype=np.float32,
            **set_kwargs,
        )

        # Save events in a separate group for subsequent (cut) efficiency studies.
        # (Only save (surviving) particle events to events group).
        # IMPORTANT: here, we use the event timestamps, NOT the trigger timestamps.
        # Because the actual trigger also aligns around the trigger timestamps 
        # (important for subsequent cuts).
        # ALSO: The shifts (if present) are applied BEFORE (such that the pulses
        # are in any case aligned around the maximum!)
        event_flag = survived_tp*did_trigger

        # The timestamps that we read now are NOT the simulation timestamps
        # anymore! We have to account for that using a shift (regardless
        # of whether an additional shift was given or not).
        if shift_samples is None: 
            mod_shift_samples = np.zeros((n_total_ch, len(event_flag)))
        else:
            mod_shift_samples = shift_samples.copy()

        offset_samples = (sim_ts - event_ts)//stream.dt_us
        mod_shift_samples += offset_samples
        
        # Furthermore, if the SEV's maximum was not previously fixed
        # at t=0 (but if e.g. t=0 corresponds to the onset), we also 
        # have to account for that fact (usually, this added shift will
        # be zero because in most analyses, the maximum is centered at 0).
        mod_shift_samples -= argmax_offset

        event_iterator = PulseSimIterator(
            stream.get_event_iterator(
                all_channels, 
                record_length=rl, 
                timestamps=event_ts[..., event_flag],
            ),
            # PulseSimIterator handles sev/sev_fitpars
            sev=sev,
            # Use the sanitized parameters (possibly a list of per-channel
            # arrays if the models have different numbers of parameters).
            sev_fitpars=sev_or_pars if using_fit else None,
            pulse_heights=sim_phs[..., event_flag],
            shift_samples=mod_shift_samples[..., event_flag],
            # PulseSimIterator handles remaining sub-sample shifts
            shift_subsamples=shift_subsamples[..., event_flag] if shift_subsamples is not None else None,
        )
        
        self.include_event_iterator(group_events, event_iterator, copy_events=False)
        self.set(
            group_events, 
            simulated_phs=sim_phs[..., event_flag],
            reconstructed_phs=trigger_val[..., event_flag],
            dtype=np.float32,
            **set_kwargs,
        )
        self.set(
            group_events, 
            simulated_ts=sim_ts[..., event_flag],
            dtype=np.int32,
            **set_kwargs,
        )

    # Simulate Dataset with specific classes
    def simulate_pulses(self,
                        path_sim,
                        size_events=0,
                        size_tp=0,
                        size_noise=0,
                        take_idx=None,
                        ev_ph_intervals=[[0, 1], [0, 1]],
                        ev_discrete_phs=None,
                        name_appendix='',
                        exceptional_sev_naming=None,
                        channels_exceptional_sev=[0],
                        tp_ph_intervals=[[0, 1], [0, 1]],
                        tp_discrete_phs=None,
                        t0_interval=[-20, 20],  # in ms
                        fake_noise=False,
                        store_of=True,
                        rms_thresholds=[1, 1],
                        lamb=0.01,
                        sample_length=None,
                        assign_labels=[1],
                        start_from_bl_idx=0,
                        saturation=False,
                        reuse_bl=False,
                        pulses_per_bl=1,
                        ps_dev=False,
                        dtype='float32',
                        indiv_tpas=False):
        """
        Simulates a data set of pulses by superposing the fitted SEV with fake or real noise.

        This method was used to simulate events in "F. Wagner, Machine Learning Methods for the Raw Data Analysis
        of crypgenic Dark Matter Experiments",
        available via https://doi.org/10.34726/hss.2020.77322 (accessed on the 9.7.2021).

        :param path_sim: The full path where to store the simulated data set.
        :type path_sim: string
        :param size_events: The number of events to simulate; if >0 we need a sev in the hdf5.
        :type size_events: int
        :param size_tp: The number of testpulses to simulate; if >0 we need a tp-sev in the hdf5.
        :type size_tp: int
        :param size_noise: The number of noise baselines to simulate.
        :type size_noise: int
        :param take_idx: Take only these event indices for the simulation. Overwrites start_from_bl_idx and rms_thresholds.
        :type take_idx: list
        :param ev_ph_intervals: The interval in which the pulse heights
            are continuously distributed.
        :type ev_ph_intervals: list of NMBR_CHANNELS 2-tuples or lists
        :param ev_discrete_phs: The discrete values, from which the pulse heights
            are uniformly sampled. If the ph_intervals argument is set, this option will be ignored. This should be one
            list per channel with have same length. The simulation is done correlated, i.e. the same index from the lists
            is chosen for all channels. This way e.g. light yields can be simulated.
        :type ev_discrete_phs: list of NMBR_CHANNELS lists
        :param name_appendix: A string that is appended to the group name stdevent, which contains the standard event
            that is used for simulation. This concerns only the simulation of event pulses and has no effect on the
            test pulses.
        :type name_appendix: string
        :param exceptional_sev_naming: If set, this is the full group name in the HDF5 set for the
            sev used for the simulation of events - by setting this, e.g. carrier events can be
            simulated. Attention! The exceptional standard events are with version 1.0 no longer maintained. Please use
            the name_appendix argument instead!
        :type exceptional_sev_naming: string or None
        :param channel_exceptional_sev: The channels for that the exceptional sev is
            used, e.g. if only for phonon channel, choose [0], if for botch phonon and light, choose [0,1].
        :type channel_exceptional_sev: list of ints
        :param tp_ph_intervals: Analogous to ev_ph_intervals, but for the testpulses.
        :type tp_ph_intervals: list of NMBR_CHANNELS 2-tuples or lists
        :param tp_discrete_phs: Analogous to ev_ph_intervals, but for the testpulses. This should be one
            list per channel with have same length. The simulation is done correlated, i.e. the same index from the lists
            is chosen for all channels. This way e.g. light yields can be simulated.
        :type tp_discrete_phs: list of NMBR_CHANNELS lists
        :param t0_interval: The interval from which the pulse onset are continuously sampled.
        :type t0_interval: 2-tuple or list
        :param fake_noise: If True the noise will be taken not from the measured baselines from the
            hdf5 set, but simulated.
        :type fake_noise: bool
        :param store_of: If True the optimum filter will be saved to the simulated datasets.
        :type store_of: bool
        :param rms_thresholds: Above which value noise baselines are excluded for the
            distribution of polynomial coefficients (i.e. a parameter for the fake noise simulation), also a
            cut parameter for the noise baselines from the h5 set if no fake ones are taken.
        :type rms_thresholds: list of two floats
        :param lamb: A parameter for the fake baseline simulation, decrease if calculation time is too long.
        :type lamb: float
        :param sample_length: The length of one sample in milliseconds (if None, it is calculated from the sample
            frequency).
        :type sample_length: float
        :param assign_labels: Pre-assign a label to all the simulated events; tp and noise are
            automatically labeled, the length of the list must match the list channels_exceptional_sev.
        :type assign_labels: list of ints
        :param start_from_bl_idx: The index of baselines that is as first taken for simulation.
        :type start_from_bl_idx: int
        :param saturation: If true apply the logistics curve to the simulated pulses.
        :type saturation: bool
        :param reuse_bl: If True the same baselines are used multiple times to have enough of them
            (use this with care to not have identical copies of events).
        :type reuse_bl: bool
        :param pulses_per_bl: Number of pulses to simulate per one baseline --> gets multiplied to size!!
        :type pulses_per_bl: int
        :param ps_dev: If True the pulse shape parameters are modelled with deviations. Attention! This will always
            model TUM40-like phonon pulse shapes! The light channel is not affected by this features. Generally, it is
            not clear how well the deviations model the actual deviations in measured data, so please handle this
            feature with care.
        :type ps_dev: bool
        :param dtype: The data format of the simulated raw data events array.
        :type dtype: string
        :param indiv_tpas: Write individual TPAs for the all channels. This results in a testpulseamplitude dataset
            of shape (nmbr_channels, nmbr_testpulses). Otherwise we have (nmbr_testpulses).
        :type indiv_tpas: bool
        """

        assert pulses_per_bl == 1, 'Only 1 pulse per baseline implemented!'

        if exceptional_sev_naming is not None:
            warnings.warn('The exceptional standard events are with version 1.0 depricated. '
                          'Please use the name_appendix argument instead!')

        if sample_length is None:
            sample_length = 1000 / self.sample_frequency

        # create file handle
        with h5py.File(path_sim, 'w') as f, h5py.File(self.path_h5, 'r') as f_read:

            nmbr_thrown_events = 0
            nmbr_thrown_testpulses = 0

            if size_events > 0:
                print('Simulating Events.')
                data = f.create_group('events')
                data.create_dataset(name='event',
                                    shape=(self.nmbr_channels, size_events * pulses_per_bl, self.record_length),
                                    dtype=dtype)
                data.create_dataset(name='true_ph',
                                    shape=(self.nmbr_channels, size_events * pulses_per_bl),
                                    dtype=float)
                data.create_dataset(name='true_onset',
                                    shape=(size_events * pulses_per_bl,),
                                    dtype=float)
                if not fake_noise:
                    data.create_dataset(name='hours',
                                        shape=(size_events * pulses_per_bl,),
                                        dtype=float)
                    if 'time_s' in f_read['noise']:
                        data.create_dataset(name='time_s',
                                            shape=(size_events * pulses_per_bl,),
                                            dtype=float)
                        data.create_dataset(name='time_mus',
                                            shape=(size_events * pulses_per_bl,),
                                            dtype=float)

                events, phs, t0s, nmbr_thrown_events, hours, time_s, time_mus = simulate_events(
                    path_h5=self.path_h5,
                    type='events',
                    name_appendix=name_appendix,
                    size=size_events,
                    record_length=self.record_length,
                    nmbr_channels=self.nmbr_channels,
                    ph_intervals=ev_ph_intervals,
                    discrete_ph=ev_discrete_phs,
                    exceptional_sev_naming=exceptional_sev_naming,
                    channels_exceptional_sev=channels_exceptional_sev,
                    t0_interval=t0_interval,  # in ms
                    fake_noise=fake_noise,
                    use_bl_from_idx=start_from_bl_idx,
                    take_idx=take_idx,
                    rms_thresholds=rms_thresholds,
                    lamb=lamb,
                    sample_length=sample_length,
                    saturation=saturation,
                    reuse_bl=reuse_bl,
                    ps_dev=ps_dev)

                if not fake_noise:
                    data['hours'][:size_events] = hours
                    if 'time_s' in f_read['noise']:
                        data['time_s'][:size_events] = time_s
                        data['time_mus'][:size_events] = time_mus

                data['event'][:, :size_events, :] = events
                data['true_ph'][:, :size_events] = phs
                data['true_onset'][:size_events] = t0s

                labels = np.ones([self.nmbr_channels, size_events])
                for c in channels_exceptional_sev:
                    labels[c] *= assign_labels[c]

                data.create_dataset(name='labels', data=labels)
                data['labels'].attrs.create(name='unlabeled', data=0)
                data['labels'].attrs.create(name='Event_Pulse', data=1)
                data['labels'].attrs.create(name='Test/Control_Pulse', data=2)
                data['labels'].attrs.create(name='Noise', data=3)
                data['labels'].attrs.create(name='Squid_Jump', data=4)
                data['labels'].attrs.create(name='Spike', data=5)
                data['labels'].attrs.create(name='Early_or_late_Trigger', data=6)
                data['labels'].attrs.create(name='Pile_Up', data=7)
                data['labels'].attrs.create(name='Carrier_Event', data=8)
                data['labels'].attrs.create(name='Strongly_Saturated_Event_Pulse', data=9)
                data['labels'].attrs.create(name='Strongly_Saturated_Test/Control_Pulse', data=10)
                data['labels'].attrs.create(name='Decaying_Baseline', data=11)
                data['labels'].attrs.create(name='Temperature_Rise', data=12)
                data['labels'].attrs.create(name='Stick_Event', data=13)
                data['labels'].attrs.create(name='Square_Waves', data=14)
                data['labels'].attrs.create(name='Human_Disturbance', data=15)
                data['labels'].attrs.create(name='Large_Sawtooth', data=16)
                data['labels'].attrs.create(name='Cosinus_Tail', data=17)
                data['labels'].attrs.create(name='Light_only_Event', data=18)
                data['labels'].attrs.create(name='Ring_Light_Event', data=19)
                data['labels'].attrs.create(
                    name='Sharp_Light_Event', data=20)
                data['labels'].attrs.create(name='unknown/other', data=99)

                # store sev

                sev = f_read['stdevent' + name_appendix]['event']
                mp = f_read['stdevent' + name_appendix]['mainpar']
                fitpar = f_read['stdevent' + name_appendix]['fitpar']

                data = f.create_group('stdevent' + name_appendix)
                data.create_dataset(name='event', data=sev)
                data.create_dataset(name='mainpar', data=mp)
                data.create_dataset(name='fitpar', data=fitpar)

            if size_tp > 0:
                print('Simulating Testpulses.')
                data = f.create_group('testpulses')
                events, phs, t0s, nmbr_thrown_testpulses, hours, time_s, time_mus = simulate_events(
                    path_h5=self.path_h5,
                    type='testpulses',
                    name_appendix='',
                    size=size_tp,
                    record_length=self.record_length,
                    nmbr_channels=self.nmbr_channels,
                    ph_intervals=tp_ph_intervals,
                    discrete_ph=tp_discrete_phs,
                    t0_interval=[-20, 20],  # in ms
                    fake_noise=fake_noise,
                    use_bl_from_idx=start_from_bl_idx + size_events + nmbr_thrown_events,
                    take_idx=take_idx,
                    rms_thresholds=rms_thresholds,
                    lamb=lamb,
                    sample_length=sample_length,
                    saturation=saturation,
                    reuse_bl=reuse_bl,
                    ps_dev=ps_dev)
                data.create_dataset(name='event', data=events, dtype=dtype)
                data.create_dataset(name='true_ph', data=phs)
                if not fake_noise:
                    data.create_dataset(name='hours', data=hours)
                    if 'time_s' in f_read['noise']:
                        data.create_dataset(name='time_s', data=time_s)
                        data.create_dataset(name='time_mus', data=time_mus)
                if saturation:
                    fp = f_read['saturation']['fitpar'][0]
                    data_to_write = phs[0] / scale_factor(*fp)
                    if indiv_tpas:
                        data_to_write = np.tile(data_to_write, (self.nmbr_channels, 1))
                    data.create_dataset(name='testpulseamplitude', data=data_to_write)
                data.create_dataset(name='true_onset', data=t0s)
                data.create_dataset(name='labels',
                                    data=2 * np.ones([self.nmbr_channels, size_tp]))  # 2 is the label for testpulses
                data['labels'].attrs.create(name='unlabeled', data=0)
                data['labels'].attrs.create(name='Event_Pulse', data=1)
                data['labels'].attrs.create(name='Test/Control_Pulse', data=2)
                data['labels'].attrs.create(name='Noise', data=3)
                data['labels'].attrs.create(name='Squid_Jump', data=4)
                data['labels'].attrs.create(name='Spike', data=5)
                data['labels'].attrs.create(name='Early_or_late_Trigger', data=6)
                data['labels'].attrs.create(name='Pile_Up', data=7)
                data['labels'].attrs.create(name='Carrier_Event', data=8)
                data['labels'].attrs.create(name='Strongly_Saturated_Event_Pulse', data=9)
                data['labels'].attrs.create(name='Strongly_Saturated_Test/Control_Pulse', data=10)
                data['labels'].attrs.create(name='Decaying_Baseline', data=11)
                data['labels'].attrs.create(name='Temperature_Rise', data=12)
                data['labels'].attrs.create(name='Stick_Event', data=13)
                data['labels'].attrs.create(name='Square_Waves', data=14)
                data['labels'].attrs.create(name='Human_Disturbance', data=15)
                data['labels'].attrs.create(name='Large_Sawtooth', data=16)
                data['labels'].attrs.create(name='Cosinus_Tail', data=17)
                data['labels'].attrs.create(name='Light_only_Event', data=18)
                data['labels'].attrs.create(name='Ring_Light_Event', data=19)
                data['labels'].attrs.create(
                    name='Sharp_Light_Event', data=20)
                data['labels'].attrs.create(name='unknown/other', data=99)

                # store sev

                sev = f_read['stdevent_tp']['event']
                mp = f_read['stdevent_tp']['mainpar']
                fitpar = f_read['stdevent_tp']['fitpar']

                data = f.create_group('stdevent_tp')
                data.create_dataset(name='event', data=sev)
                data.create_dataset(name='mainpar', data=mp)
                data.create_dataset(name='fitpar', data=fitpar)

            data = f.create_group('noise')
            # store nps new and old

            nps = f_read['noise']['nps']
            nps_sim = []
            for c in range(self.nmbr_channels):
                nps_sim.append(calculate_mean_nps(
                    events[c, :, :])[0])

            data.create_dataset(name='nps', data=nps)
            data.create_dataset(name='nps_sim', data=np.array([n for n in nps_sim]))

            if size_noise > 0:
                print('Simulating Noise.')

                events, phs, t0s, nmbr_thrown_noise, hours, time_s, time_mus = simulate_events(path_h5=self.path_h5,
                                                                                               type='noise',
                                                                                               name_appendix='',
                                                                                               size=size_noise,
                                                                                               record_length=self.record_length,
                                                                                               nmbr_channels=self.nmbr_channels,
                                                                                               fake_noise=fake_noise,
                                                                                               use_bl_from_idx=start_from_bl_idx + size_events + size_tp + nmbr_thrown_events + nmbr_thrown_testpulses,
                                                                                               take_idx=take_idx,
                                                                                               rms_thresholds=rms_thresholds,
                                                                                               lamb=lamb,
                                                                                               sample_length=sample_length,
                                                                                               saturation=saturation,
                                                                                               reuse_bl=reuse_bl,
                                                                                               ps_dev=ps_dev
                                                                                               )
                if not fake_noise:
                    data.create_dataset(name='hours', data=hours)
                    if 'time_s' in f_read['noise']:
                        data.create_dataset(name='time_s', data=time_s)
                        data.create_dataset(name='time_mus', data=time_mus)
                data.create_dataset(name='event', data=events, dtype=dtype)
                data.create_dataset(name='labels',
                                    data=3 * np.ones([self.nmbr_channels, size_noise]))  # 3 is the noise label
                data['labels'].attrs.create(name='unlabeled', data=0)
                data['labels'].attrs.create(name='Event_Pulse', data=1)
                data['labels'].attrs.create(name='Test/Control_Pulse', data=2)
                data['labels'].attrs.create(name='Noise', data=3)
                data['labels'].attrs.create(name='Squid_Jump', data=4)
                data['labels'].attrs.create(name='Spike', data=5)
                data['labels'].attrs.create(name='Early_or_late_Trigger', data=6)
                data['labels'].attrs.create(name='Pile_Up', data=7)
                data['labels'].attrs.create(name='Carrier_Event', data=8)
                data['labels'].attrs.create(name='Strongly_Saturated_Event_Pulse', data=9)
                data['labels'].attrs.create(name='Strongly_Saturated_Test/Control_Pulse', data=10)
                data['labels'].attrs.create(name='Decaying_Baseline', data=11)
                data['labels'].attrs.create(name='Temperature_Rise', data=12)
                data['labels'].attrs.create(name='Stick_Event', data=13)
                data['labels'].attrs.create(name='Square_Waves', data=14)
                data['labels'].attrs.create(name='Human_Disturbance', data=15)
                data['labels'].attrs.create(name='Large_Sawtooth', data=16)
                data['labels'].attrs.create(name='Cosinus_Tail', data=17)
                data['labels'].attrs.create(name='Light_only_Event', data=18)
                data['labels'].attrs.create(name='Ring_Light_Event', data=19)
                data['labels'].attrs.create(
                    name='Sharp_Light_Event', data=20)
                data['labels'].attrs.create(name='unknown/other', data=99)

            if store_of is True:
                print('Store OF.')
                of_real = f_read['optimumfilter' + name_appendix]['optimumfilter_real']
                of_imag = f_read['optimumfilter' + name_appendix]['optimumfilter_imag']
                data = f.create_group('optimumfilter' + name_appendix)
                data.create_dataset(name='optimumfilter_real', data=of_real)
                data.create_dataset(name='optimumfilter_imag', data=of_imag)

            print('Simulation done.')
            print('Simulation done.')
            print('Simulation done.')