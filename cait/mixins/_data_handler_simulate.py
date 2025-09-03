import warnings
from functools import partial
from typing import List, Union

import h5py
import numpy as np

import cait.versatile as vai
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
        stream: vai.datasources.stream.streambase.StreamBaseClass,
        trigger_channels: Union[str, List[str]],
        of: np.ndarray,
        threshold: Union[float, List[float]],
        sev: np.ndarray = None,
        sev_fitpars: List[List[float]] = None,
        shift_samples: List[List[int]] = None,
        passive_channels: Union[str, List[str]] = None,
        testpulse_channels: Union[str, List[str]] = None,
        tolerance_samples: int = 10,
        n_record_lens: int = 8,
        record_placement: int = 4,
        tag: str = "",
        preview: bool = False,
    ):
        """
        Perform a trigger efficiency simulation by superimposing a SEV onto random parts of a stream and running an optimum filter trigger to check whether they survive or not.
        """
        # Ensures that all inputs are lists
        trigger_channels, passive_channels, testpulse_channels, _, thresholds = _sanitize_input(
            stream=stream,
            trigger_channels=trigger_channels,
            passive_channels=passive_channels,
            testpulse_channels=testpulse_channels,
            controlpulses_above=None,
            thresholds=threshold,
        )

        if np.sum([x is None for x in [sev, sev_fitpars]]) != 1:
            raise ValueError(f"You have to specify EITHER a sev or its fit parameters. At least one. Not both.")

        using_fit = sev_fitpars is not None
        sev_or_pars = sev_fitpars if using_fit else sev

        # Want 1d timestamps but 2d pulse heights, sev, and of
        sim_ts = np.array(sim_ts).flatten()
        sim_phs, sev_or_pars, of = np.atleast_2d(sim_phs), np.atleast_2d(sev_or_pars), np.atleast_2d(of)

        if passive_channels is not None:
            all_channels = trigger_channels + passive_channels
        else:
            all_channels = trigger_channels
        n_total_ch = len(all_channels)
        n_trig_ch = len(trigger_channels)

        if sim_phs.shape[0] != n_total_ch:
            raise ValueError(f"Pulse heights are needed for all channels (including passive ones). Got pulse heights of shape {sim_phs.shape} and in total {n_total_ch} channel(s).")
        if sev_or_pars.shape[0] != n_total_ch:
            raise ValueError(f"SEVs are needed for all channels (including passive ones). Got sev/sev_fitpars of shape {sev_or_pars.shape} and in total {n_total_ch} channel(s).")
        if of.shape[0] != len(trigger_channels):
            raise ValueError(f"OFs are needed for all trigger channels. Got OF of shape {of.shape} and {len(trigger_channels)} trigger channel(s).")
        if sim_phs.shape[-1] != len(sim_ts):
            raise ValueError(f"The number of timestamps must agree with the number of pulse heights. Got {len(sim_ts)} and {sim_phs.shape[-1]}.")
        
        if shift_samples is not None:
            shift_samples = np.atleast_2d(shift_samples).astype(np.int32)
            if shift_samples.shape != sim_phs.shape:
                raise ValueError(f"The shapes of 'shift_samples' and 'sim_phs' have to be identical. Got {shift_samples.shape} and {sim_phs.shape}.")

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
        
        # PREPARING SEV TO BE USED IN PULSE_SIM_ITERATOR
        if using_fit:
            # We place the pulse in the (extended) record window as
            # defined by 'record_placement'.
            # The time=0 of the fitpars is adjusted to fall onto
            # pulse_sim_index
            # The (extended) iterator's time array is used to 
            # evaluate the fitpars. Its 0 is aligned at 1/4th as
            # usual. We compensate for that.
            shifted_pars = np.atleast_2d(sev_or_pars[:n_trig_ch,:].copy())
            shifted_pars[:,0] += (record_placement - n_record_lens/4)*rl*stream.dt_us/1000
            of_trigger = [of[i] for i in range(n_trig_ch)]

            pulse_sim_index = np.argmax(
                pulse_template(
                    (np.arange(n_record_lens*rl) - n_record_lens*rl/4)*stream.dt_us/1000, 
                    *shifted_pars[0]
                    )
                )
            iterator_kwargs = dict(sev_fitpars=shifted_pars)
        else:
            # The SEV that is placed on the stream chunks for triggering
            # is first sanitized with a window function (this prevents
            # high frequency artifacts). Reducing the pulse's area means
            # that we also have to re-normalize the OF!
            # (This is only done for the channels which are actually triggered)
            window_sev = [vai.TukeyWindow()(sev_or_pars[i]) for i in range(n_trig_ch)]
            scale = [np.max(vai.OptimumFiltering(of[i])(window_sev[i])) for i in range(n_trig_ch)]
            of_trigger = [of[i]/scale[i] for i in range(n_trig_ch)]
            
            # We now place it on the stream chunk as defined by the record_placement
            padded_sev = np.zeros((n_trig_ch, n_record_lens*rl))
            for i, s in enumerate(window_sev):
                padded_sev[i, record_placement*rl:(record_placement+1)*rl] = np.array(s)
            
            pulse_sim_index = np.argmax(padded_sev[0])
            iterator_kwargs = dict(sev=padded_sev)

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
        )
        
        # DEFINE TARGET INDEX FOR TRIGGER SURVIVAL FUNCTION
        if using_fit:
            # The maximum of the evaluated pulse shape is used as the
            # target index for the trigger survival. This index is the
            # same irrespective of the added shifts.
            target_inds = [
                np.argmax(pulse_template(chunk_iterator.t, *p)) 
                for p in shifted_pars
            ]
        else:
            target_inds = [np.argmax(ps) for ps in padded_sev]
        
        # INITIALIZE TRIGGER SURVIVAL FUNCTION
        fns = [
            vai.TriggerSurvival(
                trigger_fnc=partial(vai.trigger_of, of=ot, threshold=th),
                target_ind=tind,
                tolerance_samples=tolerance_samples
            )
            for tind, ot, th in zip(target_inds, of_trigger, thresholds)
        ]
        
        if preview:
            for i, f in enumerate(fns):
                vai.Preview(chunk_iterator[i], f)
            return
        
        # Initialize array with as many channels as total channels (including passive).
        # Triggers will be false for passive, values and inds will be -1.
        # In the next step, the rows of those arrays which correspond to trigger_channels
        # will be filled accordingly.
        separate_trigger = np.zeros((n_total_ch, len(chunk_iterator)), dtype=bool)
        trigger_val = -1*np.ones((n_total_ch, len(chunk_iterator)), dtype=np.float32)
        trigger_ind = -1*np.ones((n_total_ch, len(chunk_iterator)), dtype=np.int32)

        for i, f in enumerate(fns):
            separate_trigger[i, :], trigger_val[i, :], trigger_ind[i, :] = vai.apply(
                f, 
                chunk_iterator[i], 
                pb_prefix=f"Triggering channel {i}",
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
                np.sort(tp_ts), 
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
        
        #mod_shift_samples += (pulse_sim_index-int(rl//4))

        event_iterator = PulseSimIterator(
            stream.get_event_iterator(
                all_channels, 
                record_length=rl, 
                timestamps=event_ts[..., event_flag],
            ),
            # PulseSimIterator handles sev/sev_fitpars
            sev=sev,
            sev_fitpars=sev_fitpars,
            pulse_heights=sim_phs[..., event_flag],
            shift_samples=mod_shift_samples[..., event_flag],
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

