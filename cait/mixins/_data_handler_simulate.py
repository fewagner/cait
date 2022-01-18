# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
from ..simulate._sim_pulses import simulate_events
from ..features._mp import calc_main_parameters
from ..data._baselines import calculate_mean_nps
from ..fit._saturation import scale_factor
import warnings


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------


class SimulateMixin(object):
    """
    A Mixin Class for the DataHandler class with methods to simulate data sets.
    """

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
