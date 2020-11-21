# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
from ..simulate._sim_pulses import simulate_events
from ..features._mp import calc_main_parameters
from ..data._baselines import calculate_mean_nps
from ..fit._saturation import scale_factor

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
                        ev_ph_intervals=[[0, 1], [0, 1]],
                        ev_discrete_phs=None,
                        exceptional_sev_naming=None,
                        channels_exceptional_sev=[0],
                        tp_ph_intervals=[[0, 1], [0, 1]],
                        tp_discrete_phs=None,
                        t0_interval=[-20, 20],  # in ms
                        fake_noise=True,
                        store_of=False,
                        rms_thresholds=[1, 0.5],
                        lamb=0.01,
                        sample_length=0.04,
                        assign_labels=[1],
                        saturation=False):
        """
        Simulates a data set of pulses by superposing the fitted SEV with fake or real noise

        :param path_sim: string, the full path where to store the simulated data set
        :param size_events: int, the number of events to simulate; if >0 we need a sev in the hdf5
        :param size_tp: int, the number of testpulses to simulate; if >0 we need a tp-sev in the hdf5
        :param size_noise: int, the number of noise baselines to simulate
        :param ev_ph_intervals: list of NMBR_CHANNELS 2-tuples or lists, the interval in which the pulse heights
            are continuously distributed
        :param ev_discrete_phs: list of NMBR_CHANNELS lists - the discrete values, from which the pulse heights
            are uniformly sampled. if the ph_intervals argument is set, this option will be ignored
        :param exceptional_sev_naming: string or None, if set, this is the full group name in the HDF5 set for the
            sev used for the simulation of events - by setting this, e.g. carrier events can be
            simulated
        :param channel_exceptional_sev: list of ints, the channels for that the exceptional sev is
            used, e.g. if only for phonon channel, choose [0], if for botch phonon and light, choose [0,1]
        :param tp_ph_intervals: analogous to ev_ph_intervals, but for the testpulses
        :param tp_discrete_phs: analogous to ev_ph_intervals, but for the testpulses
        :param t0_interval: 2-tuple or list, the interval from which the pulse onset are continuously sampled
        :param fake_noise: bool, the True the noise will be taken not from the measured baselines from the
            hdf5 set, but simulated
        :param store_of: bool, if True the optimum filter will be saved to the simulated datasets
        :param rms_thresholds: list of two floats, above which value noise baselines are excluded for the
            distribution of polynomial coefficients (i.e. a parameter for the fake noise simulation), also a
            cut parameter for the noise baselines from the h5 set if no fake ones are taken
        :param lamb: float, a parameter for the fake baseline simulation, increase if calculation time is too long
        :param sample_length: float, the length of one sample in milliseconds
        :param assign_labels: list of ints, pre-assign a label to all the simulated events; tp and noise are
            automatically labeled, the length of the list must match the list channels_exceptional_sev
        :param saturation: bool, if true apply the logistics curve to the simulated pulses
        :return: -
        """

        # create file handle
        f = h5py.File(path_sim, 'w')
        f_read = h5py.File(self.path_h5, 'r')

        nmbr_thrown_events = 0
        nmbr_thrown_testpulses = 0

        if size_events > 0:
            print('Simulating Events.')
            data = f.create_group('events')
            events, phs, t0s, nmbr_thrown_events = simulate_events(path_h5=self.path_h5,
                                                                   type='events',
                                                                   size=size_events,
                                                                   record_length=self.record_length,
                                                                   nmbr_channels=self.nmbr_channels,
                                                                   ph_intervals=ev_ph_intervals,
                                                                   discrete_ph=ev_discrete_phs,
                                                                   exceptional_sev_naming=exceptional_sev_naming,
                                                                   channels_exceptional_sev=channels_exceptional_sev,
                                                                   t0_interval=t0_interval,  # in ms
                                                                   fake_noise=fake_noise,
                                                                   use_bl_from_idx=0,
                                                                   rms_thresholds=rms_thresholds,
                                                                   lamb=lamb,
                                                                   sample_length=sample_length,
                                                                   saturation=saturation)
            data.create_dataset(name='event', data=events)
            data.create_dataset(name='true_ph', data=phs)
            data.create_dataset(name='true_onset', data=t0s)

            labels = np.ones([self.nmbr_channels, size_events])
            for c in channels_exceptional_sev:
                labels[c] *= assign_labels[c]

            data.create_dataset(name='labels', data=labels)

            # store sev

            sev = f_read['stdevent']['event']
            mp = f_read['stdevent']['mainpar']
            fitpar = f_read['stdevent']['fitpar']

            data = f.create_group('stdevent')
            data.create_dataset(name='event', data=sev)
            data.create_dataset(name='mainpar', data=mp)
            data.create_dataset(name='fitpar', data=fitpar)

        if size_tp > 0:
            print('Simulating Testpulses.')
            data = f.create_group('testpulses')
            events, phs, t0s, nmbr_thrown_testpulses = simulate_events(path_h5=self.path_h5,
                                                                       type='testpulses',
                                                                       size=size_tp,
                                                                       record_length=self.record_length,
                                                                       nmbr_channels=self.nmbr_channels,
                                                                       ph_intervals=tp_ph_intervals,
                                                                       discrete_ph=tp_discrete_phs,
                                                                       t0_interval=[-20, 20],  # in ms
                                                                       fake_noise=fake_noise,
                                                                       use_bl_from_idx=size_events + nmbr_thrown_events,
                                                                       rms_thresholds=rms_thresholds,
                                                                       lamb=lamb,
                                                                       sample_length=sample_length,
                                                                       saturation=saturation)
            data.create_dataset(name='event', data=events)
            data.create_dataset(name='true_ph', data=phs)
            if saturation:
                fp = f_read['saturation']['fitpar'][0]
                data.create_dataset(name='testpulseamplitude', data=phs[0]/scale_factor(*fp))
            data.create_dataset(name='true_onset', data=t0s)
            data.create_dataset(name='labels',
                                data=2 * np.ones([self.nmbr_channels, size_tp]))  # 2 is the label for testpulses

            # store sev

            sev = f_read['stdevent_tp']['event']
            mp = np.array([calc_main_parameters(x).getArray() for x in sev])

            data = f.create_group('stdevent_tp')
            data.create_dataset(name='event', data=sev)
            data.create_dataset(name='mainpar', data=mp)

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

            events, phs, t0s, nmbr_thrown_noise = simulate_events(path_h5=self.path_h5,
                                                                  type='noise',
                                                                  size=size_noise,
                                                                  record_length=self.record_length,
                                                                  nmbr_channels=self.nmbr_channels,
                                                                  fake_noise=fake_noise,
                                                                  use_bl_from_idx=size_events + size_tp + nmbr_thrown_events + nmbr_thrown_testpulses,
                                                                  rms_thresholds=rms_thresholds,
                                                                  lamb=lamb,
                                                                  sample_length=sample_length,
                                                                  saturation=saturation
                                                                  )
            data.create_dataset(name='event', data=events)
            data.create_dataset(name='labels',
                                data=3 * np.ones([self.nmbr_channels, size_noise]))  # 3 is the noise label

        if store_of == 0:
            print('Store OF.')
            of_real = f_read['optimumfilter']['optimumfilter_real']
            of_imag = f_read['optimumfilter']['optimumfilter_imag']
            data = f.create_group('optimumfilter')
            data.create_dataset(name='optimumfilter_real', data=of_real)
            data.create_dataset(name='optimumfilter_imag', data=of_imag)

        print('Simulation done.')
        f.close()
