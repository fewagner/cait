# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
from ..simulate._sim_pulses import simulate_events
from ..features._mp import calc_main_parameters
from ..data._baselines import calculate_mean_nps

# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------


class SimulateMixin(object):

    # Simulate Dataset with specific classes
    def simulate_pulses(self,
                        path_sim,
                        size_events=0,
                        size_tp=0,
                        size_noise=0,
                        ev_ph_intervals=[[0, 1], [0, 1]],
                        ev_discrete_phs=None,
                        tp_ph_intervals=[[0, 1], [0, 1]],
                        tp_discrete_phs=None,
                        t0_interval=[-20, 20],  # in ms
                        fake_noise=True,
                        store_of=False,
                        rms_thresholds=[1, 0.5],
                        lamb=0.01,
                        sample_length=0.04):
        """


        :param path_sim:
        :param size_events:
        :param size_tp:
        :param size_noise:
        :param ev_ph_intervals:
        :param ev_discrete_phs:
        :param tp_ph_intervals:
        :param tp_discrete_phs:
        :param t0_interval:
        :param fake_noise:
        :param store_of:
        :param rms_thresholds:
        :param lamb:
        :param sample_length:
        :return:
        """

        # create file handle
        f = h5py.File(path_sim, 'w')
        f_read = h5py.File(self.path_h5, 'r')

        nmbr_thrown_events = 0
        nmbr_thrown_testpulses = 0
        nmbr_thrown_noise = 0

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
                                     t0_interval=t0_interval,  # in ms
                                     fake_noise=fake_noise,
                                     use_bl_from_idx=0,
                                     rms_thresholds=rms_thresholds,
                                     lamb=lamb,
                                     sample_length=sample_length)
            data.create_dataset(name='event', data=events)
            data.create_dataset(name='true_ph', data=phs)
            data = f.create_group('onset')
            data.create_dataset(name='true_onset', data=t0s)

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
                                     sample_length=sample_length)
            data.create_dataset(name='event', data=events)
            data.create_dataset(name='true_ph', data=phs)
            data = f.create_group('onset')
            data.create_dataset(name='true_onset', data=t0s)

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
                                     sample_length=sample_length)
            data.create_dataset(name='event', data=events)

        if store_of == 0:
            print('Store OF.')
            of_real = f_read['optimumfilter']['optimumfilter_real']
            of_imag = f_read['optimumfilter']['optimumfilter_imag']
            data = f.create_group('optimumfilter')
            data.create_dataset(name='optimumfilter_real', data=of_real)
            data.create_dataset(name='optimumfilter_imag', data=of_imag)

        print('Simulation done.')
        f.close()
