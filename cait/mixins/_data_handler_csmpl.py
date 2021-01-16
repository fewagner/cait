# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import h5py
from ..trigger._csmpl import trigger_csmpl, get_record_window, align_triggers, sample_to_time, find_nearest, \
    exclude_testpulses


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class CsmplMixin(object):
    """
    A Mixin Class to the DataHandler Class with methods for the csmpl files.
    """

    def add_triggers(self,
                     csmpl_paths,
                     thresholds,
                     max_channel_diff=0.01,
                     trigger_block=16384,
                     take_samples=-1,
                     start_time=0
                     ):
        # TODO

        # correct paths

        csmpl_paths = ["{}/run{}_{}/{}".format(self.path_directory, self.run, self.module, c) for c in csmpl_paths]

        # open write file
        h5f = h5py.File(self.path_h5, 'r+')
        of = np.zeros((2, int(self.record_length / 2 + 1)), dtype=np.complex)
        of.real = h5f['optimumfilter']['optimumfilter_real']
        of.imag = h5f['optimumfilter']['optimumfilter_imag']
        stream = h5f.require_group(name='stream')

        time = []

        for c in range(self.nmbr_channels):
            print('TRIGGER CHANNEL ', c)
            # get triggers
            trig = trigger_csmpl(paths=[csmpl_paths[c]],
                                 trigger_tres=thresholds[c],
                                 transfer_function=of[c],
                                 take_samples=take_samples,
                                 record_length=self.record_length,
                                 sample_length=1 / self.sample_frequency,
                                 start_time=start_time,
                                 trigger_block=trigger_block,
                                 )

            time.append(trig)

        # fix the number of triggers
        print('ALIGN TRIGGERS')
        aligned_triggers = align_triggers(triggers=time,  # in seconds
                                          trigger_block=trigger_block,
                                          sample_duration=1 / self.sample_frequency,
                                          )

        # write them to file
        print('ADD DATASETS TO HDF5')
        if "trigger_hours" in stream:
            print('overwrite old trigger_hours')
            del stream['trigger_hours']
        stream.create_dataset(name='trigger_hours',
                              data=aligned_triggers / 3600)

        print('DONE')

    def create_triggered_events_h5(self,
                                   csmpl_paths,
                                   new_h5_name,
                                   max_time_diff=0.03,
                                   exclude_tp=True,
                                   sample_duration=0.00004):
        # TODO
        csmpl_paths = ["{}/run{}_{}/{}".format(self.path_directory, self.run, self.module, c) for c in csmpl_paths]

        # write to new file: triggered events, time trig events, timestamps tpa and mainpar tp, nps, of, stdeven inkl fit and main par,
        # open read file
        read_h5f = h5py.File(self.path_h5, 'r')
        stream = read_h5f['stream']
        trigger_hours = np.array(stream['trigger_hours'])
        tp_hours = np.array(read_h5f['testpulses']['hours'])

        # open write file
        write_h5f = h5py.File("{}/run{}_{}/{}.h5".format(self.path_directory, self.run, self.module, new_h5_name), 'w')
        write_events = write_h5f.create_group('events')
        write_noise = write_h5f.create_group('noise')
        write_optimumfilter = write_h5f.create_group('optimumfilter')
        write_stdevent = write_h5f.create_group('stdevent')
        write_testpulses = write_h5f.create_group('testpulses')
        write_stream = write_h5f.create_group('stream')

        if exclude_tp:
            print('Exclude Testpulses')
            flag = exclude_testpulses(trigger_hours=trigger_hours,
                                      tp_hours=tp_hours,
                                      max_time_diff=max_time_diff)
            trigger_hours = trigger_hours[flag]
        write_events.create_dataset(name='hours',
                                    data=trigger_hours)
        write_stream.create_dataset(name='trigger_hours',
                                    data=trigger_hours)

        write_events.create_dataset(name='event',
                                    shape=(self.nmbr_channels, len(trigger_hours), self.record_length),
                                    dtype=float)
        for c in range(self.nmbr_channels):
            print('Channel ', c)
            for i in range(len(trigger_hours)):
                if i % 100 == 0:
                    print('Get Event Nmbr ', i)
                write_events['event'][c, i, :], _ = get_record_window(path=csmpl_paths[c],
                                                                      start_time=trigger_hours[
                                                                                     i] * 3600 - sample_to_time(
                                                                          self.record_length / 4,
                                                                          sample_duration=sample_duration),
                                                                      record_length=self.record_length,
                                                                      sample_duration=sample_duration)
        print('Write NPS')
        write_noise.create_dataset(name='nps',
                                   data=read_h5f['noise']['nps'])
        print('Write OF')
        write_optimumfilter.create_dataset(name='optimumfilter_real',
                                           data=read_h5f['optimumfilter']['optimumfilter_real'])
        write_optimumfilter.create_dataset(name='optimumfilter_imag',
                                           data=read_h5f['optimumfilter']['optimumfilter_imag'])
        print('Write SEV')
        write_stdevent.create_dataset(name='event',
                                      data=read_h5f['stdevent']['event'])
        write_stdevent.create_dataset(name='fitpar',
                                      data=read_h5f['stdevent']['fitpar'])
        write_stdevent.create_dataset(name='mainpar',
                                      data=read_h5f['stdevent']['mainpar'])
        print('Write TP parameters')
        write_testpulses.create_dataset(name='hours',
                                        data=read_h5f['testpulses']['hours'])
        write_testpulses.create_dataset(name='mainpar',
                                        data=read_h5f['testpulses']['mainpar'])
        print('DONE')
