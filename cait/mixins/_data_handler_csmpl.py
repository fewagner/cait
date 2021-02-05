# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import h5py
from ..trigger._csmpl import trigger_csmpl, get_record_window, align_triggers, sample_to_time, \
    exclude_testpulses, get_starttime, get_test_stamps


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class CsmplMixin(object):
    """
    A Mixin Class to the DataHandler Class with methods for the csmpl files.
    """

    def include_csmpl_triggers(self,
                               csmpl_paths,  # list of all paths for the channels
                               thresholds,
                               trigger_block=None,
                               take_samples=-1,
                               start_time=0,
                               of=None,
                               path_sql=None,
                               csmpl_channels=None,
                               filenames=None
                               ):
        # TODO

        if trigger_block is None:
            trigger_block = self.record_length

        if path_sql is not None and (csmpl_channels is None or filenames is None):
            raise KeyError(
                'If you want to read start time from SQL database, you must provide the csmpl_channels and filenames!')

        # correct paths

        csmpl_paths = ["{}/run{}_{}/{}".format(self.path_directory, self.run, self.module, c) for c in csmpl_paths]

        # open write file
        h5f = h5py.File(self.path_h5, 'r+')
        if of is None:
            print("Read OF Transfer Function from h5 file. Alternatively provide one through the of argument.")
            of = np.zeros((self.nmbr_channels, int(self.record_length / 2 + 1)), dtype=np.complex)
            of.real = h5f['optimumfilter']['optimumfilter_real']
            of.imag = h5f['optimumfilter']['optimumfilter_imag']
        stream = h5f.require_group(name='stream')

        # do the triggering
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
                                 start_hours=start_time,
                                 trigger_block=trigger_block,
                                 )

            time.append(trig)

        # fix the number of triggers
        if len(csmpl_paths) > 1:
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

        if path_sql is not None:
            # get file handle
            if "trigger_time_stamp_high" in stream:
                print('overwrite old trigger_time_stamp_high')
                del stream['trigger_time_stamp_high']
            stream.create_dataset(name='trigger_time_stamp_high',
                                  shape=(aligned_triggers.shape),
                                  dtype=float)
            if "trigger_time_stamp_low" in stream:
                print('overwrite old trigger_time_stamp_low')
                del stream['trigger_time_stamp_low']
            stream.create_dataset(name='trigger_time_stamp_low',
                                  shape=(aligned_triggers.shape),
                                  dtype=float)

            # get start second from sql
            file_start = get_starttime(path_sql=path_sql,
                                       csmpl_channel=csmpl_channels[0],
                                       filename=filenames[0])

            stream['trigger_time_stamp_high'][...] = np.floor(aligned_triggers[c] + file_start)
            stream['trigger_time_stamp_low'][...] = 1e6 * (
                    aligned_triggers[c] + file_start - np.floor(aligned_triggers[c] + file_start))

        print('DONE')

    def include_nps(self, nps):
        # TODO
        print('Write NPS')
        f = h5py.File(self.path_h5, 'r+')
        noise = f.require_group(name='noise')
        noise.create_dataset(name='nps',
                             data=nps)

    def include_sev(self, sev, fitpar, mainpar):
        # TODO
        print('Write SEV')
        f = h5py.File(self.path_h5, 'r+')
        stdevent = f.require_group(name='stdevent')
        stdevent.create_dataset(name='event',
                                data=sev)
        stdevent.create_dataset(name='fitpar',
                                data=fitpar)
        stdevent.create_dataset(name='mainpar',
                                data=mainpar)

    def include_of(self, of_real, of_imag):
        # TODO
        print('Write OF')
        f = h5py.File(self.path_h5, 'r+')
        optimumfilter = f.require_group(name='optimumfilter')
        optimumfilter.create_dataset(name='optimumfilter_real',
                                     data=of_real)
        optimumfilter.create_dataset(name='optimumfilter_imag',
                                     data=of_imag)

    def include_triggered_events_h5(self,
                                    csmpl_paths,
                                    max_time_diff=0.03,
                                    exclude_tp=True,
                                    sample_duration=0.00004,
                                    datatype='float32',
                                    min_tpa=0.001,
                                    min_cpa=10.1):

        # TODO
        csmpl_paths = ["{}/run{}_{}/{}".format(self.path_directory, self.run, self.module, c) for c in csmpl_paths]

        # write to new file: triggered events, time trig events, timestamps tpa and mainpar tp, nps, of, stdeven inkl fit and main par,
        # open read file
        h5f = h5py.File(self.path_h5, 'r+')
        if "events" in h5f:
            while val != 'y':
                val = input("An events group exists in this file. Overwrite? y/n")
                if val == 'n':
                    raise FileExistsError(
                        "Events group exists already in the H5 file. Create new file to not overwrite.")

        stream = h5f['stream']
        trigger_hours = np.array(stream['trigger_hours'])
        if "trigger_time_stamp_high" in stream:
            trigger_s = np.array(stream['trigger_time_stamp_high'])
            trigger_mus = np.array(stream['trigger_time_stamp_low'])
        if exclude_tp:
            tp_hours = np.array(h5f['stream']['tp_hours'])
            tpas = np.array(h5f['stream']['tpa'])
            if "tp_time_stamp_high" in stream:
                tp_s = np.array(stream['tp_time_stamp_high'])
                tp_mus = np.array(stream['tp_time_stamp_low'])

        # open write file
        write_events = h5f.create_group('events')
        if exclude_tp:
            write_testpulses = h5f.create_group('testpulses')
            write_controlpulses = h5f.create_group('controlpulses')
            write_testpulses.create_dataset(name="hours",
                                            data=tp_hours[np.logical_and(tpas > min_tpa, tpas < min_cpa)])
            write_testpulses.create_dataset(name="testpulseamplitude",
                                            data=tpas[np.logical_and(tpas > min_tpa, tpas < min_cpa)])
            write_controlpulses.create_dataset(name="hours",
                                               data=tp_hours[tpas > min_cpa])
            if "tp_time_stamp_high" in stream:
                write_testpulses.create_dataset(name="time_s",
                                                data=tp_s[np.logical_and(tpas > min_tpa, tpas < min_cpa)])
                write_testpulses.create_dataset(name="time_mus",
                                                data=tp_mus[np.logical_and(tpas > min_tpa, tpas < min_cpa)])
                write_controlpulses.create_dataset(name="time_s",
                                                   data=tp_s[tpas > min_cpa])
                write_controlpulses.create_dataset(name="time_mus",
                                                   data=tp_mus[tpas > min_cpa])

            print('Exclude Testpulses.')
            flag = exclude_testpulses(trigger_hours=trigger_hours,
                                      tp_hours=tp_hours[tpas > min_tpa],
                                      max_time_diff=max_time_diff)
            trigger_hours = trigger_hours[flag]
            if "trigger_time_stamp_high" in stream:
                trigger_s = trigger_s[flag]
                trigger_mus = trigger_mus[flag]

        write_events.create_dataset(name='hours',
                                    data=trigger_hours)
        if "trigger_time_stamp_high" in stream:
            write_events.create_dataset(name='time_s',
                                        data=trigger_s)
            write_events.create_dataset(name='time_mus',
                                        data=trigger_mus)

        write_events.create_dataset(name='event',
                                    shape=(self.nmbr_channels, len(trigger_hours), self.record_length),
                                    dtype=datatype)

        print('Include the triggered events.')
        for c in range(self.nmbr_channels):

            print('Channel ', c)
            for i in range(len(trigger_hours)):
                if i % 1000 == 0:
                    print('Get Event Nmbr ', i)
                write_events['event'][c, i, :], _ = get_record_window(path=csmpl_paths[c],
                                                                      start_time=trigger_hours[
                                                                                     i] * 3600 - sample_to_time(
                                                                          self.record_length / 4,
                                                                          sample_duration=sample_duration),
                                                                      record_length=self.record_length,
                                                                      sample_duration=sample_duration)

        if exclude_tp:

            # ------------
            print('Include Testpulse Events.')

            tp_ev = write_testpulses.create_dataset(name='events',
                                                    shape=(self.nmbr_channels, len(
                                                        tp_hours[np.logical_and(tpas > min_tpa, tpas < min_cpa)])),
                                                    dtype=datatype)

            for c in range(self.nmbr_channels):

                print('Channel ', c)
                for i in range(len(trigger_hours)):
                    if i % 1000 == 0:
                        print('Calc Pulse Nmbr ', i)
                    tp_ev[c, ...], _ = get_record_window(path=csmpl_paths[c],
                                                         start_time=tp_hours[np.logical_and(tpas > min_tpa,
                                                                                            tpas < min_cpa)] * 3600 - sample_to_time(
                                                             self.record_length / 4,
                                                             sample_duration=sample_duration),
                                                         record_length=self.record_length,
                                                         sample_duration=sample_duration)

            # ------------
            print('Calculate Control Pulse Heights.')
            cphs = write_controlpulses.create_dataset(name="events",
                                                      shape=(self.nmbr_channels, len(tp_hours[tpas > min_cpa])),
                                                      dtype=float)

            for c in range(self.nmbr_channels):

                print('Channel ', c)
                for i in range(len(trigger_hours)):
                    if i % 1000 == 0:
                        print('Calc Pulse Nmbr ', i)
                    cp_array, _ = get_record_window(path=csmpl_paths[c],
                                                    start_time=tp_hours[tpas > min_cpa] * 3600 - sample_to_time(
                                                        self.record_length / 4,
                                                        sample_duration=sample_duration),
                                                    record_length=self.record_length,
                                                    sample_duration=sample_duration)

                    # subtract offset
                    cp_array -= np.mean(cp_array[:int(len(cp_array) / 8)])

                    # write the heights to file
                    cphs[c, ...] = np.max(cp_array)

        print('DONE')

    def include_test_stamps(self, path_teststamps, path_dig_stamps, path_sql, csmpl_channels, csmpl_filenames,
                            triggerdelay=2081024,
                            samplediv=2,
                            sample_length=0.00004):

        # open file stream
        h5f = h5py.File(self.path_h5, 'r+')
        h5f.require_group('stream')

        # get start second from sql
        file_start = get_starttime(path_sql=path_sql,
                                   csmpl_channel=csmpl_channels[0],
                                   filename=csmpl_filenames[0])

        # determine the offset of the trigger time stamps from the digitizer stamps

        dig = np.dtype([
            ('stamp', np.uint64),
            ('bank', np.uint32),
            ('bank2', np.uint32),
        ])

        diq_stamps = np.fromfile(path_dig_stamps, dtype=dig)
        dig_s = diq_stamps['stamp'] / 400  # the digitizer stamps are oversampled
        one_bank = (triggerdelay / samplediv)  # the number of samples after which one dig stamp is written
        offset_hours = np.abs(
            one_bank - dig_s[0]) * sample_length / 3600  # the dig stamp is delayed by offset_hours

        # read the test pulse time stamps from the test_stamps file

        hours, tpas, _ = get_test_stamps(path=path_teststamps, channels=[0])

        # remove the offset

        hours -= offset_hours

        # calc the time stamp low and high

        time_high = np.floor(file_start + hours*3600)
        time_low = 1e6*(hours*3600 - np.floor(hours*3600))

        # write to file

        h5f['stream'].require_dataset(name='tp_hours',
                                      shape=hours.shape,
                                      dtype=float)
        h5f['stream'].require_dataset(name='tpa',
                                      shape=tpas.shape,
                                      dtype=float)
        h5f['stream'].require_dataset(name='tp_time_stamp_high',
                                      shape=hours.shape,
                                      dtype=int)
        h5f['stream'].require_dataset(name='tp_time_stamp_low',
                                      shape=hours.shape,
                                      dtype=int)

        h5f['stream']['tp_hours'][...] = hours - offset_hours
        h5f['stream']['tpa'][...] = tpas
        h5f['stream']['tp_time_stamp_high'][...] = time_high
        h5f['stream']['tp_time_stamp_low'][...] = time_low

        print('Test Stamps included.')

