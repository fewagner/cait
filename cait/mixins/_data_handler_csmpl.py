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
                               filenames=None,
                               down=1
                               ):
        # TODO

        if trigger_block is None:
            trigger_block = self.record_length

        if path_sql is not None and (csmpl_channels is None or filenames is None):
            raise KeyError(
                'If you want to read start time from SQL database, you must provide the csmpl_channels and filenames!')

        # correct paths

        csmpl_paths = ["{}/{}".format(self.path_directory, c) for c in csmpl_paths]

        # open write file
        with h5py.File(self.path_h5, 'a') as h5f:
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
                                     down=down,
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

                stream['trigger_time_s'][...] = np.floor(aligned_triggers[c] + file_start)
                stream['trigger_time_mus'][...] = 1e6 * (
                        aligned_triggers[c] + file_start - np.floor(aligned_triggers[c] + file_start))

            print('DONE')


    def include_nps(self, nps):
        # TODO
        print('Write NPS')
        with h5py.File(self.path_h5, 'r+') as f:
            noise = f.require_group(name='noise')
            if 'nps' in noise:
                del noise['nps']
            noise.create_dataset(name='nps',
                                 data=nps)


    def include_sev(self, sev, fitpar, mainpar):
        # TODO
        print('Write SEV')
        with h5py.File(self.path_h5, 'r+') as f:
            stdevent = f.require_group(name='stdevent')
            if 'event' in stdevent:
                del stdevent['event']
            if 'fitpar' in stdevent:
                del stdevent['fitpar']
            if 'mainpar' in stdevent:
                del stdevent['mainpar']
            stdevent.create_dataset(name='event',
                                    data=sev)
            stdevent.create_dataset(name='fitpar',
                                    data=fitpar)
            stdevent.create_dataset(name='mainpar',
                                    data=mainpar)

    def include_of(self, of_real, of_imag, down=1):
        # TODO
        print('Write OF')
        with h5py.File(self.path_h5, 'r+') as f:
            optimumfilter = f.require_group(name='optimumfilter')
            if down > 1:
                if 'optimumfilter_real_down{}'.format(down) in optimumfilter:
                    del optimumfilter['optimumfilter_real_down{}'.format(down)]
                if 'optimumfilter_imag_down{}'.format(down) in optimumfilter:
                    del optimumfilter['optimumfilter_imag_down{}'.format(down)]
                optimumfilter.create_dataset(name='optimumfilter_real_down{}'.format(down),
                                             data=of_real)
                optimumfilter.create_dataset(name='optimumfilter_imag_down{}'.format(down),
                                             data=of_imag)
            else:
                if 'optimumfilter_real' in optimumfilter:
                    del optimumfilter['optimumfilter_real']
                if 'optimumfilter_imag' in optimumfilter:
                    del optimumfilter['optimumfilter_imag']
                optimumfilter.create_dataset(name='optimumfilter_real',
                                             data=of_real)
                optimumfilter.create_dataset(name='optimumfilter_imag',
                                             data=of_imag)


    def include_triggered_events(self,
                                 csmpl_paths,
                                 max_time_diff=0.03, # in sec
                                 exclude_tp=True,
                                 sample_duration=0.00004,
                                 datatype='float32',
                                 min_tpa=0.001,
                                 min_cpa=10.1,
                                 down=1):

        # write to new file: triggered events, time trig events, timestamps tpa and mainpar tp, nps, of, stdeven inkl fit and main par,
        # open read file
        with h5py.File(self.path_h5, 'r+') as h5f:
            if "events" in h5f:
                val = None
                while val != 'y':
                    val = input("An events group exists in this file. Overwrite? y/n")
                    if val == 'n':
                        raise FileExistsError(
                            "Events group exists already in the H5 file. Create new file to not overwrite.")

            stream = h5f['stream']
            trigger_hours = np.array(stream['trigger_hours'])
            if "trigger_time_s" in stream:
                trigger_s = np.array(stream['trigger_time_s'])
                trigger_mus = np.array(stream['trigger_time_mus'])
            if exclude_tp:
                tp_hours = np.array(h5f['stream']['tp_hours'])
                tpas = np.array(h5f['stream']['tpa'])
                if "tp_time_s" in stream:
                    tp_s = np.array(stream['tp_time_s'])
                    tp_mus = np.array(stream['tp_time_mus'])

            # open write file
            write_events = h5f.require_group('events')
            if exclude_tp:
                write_testpulses = h5f.require_group('testpulses')
                write_controlpulses = h5f.require_group('controlpulses')
                if "hours" in write_testpulses:
                    del write_testpulses["hours"]
                if "testpulseamplitude" in write_testpulses:
                    del write_testpulses["testpulseamplitude"]
                if "hours" in write_controlpulses:
                    del write_controlpulses["hours"]
                write_testpulses.create_dataset(name="hours",
                                                data=tp_hours[np.logical_and(tpas > min_tpa, tpas < min_cpa)])
                write_testpulses.create_dataset(name="testpulseamplitude",
                                                data=tpas[np.logical_and(tpas > min_tpa, tpas < min_cpa)])
                write_controlpulses.create_dataset(name="hours",
                                                   data=tp_hours[tpas > min_cpa])
                if "tp_time_s" in stream:
                    if "time_s" in write_testpulses:
                        del write_testpulses["time_s"]
                    if "time_mus" in write_testpulses:
                        del write_testpulses["time_mus"]
                    if "time_s" in write_controlpulses:
                        del write_controlpulses["time_s"]
                    if "time_mus" in write_controlpulses:
                        del write_controlpulses["time_mus"]
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
                if "trigger_time_s" in stream:
                    trigger_s = trigger_s[flag]
                    trigger_mus = trigger_mus[flag]

            if 'hours' in write_events:
                del write_events['hours']
            write_events.create_dataset(name='hours',
                                        data=trigger_hours)
            if "trigger_time_stamp_high" in stream:
                if "time_s" in write_events:
                    del write_events['time_s']
                if "time_mus" in write_events:
                    del write_events['time_mus']
                write_events.create_dataset(name='time_s',
                                            data=trigger_s)
                write_events.create_dataset(name='time_mus',
                                            data=trigger_mus)

            if 'event' in write_events:
                del write_events['event']
            write_events.create_dataset(name='event',
                                        shape=(self.nmbr_channels, len(trigger_hours), self.record_length / down),
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
                                                                          sample_duration=sample_duration,
                                                                          down=down)

            if exclude_tp:

                # ------------
                print('Include Testpulse Events.')

                if 'event' in write_testpulses:
                    del write_testpulses['event']
                tp_ev = write_testpulses.create_dataset(name='event',
                                                        shape=(self.nmbr_channels, len(
                                                            tp_hours[np.logical_and(tpas > min_tpa, tpas < min_cpa)]),
                                                               self.record_length / down),
                                                        dtype=datatype)

                this_hours = tp_hours[np.logical_and(tpas > min_tpa, tpas < min_cpa)]
                for c in range(self.nmbr_channels):

                    print('Channel ', c)
                    for i in range(len(this_hours)):
                        if i % 1000 == 0:
                            print('Calc Pulse Nmbr ', i)

                        tp_ev[c, i, :], _ = get_record_window(path=csmpl_paths[c],
                                                             start_time= this_hours[i]* 3600 - sample_to_time(
                                                                 self.record_length / 4,
                                                                 sample_duration=sample_duration),
                                                             record_length=self.record_length,
                                                             sample_duration=sample_duration,
                                                             down=down)

                # ------------
                print('Calculate Control Pulse Heights.')
                if 'pulse_height' in write_controlpulses:
                    del write_controlpulses['pulse_height']
                cphs = write_controlpulses.create_dataset(name="pulse_height",
                                                          shape=(self.nmbr_channels, len(tp_hours[tpas > min_cpa])),
                                                          dtype=float)
                this_hours = tp_hours[tpas > min_cpa]
                for c in range(self.nmbr_channels):

                    print('Channel ', c)
                    for i in range(len(this_hours)):
                        if i % 1000 == 0:
                            print('Calc Pulse Nmbr ', i)
                        cp_array, _ = get_record_window(path=csmpl_paths[c],
                                                        start_time=this_hours[i] * 3600 - sample_to_time(
                                                            self.record_length / 4,
                                                            sample_duration=sample_duration),
                                                        record_length=self.record_length,
                                                        sample_duration=sample_duration)

                        # subtract offset
                        cp_array -= np.mean(cp_array[:int(len(cp_array) / 8)])

                        # write the heights to file
                        cphs[c, ...] = np.max(cp_array)

            print('DONE')


    def include_test_stamps(self, path_teststamps, path_dig_stamps, path_sql, csmpl_channels, csmpl_file_identity,
                            triggerdelay=2081024,
                            samplediv=2,
                            sample_length=0.00004):

        # open file stream
        with h5py.File(self.path_h5, 'r+') as h5f:
            h5f.require_group('stream')

            # get start second from sql
            file_start = get_starttime(path_sql=path_sql,
                                       csmpl_channel=csmpl_channels[0],
                                       csmpl_file_identity=csmpl_file_identity)

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

            time_s = np.floor(file_start + hours * 3600)
            time_mus = 1e6 * (hours * 3600 - np.floor(hours * 3600))

            # write to file

            h5f['stream'].require_dataset(name='tp_hours',
                                          shape=hours.shape,
                                          dtype=float)
            h5f['stream'].require_dataset(name='tpa',
                                          shape=tpas.shape,
                                          dtype=float)
            h5f['stream'].require_dataset(name='tp_time_s',
                                          shape=hours.shape,
                                          dtype=int)
            h5f['stream'].require_dataset(name='tp_time_mus',
                                          shape=hours.shape,
                                          dtype=int)

            h5f['stream']['tp_hours'][...] = hours
            h5f['stream']['tpa'][...] = tpas
            h5f['stream']['tp_time_s'][...] = time_s
            h5f['stream']['tp_time_mus'][...] = time_mus

        print('Test Stamps included.')
