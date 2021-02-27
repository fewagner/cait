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
    A Mixin Class to the DataHandler Class with methods for the triggering of *.csmpl files.
    """

    def include_csmpl_triggers(self,
                               csmpl_paths: list,  # list of all paths for the channels
                               thresholds: list,  # in V
                               trigger_block:int=None,
                               take_samples:int=None,
                               of:list=None,
                               path_sql:str=None,
                               csmpl_channels:list=None,
                               sql_file_label:str=None,
                               down:int=1
                               ):
        """
        Trigger *.csmpl files of a detector module and include them in the HDF5 set.

        The trigger time stamps of all channels get aligned, by applying a trigger block to channels that belong to the
        same module. For determining the absolute time stamp of the triggers, we also need the SQL file that belongs to
        the measurement. The absolute time stamp will be precise only to seconds, as it is only stored with this
        precision in the SQL file. This is not a problem for out analysis, as all events within this files are down to
        micro seconds precisely matched to each other.

        :param csmpl_paths: The full paths for the csmpl files of all channels.
        :type csmpl_paths: list of strings
        :param thresholds: The trigger tresholds for all channels in Volt.
        :type thresholds: list of floats
        :param trigger_block: The number of samples for that the trigger is blocked after a trigger. If None, it is
            the length of a record window.
        :type trigger_block: int or None
        :param take_samples: The number of samples that we trigger from the stream file. Standard argument is None, which
            means all samples are triggered.
        :type take_samples: int or None
        :param of: The optimum filter transfer functions for all channels.
        :type of: list of arrays
        :param path_sql: The path to the SQL database that contains the start of file timestamp.
        :type path_sql: string
        :param csmpl_channels: The CDAQ channels that we are triggering. The channels numbers are usually appended to
            the file name of the CSMPL files and written in the SQL database.
        :type csmpl_channels: list of ints
        :param sql_file_label: In the SQL database, we need to access the start time of file with the corresponding
            label of the file. The label can be looked up in the SQL file.
        :type sql_file_label: string
        :param down: The downsampling factor for triggering.
        :type down: int
        """

        if take_samples is None:
            take_samples = -1

        if trigger_block is None:
            trigger_block = self.record_length

        if path_sql is not None and (csmpl_channels is None or sql_file_label is None):
            raise KeyError(
                'If you want to read start time from SQL database, you must provide the csmpl_channels and sql_file_label!')

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
                                     start_hours=0,
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
                if "trigger_time_s" in stream:
                    print('overwrite old trigger_time_s')
                    del stream['trigger_time_s']
                stream.create_dataset(name='trigger_time_s',
                                      shape=(aligned_triggers.shape),
                                      dtype=int)
                if "trigger_time_mus" in stream:
                    print('overwrite old trigger_time_mus')
                    del stream['trigger_time_mus']
                stream.create_dataset(name='trigger_time_mus',
                                      shape=(aligned_triggers.shape),
                                      dtype=int)

                # get start second from sql
                file_start = get_starttime(path_sql=path_sql,
                                           csmpl_channel=csmpl_channels[0],
                                           sql_file_label=sql_file_label)

                stream['trigger_time_s'][...] = np.array(aligned_triggers + file_start, dtype='int32')
                stream['trigger_time_mus'][...] = np.array(1e6 * (
                        aligned_triggers + file_start - np.floor(aligned_triggers + file_start)), dtype='int32')

            print('DONE')


    def include_nps(self, nps):
        """
        Include the Noise Power Spectrum to the HDF5 file.

        :param nps: The optimum filter transfer function.
        :type nps: array of shape (channels, samples/2 + 1)
        """
        with h5py.File(self.path_h5, 'r+') as f:
            noise = f.require_group(name='noise')
            if 'nps' in noise:
                del noise['nps']
            noise.create_dataset(name='nps',
                                 data=nps)
        print('NPS written.')


    def include_sev(self, sev, fitpar, mainpar):
        """
        Include the Standard Event to a HDF5 file.

        :param sev: The standard events of all channels.
        :type sev: array with shape (channels, samples)
        :param fitpar: The Proebst pulse shape fit parameters for all standard events.
        :type fitpar: array with shape (channels, 6)
        :param mainpar: The main parameters of all standard events.
        :type mainpar: array with shape (channels, 10)
        """
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
        print('SEV written.')


    def include_of(self, of_real, of_imag, down=1):
        """
        Include the optimum filter transfer function into the HDF5 file.

        :param of_real: The real part of the transfer function.
        :type of_real: array of shape (channels, samples/2 + 1)
        :param of_imag: The imaginary part of the transfer function.
        :type of_imag: array of shape (channels, samples/2 + 1)
        :param down: The downsample rate of the transfer function.
        :type down: int
        """
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
        print('OF written.')


    def include_triggered_events(self,
                                 csmpl_paths,
                                 max_time_diff=0.03, # in sec
                                 exclude_tp=True,
                                 sample_duration=0.00004,
                                 datatype='float32',
                                 min_tpa=0.001,
                                 min_cpa=10.1,
                                 down=1):
        """
        Include the triggered events from the CSMPL files.

        It is recommended to exclude the testpulses. This means, we exclude them from the events group and put them
        separately in the testpulses and control pulses groups.

        :param csmpl_paths: The full paths for the csmpl files of all channels.
        :type csmpl_paths: list of strings
        :param max_time_diff: The maximal time difference between a trigger and a test pulse time stamp such that the
            trigger is still counted as test pulse.
        :type max_time_diff: float
        :param exclude_tp: If true, we separate the test pulses from the triggered events and put them in individual
            groups.
        :type exclude_tp: bool
        :param sample_duration: The duration of a sample in seconds.
        :type sample_duration: float
        :param datatype: The datatype of the stored events.
        :type datatype: string
        :param min_tpa: TPA values below this are not counted as testpulses.
        :type min_tpa: float
        :param min_cpa: TPA values above this are counted as control pulses.
        :type min_cpa: float
        :param down: The events get stored downsampled by this factor.
        :type down: int
        """

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
            if "trigger_time_s" in stream:
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
                                        shape=(self.nmbr_channels, len(trigger_hours), int(self.record_length / down)),
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
                                                               int(self.record_length / down)),
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


    def include_test_stamps(self, path_teststamps, path_dig_stamps, path_sql, csmpl_channels, sql_file_label,
                            triggerdelay=2081024,
                            samplediv=2,
                            sample_length=0.00004):
        """
        Include the test pulse time stamps in the HDF5 data set.

        :param path_teststamps: The path to the TEST_STAMPS file.
        :type path_teststamps: string
        :param path_dig_stamps: The path to the DIG_STAMPS file.
        :type path_dig_stamps: string
        :param path_sql: The path to the SQL database that contains the start of file timestamp.
        :type path_sql: string
        ::param csmpl_channels: The CDAQ channels that we are triggering. The channels numbers are usually appended to
            the file name of the CSMPL files and written in the SQL database.
        :type csmpl_channels: list of ints
        :param sql_file_label: In the SQL database, we need to access the start time of file with the corresponding
            label of the file. The label can be looked up in the SQL file.
        :type sql_file_label: string
        :param triggerdelay: The number of samples recorded in one bankswitch.
        :type triggerdelay: int
        :param samplediv: The oversampling rate by the CDAQ compared to the saved sample rate.
        :type samplediv: int
        :param sample_length: The duration of a sample in seconds.
        :type sample_length: float
        """

        # open file stream
        with h5py.File(self.path_h5, 'r+') as h5f:
            h5f.require_group('stream')

            # get start second from sql
            file_start = get_starttime(path_sql=path_sql,
                                       csmpl_channel=csmpl_channels[0],
                                       sql_file_label=sql_file_label)

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

            # calc the time stamp seconds and mus

            time_s = np.array(file_start + hours * 3600, dtype='int32')
            time_mus = np.array(1e6 * (hours * 3600 - np.floor(hours * 3600)), dtype='int32')

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
