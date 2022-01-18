# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import h5py
from ..features._mp import calc_main_parameters
from ..trigger._csmpl import trigger_csmpl, get_record_window, align_triggers, sample_to_time, \
    exclude_testpulses, get_starttime, get_test_stamps, get_offset
from tqdm.auto import tqdm
from ..fit._pm_fit import fit_pulse_shape
from ..fit._templates import pulse_template
import os


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class CsmplMixin(object):
    """
    A Mixin Class to the DataHandler Class with methods for the triggering of *.csmpl files.
    """

    def include_ctrigger_stamps(self,
                                paths,
                                name_appendix='',
                                trigger_block: int = None,
                                path_sql: str = None,
                                csmpl_channels: list = None,
                                sql_file_label: str = None,
                                ):
        """
        Include time stamps from a CAT CTrigger data format.

        These are included in the group stream. Choose and appropriate
        name_appendix to distinguish them from time stamps that are calculated with the Cait trigger!

        The data format and method was described in "(2018) N. Ferreiro Iachellini, Increasing the sensitivity to
        low mass dark matter in cresst-iii witha new daq and signal processing", doi 10.5282/edoc.23762.

        :param paths: The paths to the *.csmpl.trig files that contain the time stamps from the CAT CTrigger.
        :type paths: tuple of strings
        :param name_appendix: A string that is appended to the HDF5 data sets trigger_hours, trigger_time_mus,
            trigger_time_s.
        :type name_appendix: string
        :param trigger_block: The value of samples that is blocked for triggering in all channels, after one trigger.
        :type trigger_block: int
        :param path_sql: Path to the SQL database file of the run.
        :type path_sql: string
        :param csmpl_channels: The csmpl channel numbers.
        :type csmpl_channels: list of ints
        :param sql_file_label: In the SQL database, we need to access the start time of file with the corresponding
            label of the file. The label can be looked up in the SQL file.
        :type sql_file_label: string
        """

        if trigger_block is None:
            trigger_block = self.record_length

        if path_sql is not None and (csmpl_channels is None or sql_file_label is None):
            raise KeyError(
                'If you want to read start time from SQL database, you must provide the csmpl_channels and sql_file_label!')

        dt = np.dtype([('ch', 'int64', (1,)),
                       ('tstamp', 'uint64', (1,)),
                       ('ftstamp', 'uint64', (1,)),
                       ('amp', 'float64', (1,)),
                       ('iamp', 'float64', (1,)),
                       ('rms', 'float64', (1,)),
                       ('tdelay', 'int64', (1,)),
                       ])

        time = []

        for p in paths:
            x = np.fromfile(file=p,
                            dtype=dt,
                            count=-1)

            time.append(x['tstamp'] / 10e6)

        # fix the number of triggers
        if len(paths) > 1:
            print('ALIGN TRIGGERS')
            aligned_triggers = align_triggers(triggers=time,  # in seconds
                                              trigger_block=trigger_block,
                                              sample_duration=1 / self.sample_frequency,
                                              )
        else:
            aligned_triggers = time[0].reshape(-1)

        with h5py.File(self.path_h5, 'a') as h5f:

            stream = h5f.require_group(name='stream')
            # write them to file
            print('ADD DATASETS TO HDF5')
            if "trigger_hours{}".format(name_appendix) in stream:
                print('overwrite old trigger_hours{}'.format(name_appendix))
                del stream['trigger_hours{}'.format(name_appendix)]
            stream.create_dataset(name='trigger_hours' + name_appendix,
                                  data=aligned_triggers / 3600)

            if path_sql is not None:
                # get file handle
                if "trigger_time_s{}".format(name_appendix) in stream:
                    print('overwrite old trigger_time_s{}'.format(name_appendix))
                    del stream['trigger_time_s{}'.format(name_appendix)]
                stream.create_dataset(name='trigger_time_s' + name_appendix,
                                      shape=(aligned_triggers.shape),
                                      dtype=int)
                if "trigger_time_mus{}".format(name_appendix) in stream:
                    print('overwrite old trigger_time_mus{}'.format(name_appendix))
                    del stream['trigger_time_mus{}'.format(name_appendix)]
                stream.create_dataset(name='trigger_time_mus{}'.format(name_appendix),
                                      shape=(aligned_triggers.shape),
                                      dtype=int)

                # get start second from sql
                file_start = get_starttime(path_sql=path_sql,
                                           csmpl_channel=csmpl_channels[0],
                                           sql_file_label=sql_file_label)

                stream['trigger_time_s{}'.format(name_appendix)][...] = np.array(aligned_triggers + file_start,
                                                                                 dtype='int32')
                stream['trigger_time_mus{}'.format(name_appendix)][...] = np.array(1e6 * (
                        aligned_triggers + file_start - np.floor(aligned_triggers + file_start)), dtype='int32')

            print('DONE')

    def include_csmpl_triggers(self,
                               csmpl_paths: list,  # list of all paths for the channels
                               thresholds: list,  # in V
                               trigger_block: int = None,
                               take_samples: int = None,
                               of: list = None,
                               path_dig: str = None,
                               clock: int = 10000000,
                               path_sql: str = None,
                               csmpl_channels: list = None,
                               sql_file_label: str = None,
                               down: int = 1,
                               window: bool = True,
                               overlap: float = None,
                               read_triggerstamps: bool = False,
                               ):
        """
        Trigger *.csmpl files of a detector module and include them in the HDF5 set.

        The trigger time stamps of all channels get aligned, by applying a trigger block to channels that belong to the
        same module. For determining the absolute time stamp of the triggers, we also need the SQL file that belongs to
        the measurement. The absolute time stamp will be precise only to seconds, as it is only stored with this
        precision in the SQL file. This is not a problem for our analysis, as all events within this files are down to
        micro seconds precisely matched to each other.

        New in v1.1: We can also read the start time from the PAR file. In this case, the start time needs to be already stored
        in the metainfo dataset. We still need to provide a DIG file, to calculate the offset between the files.
        If we do not want to exclude the offset, we don't need to provide a DIG file.

        The data format and method was described in "(2018) N. Ferreiro Iachellini, Increasing the sensitivity to
        low mass dark matter in cresst-iii witha new daq and signal processing", doi 10.5282/edoc.23762.

        :param csmpl_paths: The full paths for the csmpl files of all channels. If you want to trigger only one channel,
            then only put the path of this channel here.
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
        :param path_dig: Path to the DIG file, to read the offset between continuous DAQ and CCS. For this you need to
            include the metadata first! If not provided, no offset is calculated.
        :type path_dig: str
        :param clock: The frequency of the clock that times the data recording.
        :type clock: int
        :param csmpl_channels: The CDAQ channels that we are triggering. The channels numbers are usually appended to
            the file name of the CSMPL files and written in the SQL database.
        :type csmpl_channels: list of ints
        :param sql_file_label: In the SQL database, we need to access the start time of file with the corresponding
            label of the file. The label can be looked up in the SQL file.
        :type sql_file_label: string
        :param down: The downsampling factor for triggering.
        :type down: int
        :param window: If true, the trigger window is multiplied with a window function before the filtering.
            This is strongly recommended! Otherwise some artifacts from major differences in the left and right baseline
            level can appear somewhere in the middle of the record window.
        :type window: bool
        :param overlap: A value between 0 and 1 that defines the part of the record window that overlaps with the
            previous/next one. Standard value is 1/4 - it is recommended to use this value!
        :type overlap: float
        :param read_triggerstamps: In case there is already a trigger_hours data set in the HDF5 stream group, we can read
            it instead of doing the triggering again. For this, set this argument to True.
        :type read_triggerstamps: bool
        """

        assert all([os.path.isfile(p) for p in csmpl_paths]), 'One of the csmpl files does not exists!'
        if path_dig is not None:
            assert os.path.isfile(path_dig), 'Dig file does not exists!'
        if path_sql is not None:
            assert os.path.isfile(path_sql), 'Sql file does not exists!'
        assert np.logical_or(path_dig is None, path_sql is None), 'Read the start time either from PAR or SQL file!'

        if take_samples is None:
            take_samples = -1

        if trigger_block is None:
            trigger_block = self.record_length

        if path_sql is not None and (csmpl_channels is None or sql_file_label is None):
            raise KeyError(
                'If you want to read start time from SQL database, you must provide the csmpl_channels and sql_file_label!')

        # open write file
        with h5py.File(self.path_h5, 'a') as h5f:

            stream = h5f.require_group(name='stream')

            if not read_triggerstamps:

                if of is None:
                    print("Read OF Transfer Function from h5 file. Alternatively provide one through the of argument.")
                    of = np.zeros((self.nmbr_channels, int(self.record_length / 2 + 1)), dtype=np.complex)
                    of.real = h5f['optimumfilter']['optimumfilter_real']
                    of.imag = h5f['optimumfilter']['optimumfilter_imag']

                # do the triggering
                time = []
                for c in range(len(csmpl_paths)):
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
                                         window=window,
                                         overlap=overlap,
                                         )

                    time.append(trig)

                # fix the number of triggers
                if len(csmpl_paths) > 1:
                    print('ALIGN TRIGGERS')
                    aligned_triggers = align_triggers(triggers=time,  # in seconds
                                                      trigger_block=trigger_block,
                                                      sample_duration=1 / self.sample_frequency,
                                                      )
                else:
                    aligned_triggers = time[0]

                # write them to file
                print('ADD DATASETS TO HDF5')
                if "trigger_hours" in stream:
                    print('overwrite old trigger_hours')
                    del stream['trigger_hours']
                stream.create_dataset(name='trigger_hours',
                                      data=aligned_triggers / 3600)

            else:

                aligned_triggers = stream['trigger_hours'][:] * 3600

            if path_sql is not None or path_dig is not None:
                # get file handle
                if "trigger_time_s" in stream:
                    print('overwrite old trigger_time_s')
                    del stream['trigger_time_s']
                stream.create_dataset(name='trigger_time_s',
                                      shape=aligned_triggers.shape,
                                      dtype=np.int64)
                if "trigger_time_mus" in stream:
                    print('overwrite old trigger_time_mus')
                    del stream['trigger_time_mus']
                stream.create_dataset(name='trigger_time_mus',
                                      shape=aligned_triggers.shape,
                                      dtype=np.int64)

                if path_sql is not None:
                    # get start second from sql
                    file_start = get_starttime(path_sql=path_sql,
                                               csmpl_channel=csmpl_channels[0],
                                               sql_file_label=sql_file_label)  # in seconds

                    stream['trigger_time_s'][...] = np.array(aligned_triggers + file_start, dtype='int32')
                    stream['trigger_time_mus'][...] = np.array(1e6 * (
                            aligned_triggers + file_start - np.floor(aligned_triggers + file_start)), dtype='int32')

                elif path_dig is not None:

                        file_start_s = h5f['metainfo']['start_s'][()]
                        file_start_mus = h5f['metainfo']['start_mus'][()]
                        offset = get_offset(path_dig) / clock  # in seconds

                        stream['trigger_time_s'][...] = np.array(aligned_triggers + file_start_s + file_start_mus/1e6 - offset, dtype='int32')
                        stream['trigger_time_mus'][...] = np.array(1e6 * (
                                aligned_triggers + file_start_s + file_start_mus/1e6 - offset -
                                np.floor(aligned_triggers + file_start_s + file_start_mus/1e6 - offset)), dtype='int32')

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

    def include_sev(self,
                    sev,
                    fitpar=None,
                    mainpar=None,
                    scale_fit_height=True,
                    sample_length=None,
                    t0_start=None,
                    opt_start=False,
                    group_name_appendix='',
                    ):
        """
        Include the Standard Event to a HDF5 file.

        :param sev: The standard events of all channels.
        :type sev: array with shape (channels, samples)
        :param fitpar: The Proebst pulse shape fit parameters for all standard events.
        :type fitpar: array with shape (channels, 6)
        :param mainpar: The main parameters of all standard events.
        :type mainpar: array with shape (channels, 10)
        :param scale_fit_height: If true, the fitted standard event amplitude parameters get divided by 1/height of the
            fitted parametric model. Use this with caution! With this you intentionally deviate from the best fit
            parameters.
        :type scale_fit_height: bool
        :param sample_length: The sample length in milliseconds. If None, it is calculated from the sample frequency.
        :type sample_length: float
        :param t0_start: The start values for the fit of the parametric model, for all channels.
        :type t0_start: list of floats
        :param opt_start: If true, before the nelder-mead fit startes, a differential evolution fit is looking for suitable
            starting values.
        :type opt_start: bool
        :param group_name_appendix: A string that gets appended to the name stdevent of the group. Typically _tp for the
            test pulse standard event.
        :type group_name_appendix: string
        """

        if sample_length is None:
            sample_length = 1 / self.sample_frequency * 1000

        std_evs = []

        if t0_start is None:
            t0_start = [-3 for i in range(self.nmbr_channels)]

        for c in range(self.nmbr_channels):

            std_evs.append([])

            std_evs[c].append(sev[c])

            if fitpar is None:
                std_evs[c].append(
                    fit_pulse_shape(sev[c], sample_length=sample_length, t0_start=t0_start[c], opt_start=opt_start))

                if scale_fit_height:
                    t = (np.arange(0, len(sev[c]), dtype=float) - len(sev[c]) / 4) * sample_length
                    fit_max = np.max(pulse_template(t, *std_evs[c][1]))
                    print('Parameters [t0, An, At, tau_n, tau_in, tau_t]:\n', std_evs[c][1])
                    if not np.isclose(fit_max, 0):
                        std_evs[c][1][1] /= fit_max
                        std_evs[c][1][2] /= fit_max
            else:
                std_evs[c].append(fitpar[c])

        if mainpar is None:
            mainpar = np.array([calc_main_parameters(x[0]).getArray() for x in std_evs])

        with h5py.File(self.path_h5, 'r+') as f:

            stdevent = f.require_group(name='stdevent' + group_name_appendix)

            stdevent.require_dataset('event',
                                     shape=(self.nmbr_channels, len(std_evs[0][0])),  # this is then length of sev
                                     dtype='f')
            stdevent['event'][...] = np.array([x[0] for x in std_evs])
            stdevent.require_dataset('fitpar',
                                     shape=(self.nmbr_channels, len(std_evs[0][1])),
                                     dtype='f')
            stdevent['fitpar'][...] = np.array([x[1] for x in std_evs])

            # description of the fitparameters (data=column_in_fitpar)
            stdevent['fitpar'].attrs.create(name='t_0', data=0)
            stdevent['fitpar'].attrs.create(name='A_n', data=1)
            stdevent['fitpar'].attrs.create(name='A_t', data=2)
            stdevent['fitpar'].attrs.create(name='tau_n', data=3)
            stdevent['fitpar'].attrs.create(name='tau_in', data=4)
            stdevent['fitpar'].attrs.create(name='tau_t', data=5)

            stdevent.require_dataset('mainpar',
                                     shape=mainpar.shape,
                                     dtype='f')

            stdevent['mainpar'][...] = mainpar

            # description of the mainpar (data=col_in_mainpar)
            stdevent['mainpar'].attrs.create(name='pulse_height', data=0)
            stdevent['mainpar'].attrs.create(name='t_zero', data=1)
            stdevent['mainpar'].attrs.create(name='t_rise', data=2)
            stdevent['mainpar'].attrs.create(name='t_max', data=3)
            stdevent['mainpar'].attrs.create(name='t_decaystart', data=4)
            stdevent['mainpar'].attrs.create(name='t_half', data=5)
            stdevent['mainpar'].attrs.create(name='t_end', data=6)
            stdevent['mainpar'].attrs.create(name='offset', data=7)
            stdevent['mainpar'].attrs.create(name='linear_drift', data=8)
            stdevent['mainpar'].attrs.create(name='quadratic_drift', data=9)
        print('SEV written.')

    def include_of(self, of_real, of_imag, down=1, group_name_appendix=''):
        """
        Include the optimum filter transfer function into the HDF5 file.

        :param of_real: The real part of the transfer function.
        :type of_real: array of shape (channels, samples/2 + 1)
        :param of_imag: The imaginary part of the transfer function.
        :type of_imag: array of shape (channels, samples/2 + 1)
        :param down: The downsample rate of the transfer function.
        :type down: int
        :param group_name_appendix: A string that gets appended to the name optimumfilter of the group. Typically _tp for the
            test pulse standard event.
        :type group_name_appendix: string
        """
        with h5py.File(self.path_h5, 'r+') as f:
            optimumfilter = f.require_group(name='optimumfilter' + group_name_appendix)
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
                                 max_time_diff=0.5,  # in sec
                                 exclude_tp=True,
                                 sample_duration=None,
                                 name_appendix='',
                                 datatype='float32',
                                 min_tpa=0.0001,
                                 min_cpa=10.1,
                                 down=1,
                                 noninteractive=True,
                                 origin=None,
                                 individual_tpas=False,):
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
        :param sample_duration: The duration of a sample in seconds. If None, the inverse of the sample frequency is taken.
        :type sample_duration: float
        :param name_appendix: The name appendix of the data sets trigger_hours, trigger_time_s and trigger_time_mus in
            the HDF5 file. This is typically needed, when we want to include the events corresponding to time stamps
            that were triggered with the CAT CTrigger and then included in the HDF5 set.
        :type name_appendix: string
        :param datatype: The datatype of the stored events.
        :type datatype: string
        :param min_tpa: TPA values below this are not counted as testpulses.
        :type min_tpa: float
        :param min_cpa: TPA values above this are counted as control pulses.
        :type min_cpa: float
        :param down: The events get stored downsampled by this factor.
        :type down: int
        :param origin: The name of the csmpl file from which we read, e.g. bck_xxx
        :type origin: str
        :param individual_tpas: Write individual TPAs for the all channels. This results in a testpulseamplitude dataset
            of shape (nmbr_channels, nmbr_testpulses). Otherwise we have (nmbr_testpulses).
        :type individual_tpas: bool
        """

        if sample_duration is None:
            sample_duration = 1 / self.sample_frequency

        # open read file
        with h5py.File(self.path_h5, 'r+') as h5f:
            if "events" in h5f and not noninteractive:
                val = None
                while val != 'y':
                    val = input(
                        "An events group exists in this file. Overwriting with a different number of events, e.g."
                        "after retriggering, might lead to issues in feature calculations. Overwrite anyway? y/n")
                    if val == 'n':
                        raise FileExistsError(
                            "Events group exists already in the H5 file. Create new file to not overwrite.")

            stream = h5f['stream']
            trigger_hours = np.array(stream['trigger_hours{}'.format(name_appendix)])
            if "trigger_time_s{}".format(name_appendix) in stream:
                trigger_s = np.array(stream['trigger_time_s{}'.format(name_appendix)])
                trigger_mus = np.array(stream['trigger_time_mus{}'.format(name_appendix)])
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
                if origin is None:
                    write_controlpulses = h5f.require_group('controlpulses')
                    if "hours" in write_testpulses:
                        del write_testpulses["hours"]
                    if "testpulseamplitude" in write_testpulses:
                        del write_testpulses["testpulseamplitude"]
                    if "hours" in write_controlpulses:
                        del write_controlpulses["hours"]
                    write_testpulses.create_dataset(name="hours",
                                                    data=tp_hours[np.logical_and(tpas >= min_tpa, tpas < min_cpa)])
                    data = tpas[np.logical_and(tpas >= min_tpa, tpas < min_cpa)]
                    if individual_tpas:
                        data = np.tile(data, (self.nmbr_channels, 1))
                    write_testpulses.create_dataset(name="testpulseamplitude",
                                                    data=data)
                    write_controlpulses.create_dataset(name="hours",
                                                       data=tp_hours[tpas >= min_cpa])
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
                                                        data=tp_s[np.logical_and(tpas >= min_tpa, tpas < min_cpa)])
                        write_testpulses.create_dataset(name="time_mus",
                                                        data=tp_mus[np.logical_and(tpas >= min_tpa, tpas < min_cpa)])
                        write_controlpulses.create_dataset(name="time_s",
                                                           data=tp_s[tpas >= min_cpa])
                        write_controlpulses.create_dataset(name="time_mus",
                                                           data=tp_mus[tpas >= min_cpa])

                print('Exclude Testpulses.')
                flag = exclude_testpulses(trigger_hours=trigger_hours,
                                          tp_hours=tp_hours[tpas >= min_tpa],
                                          max_time_diff=max_time_diff)

                trigger_hours = trigger_hours[flag]
                if "trigger_time_s{}".format(name_appendix) in stream:
                    trigger_s = trigger_s[flag]
                    trigger_mus = trigger_mus[flag]

            if origin is None:
                if 'hours' in write_events:
                    del write_events['hours']
                write_events.create_dataset(name='hours',
                                            data=trigger_hours)
                if "trigger_time_s{}".format(name_appendix) in stream:
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
                                            shape=(
                                                self.nmbr_channels, len(trigger_hours), int(self.record_length / down)),
                                            dtype=datatype)
            else:
                write_events.require_dataset(name='event',
                                             shape=(
                                                 self.nmbr_channels, len(trigger_hours),
                                                 int(self.record_length / down)),
                                             dtype=datatype)

            print('Include the triggered events.')
            for c in range(self.nmbr_channels):

                print('Channel ', c)
                if origin is None:
                    iterable = range(len(trigger_hours))
                else:
                    iterable = np.array([st.decode() == origin for st in h5f['events']['origin'][:]]).nonzero()[0]
                for i in tqdm(iterable):
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

                this_hours = tp_hours[np.logical_and(tpas > min_tpa, tpas < min_cpa)]

                if origin is None:
                    if 'event' in write_testpulses:
                        del write_testpulses['event']
                    tp_ev = write_testpulses.create_dataset(name='event',
                                                            shape=(self.nmbr_channels, len(this_hours),
                                                                   int(self.record_length / down)),
                                                            dtype=datatype)
                else:
                    tp_ev = write_testpulses.require_dataset(name='event',
                                                             shape=(self.nmbr_channels, len(this_hours),
                                                                    int(self.record_length / down)),
                                                             dtype=datatype)

                for c in range(self.nmbr_channels):

                    print('Channel ', c)
                    if origin is None:
                        iterable = range(len(this_hours))
                    else:
                        iterable = np.array([st.decode() == origin for st in h5f['testpulses']['origin'][:]]).nonzero()[
                            0]
                    for i in tqdm(range(len(iterable))):
                        tp_ev[c, i, :], _ = get_record_window(path=csmpl_paths[c],
                                                              start_time=this_hours[i] * 3600 - sample_to_time(
                                                                  self.record_length / 4,
                                                                  sample_duration=sample_duration),
                                                              record_length=self.record_length,
                                                              sample_duration=sample_duration,
                                                              down=down)

                # ------------
                if origin is None:
                    print('Calculate Control Pulse Heights.')
                    if 'pulse_height' in write_controlpulses:
                        del write_controlpulses['pulse_height']
                    cphs = write_controlpulses.create_dataset(name="pulse_height",
                                                              shape=(self.nmbr_channels, len(tp_hours[tpas > min_cpa])),
                                                              dtype=float)
                    this_hours = tp_hours[tpas > min_cpa]
                    for c in range(self.nmbr_channels):

                        print('Channel ', c)
                        for i in tqdm(range(len(this_hours))):
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

    def include_test_stamps(self, path_teststamps, path_dig_stamps, path_sql=None, csmpl_channels=None,
                            sql_file_label=None,
                            clock=10000000,
                            fix_offset=True,
                            ):
        """
        Include the test pulse time stamps in the HDF5 data set.

        If the SQL path is not provided, then the filestart is read from the metainfo (recommended)!

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
        :param clock: The Frequency of the time clock, in Hz. Standard for CRESST is 10MHz.
        :type clock: int
        :param fix_offset: This fixes the time offset between the trigger time stamps and the DAQ time stamps. Strongly
            recommended!!
        :type fix_offset: bool
        """

        # open file stream
        with h5py.File(self.path_h5, 'a') as h5f:
            h5f.require_group('stream')

            if path_sql is not None:
                # get start second from sql
                file_start = get_starttime(path_sql=path_sql,
                                           csmpl_channel=csmpl_channels[0],
                                           sql_file_label=sql_file_label)

            else:

                file_start = h5f['metainfo']['start_s'][()]
                file_start += 1e-6 * h5f['metainfo']['start_mus'][()]

            # read the test pulse time stamps from the test_stamps file

            hours, tpas, _ = get_test_stamps(path=path_teststamps, channels=[0])

            if fix_offset:
                # determine the offset of the trigger time stamps from the digitizer stamps

                offset_hours = get_offset(path_dig_stamps) / clock / 3600  # the dig stamp is delayed by offset_hours

                # remove the offset and throw all below zero away

                hours += offset_hours
                tpas = tpas[hours > 0]
                hours = hours[hours > 0]

            # calc the time stamp seconds and mus

            time_s = np.array(file_start + hours * 3600, dtype='int32')
            time_mus = np.array(1e6 * (file_start + hours * 3600 - np.floor(file_start + hours * 3600)), dtype='int32')

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

    def include_noise_triggers(self,
                               nmbr,
                               min_distance=0.5,
                               max_distance=60,
                               record_window_length=None,
                               max_attempts=5,
                               no_pileup=False,
                               ):
        """
        Include a number of random triggers from the Stream.

        The random triggers are only choosen in intervals that are measurement time (frequently occuring test pulses)
        and that are away from test pulses. The triggers are stored in the stream group in the HDF5 file.

        The data format and method was described in "(2018) N. Ferreiro Iachellini, Increasing the sensitivity to
        low mass dark matter in cresst-iii witha new daq and signal processing", doi 10.5282/edoc.23762.

        :param nmbr: The number of noise triggers we want to include.
        :type nmbr: int
        :param min_distance: The minimal distance in seconds of the start and end of a noise trigger window from a
            test pulse. If you choose record_length * 0.8 or higher, you avoid pile up.
        :type min_distance: float
        :param max_distance: The maximal distance of two test pulses in seconds, such that the interval in between still
            counts as measurement time.
        :type max_distance: float
        :param record_window_length: The length of the record window in seconds. If None, it is calculated from the
            record length and sample frequency.
        :type record_window_length: float
        :param max_attempts: In case the chosen time stamp is piled up with an already chosen noise trigger, we chose
            another time stamp. This counts as one attempt. If the maximal number of attempts is exceeded, we skip one
            noise trigger and try for another gap. In this case, we will include one less time stamp than the parameter
            nmbr. This procedure prevents infinite loops due to a too high nmbr parameter.
        :type max_attempts: int
        :param no_pileup: If activated, not only test pulses but also triggered events are excluded from the noise
            trigger windows.
        :type no_pileup: bool
        """

        if record_window_length is None:
            record_window_length = self.record_length / self.sample_frequency

        min_distance /= 3600  # all in hours
        max_distance /= 3600
        record_window_length /= 3600

        # open file stream
        with h5py.File(self.path_h5, 'r+') as h5f:
            test_stamps = h5f['stream']['tp_hours']

            if no_pileup:
                trigger_stamps = h5f['stream']['trigger_hours']
                all_stamps = np.concatenate((trigger_stamps, test_stamps))
                all_stamps.sort(kind='mergesort')
            else:
                all_stamps = test_stamps
            gaps = np.diff(all_stamps)
            gaps_idx = np.arange(start=0, stop=len(gaps), step=1)
            good_gaps_flag = np.logical_and(gaps > 2 * min_distance + record_window_length,
                                            gaps < max_distance)
            good_gaps_flag[0] = 0  # this prevents a bug in the loop later on

            noise_triggers = np.zeros(nmbr)

            pre_dist = record_window_length / 4 + min_distance
            post_dist = record_window_length * 3 / 4 + min_distance

            free_time = np.sum(gaps[good_gaps_flag])
            probabilities = gaps[good_gaps_flag] / free_time

            from_gap = np.zeros(nmbr)

            print('Found {} suitable gaps in stream, with total time of {} h.'.format(np.sum(good_gaps_flag),
                                                                                      free_time))

            for i in tqdm(range(nmbr)):

                attempts = 0

                while attempts < max_attempts:

                    # choose one of the good gaps
                    idx = np.random.choice(gaps_idx[good_gaps_flag],
                                           size=1,
                                           p=probabilities)

                    # take a time stamp within this gap
                    trig = np.random.uniform(low=all_stamps[idx] + pre_dist,
                                             high=all_stamps[idx + 1] - post_dist)
                    attempts += 1

                    # check if the time stamp is piled up with another already chosen time stamp
                    if all(np.abs(trig - noise_triggers[from_gap == idx]) > record_window_length):
                        noise_triggers[i] = trig
                        from_gap[i] = idx
                        break

            if all(noise_triggers != 0):
                print('Success, include {} noise triggers.'.format(nmbr))
            else:
                print('Fail, include only {} noise triggers.'.format(np.sum(noise_triggers != 0)))

            noise_triggers = noise_triggers[noise_triggers != 0]
            noise_triggers.sort()

            # match the s and mus stamps
            first_tp_hours = test_stamps[0]
            first_tp = h5f['stream']['tp_time_s'][0] + 10e-6 * h5f['stream']['tp_time_mus'][0]  # in sec

            difference_s = (noise_triggers - first_tp_hours) * 3600
            noise_trigger_s = np.floor(first_tp + difference_s)
            noise_trigger_mus = np.floor(((first_tp + difference_s) % 1) * 1e6)

            h5f['stream'].require_dataset(name='noise_hours',
                                          shape=noise_triggers.shape,
                                          dtype=float)
            h5f['stream'].require_dataset(name='noise_time_s',
                                          shape=noise_trigger_s.shape,
                                          dtype=int)
            h5f['stream'].require_dataset(name='noise_time_mus',
                                          shape=noise_trigger_mus.shape,
                                          dtype=int)

            h5f['stream']['noise_hours'][...] = noise_triggers
            h5f['stream']['noise_time_s'][...] = noise_trigger_s
            h5f['stream']['noise_time_mus'][...] = noise_trigger_mus

        print('Done.')

    def include_noise_events(self,
                             csmpl_paths,
                             datatype='float32',
                             origin=None,
                             down=1,
                             ):
        """
        Include the events corresponding to chosen noise triggers.

        :param csmpl_paths: The paths to the *.csmpl files of all channels, should be nmbr_channels long.
        :type csmpl_paths: list of strings
        :param datatype: The datatype of the events that we want to store. Typically float32 is a good choice, float16
            would lead to significantly reduced precision.
        :type datatype: string
        :param origin: This is needed in case you want to include events to a dataset that itself is a merge of other
            datasets. Then typically a set of origin strings is included in the set, which specify the origin of the
            individual events within the merged set. By putting a origin string here, the events get included in the
            event data set exactly at the positions where the origin attribute is the same as the handed origin string.
        :type origin: string
        :param down: The factor by which we want to downsample the included events.
        :type down: int
        """

        with h5py.File(self.path_h5, 'r+') as h5f:

            # get the time stamps
            noise_hours = h5f['stream']['noise_hours']
            noise_time_s = h5f['stream']['noise_time_s']
            noise_time_mus = h5f['stream']['noise_time_mus']

            nmbr_all_events = len(noise_hours)

            # make data sets
            noise = h5f.require_group('noise')
            noise.require_dataset(name='event',
                                  shape=(
                                      self.nmbr_channels, nmbr_all_events,
                                      int(self.record_length / down)),
                                  dtype=datatype)

            if origin is None:
                hours_h5 = noise.require_dataset(name='hours',
                                                 shape=(nmbr_all_events),
                                                 dtype=float)
                time_s_h5 = noise.require_dataset(name='time_s',
                                                  shape=(nmbr_all_events),
                                                  dtype=int)
                time_mus_h5 = noise.require_dataset(name='time_mus',
                                                    shape=(nmbr_all_events),
                                                    dtype=int)
                hours_h5[...] = noise_hours
                time_s_h5[...] = noise_time_s
                time_mus_h5[...] = noise_time_mus

            # get the record windows
            print('Include the triggered events.')
            for c in range(self.nmbr_channels):
                print('Channel ', c)
                if origin is None:
                    iterable = range(nmbr_all_events)
                else:
                    iterable = np.array([st.decode() == origin for st in noise['origin'][:]]).nonzero()[0]
                for i in tqdm(iterable):
                    noise['event'][c, i, :], _ = get_record_window(path=csmpl_paths[c],
                                                                   start_time=noise_hours[
                                                                                  i] * 3600 - sample_to_time(
                                                                       self.record_length / 4,
                                                                       sample_duration=1 / self.sample_frequency),
                                                                   record_length=self.record_length,
                                                                   sample_duration=1 / self.sample_frequency,
                                                                   down=down)

            print('Done.')
