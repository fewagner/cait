************************
Stream Triggering Utils
************************

.. code:: python

    """
    trigger_utils.py

    This file contains helper functions that are needed for the triggering and efficiency simulation. Put it in the same directory as the trigger.py and efficiency_simulation.py scripts.

    Usage:

    The function read_hw_data is optimized for the analysis of a phonon + light channel detector. You might have to adapt the imports of the XY files to your
    detector accordingly.

    """

    import numpy as np
    import cait as ai


    def get_filestart(path_file, args):
        dh = ai.DataHandler(channels=args['rdt_channels'],
                            record_length=args['record_length'],
                            sample_frequency=args['sample_frequency'])

        path, fname = path_file.rsplit('/', 1)
        fname = fname.split('.')[0]

        dh.set_filepath(path_h5=path,
                        fname=fname,
                        appendix=False)

        try:
            start_s = dh.get('metainfo', 'start_s')
            start_mus = dh.get('metainfo', 'start_mus')
        except:
            raise ValueError('No metainfo in the Hdf5 file - did you include the info from PAR file?')

        if len(start_s.shape) == 0:
            pass
        elif len(start_s.shape) == 1:
            start_s = start_s[0]
            start_mus = start_mus[0]
        else:
            raise ValueError('Shape of start times in HDF5 files weird - malicious file?')

        return start_s, start_mus


    def read_hw_data(args):
        xy_files = {
            'of': np.zeros((len(args['rdt_channels']), int(args['record_length'] / 2) + 1), dtype=complex),
            'of_direct': np.zeros((len(args['rdt_channels']), int(args['record_length'] / 2) + 1), dtype=complex),
            'of_tp': np.zeros((len(args['rdt_channels']), int(args['record_length'] / 2) + 1), dtype=complex),
            'sev': np.zeros((len(args['rdt_channels']), args['record_length']), dtype=float),
            'sev_mainpar': np.zeros((len(args['rdt_channels']), 10), dtype=float),
            'sev_fitpar': np.zeros((len(args['rdt_channels']), 6), dtype=float),
            'sev_direct': np.zeros((len(args['rdt_channels']), args['record_length']), dtype=float),
            'sev_direct_mainpar': np.zeros((len(args['rdt_channels']), 10), dtype=float),
            'sev_direct_fitpar': np.zeros((len(args['rdt_channels']), 6), dtype=float),
            'sev_tp': np.zeros((len(args['rdt_channels']), args['record_length']), dtype=float),
            'sev_tp_mainpar': np.zeros((len(args['rdt_channels']), 10), dtype=float),
            'sev_tp_fitpar': np.zeros((len(args['rdt_channels']), 6), dtype=float),
            'nps': np.zeros((len(args['rdt_channels']), int(args['record_length'] / 2) + 1), dtype=float),
        }

        for i, c in enumerate(args['rdt_channels']):

            of = ai.data.read_xy_file(args['xy_files'] + 'Channel_{}_OF_Particle.xy'.format(c), skip_lines=4)
            xy_files['of'][i] += of[:, 1]
            xy_files['of'][i] += 1j * of[:, 2]

            if i == 1:
                of_direct = ai.data.read_xy_file(args['xy_files'] + 'Channel_{}_OF_Direct.xy'.format(c), skip_lines=4)
                xy_files['of_direct'][i] += of_direct[:, 1]
                xy_files['of_direct'][i] += 1j * of_direct[:, 2]

            of_tp = ai.data.read_xy_file(args['xy_files'] + 'Channel_{}_OF_TP.xy'.format(c), skip_lines=4)
            xy_files['of_tp'][i] += of_tp[:, 1]
            xy_files['of_tp'][i] += 1j * of_tp[:, 2]

            xy_files['sev'][i] += ai.data.read_xy_file(args['xy_files'] + 'Channel_{}_SEV_Particle.xy'.format(c),
                                                       skip_lines=3)[:, 1]
            xy_files['sev_mainpar'][i] += ai.data.read_xy_file(
                args['xy_files'] + 'Channel_{}_SEV_Particle_Mainpar.xy'.format(c), skip_lines=2)
            xy_files['sev_fitpar'][i] += ai.data.read_xy_file(
                args['xy_files'] + 'Channel_{}_SEV_Particle_Fitpar.xy'.format(c), skip_lines=2)

            if i == 1:
                xy_files['sev_direct'][i] += ai.data.read_xy_file(args['xy_files'] + 'Channel_{}_SEV_Direct.xy'.format(c),
                                                                  skip_lines=3)[:, 1]
                xy_files['sev_direct_mainpar'][i] += ai.data.read_xy_file(
                    args['xy_files'] + 'Channel_{}_SEV_Direct_Mainpar.xy'.format(c), skip_lines=2)
                xy_files['sev_direct_fitpar'][i] += ai.data.read_xy_file(
                    args['xy_files'] + 'Channel_{}_SEV_Direct_Fitpar.xy'.format(c), skip_lines=2)

            xy_files['sev_tp'][i] += ai.data.read_xy_file(args['xy_files'] + 'Channel_{}_SEV_TP.xy'.format(c),
                                                          skip_lines=3)[:, 1]
            xy_files['sev_tp_mainpar'][i] += ai.data.read_xy_file(
                args['xy_files'] + 'Channel_{}_SEV_TP_Mainpar.xy'.format(c), skip_lines=2)
            xy_files['sev_tp_fitpar'][i] += ai.data.read_xy_file(args['xy_files'] + 'Channel_{}_SEV_TP_Fitpar.xy'.format(c),
                                                                 skip_lines=2)

            xy_files['nps'][i] += ai.data.read_xy_file(args['xy_files'] + 'Channel_{}_NPS.xy'.format(c), skip_lines=3)[:, 1]

        return xy_files

