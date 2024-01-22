***********
Quick Start
***********

The quickest way to install Cait is via the Python package index:

.. code:: console

    $ pip install cait

For your first steps, you can create a standard event of data from a given `*.rdt` file with a few lines
of code. In this example, we create a mock data set to demonstrate the calculation of main parameters and the
creation of the standard event.

.. code:: python

    >>> import cait as ai
    >>> test_data = ai.data.TestData(filepath='test_data/mock_001', duration=1800)
    >>> test_data._generate_rdt_file()
    Rdt file written.
    >>> dh = ai.DataHandler(channels=[0,1])
    DataHandler Instance created.
    >>> dh.convert_dataset(path_rdt='test_data/', fname='mock_001', path_h5='test_data/', tpa_list=[1., 0., -1.])
    Start converting.

    READ EVENTS FROM RDT FILE.
    Total Records in File:  800
    Getting good idx. (Depending on OS and drive reading speed, this might take some minutes!)
    Event Counts Channel 0: 400
    Event Counts Channel 1: 400
    Getting good tpas.
    Good consecutive counts: 400

    WORKING ON EVENTS WITH TPA = 0.
    CREATE DATASET WITH EVENTS.
    100%|████████████████████████████████████████████████████████████| 160/160 [00:00<00:00, 668.69it/s]

    WORKING ON EVENTS WITH TPA = -1.
    CREATE DATASET WITH NOISE.
    100%|███████████████████████████████████████████████████████████| 160/160 [00:00<00:00, 2426.41it/s]

    WORKING ON EVENTS WITH TPA > 0.
    CREATE DATASET WITH TESTPULSES.
    100%|███████████████████████████████████████████████████████████| 480/480 [00:00<00:00, 4532.36it/s]
    Hdf5 dataset created in  test_data/
    Filepath and -name saved.
    >>> dh.calc_mp()
    CALCULATE MAIN PARAMETERS.
    >>> dh.calc_sev(decay_time_interval=[(5, 6), (0, 100)])

    Calculating SEV for Channel 0
    80 Events handed.
    43 left after decay time cut.
    43 Events used to generate Standardevent.
    Parameters [t0, An, At, tau_n, tau_in, tau_t]:
     [-1.10013111  3.19462707 -0.12586458  4.73959075  2.08445536  0.36166834]

    Calculating SEV for Channel 1
    80 Events handed.
    80 left after decay time cut.
    80 Events used to generate Standardevent.
    Parameters [t0, An, At, tau_n, tau_in, tau_t]:
     [6.36436168e-01 1.60604957e+00 3.09349466e-03 3.27436757e+01
     4.43479367e+00 1.00000068e-02]
    events SEV calculated.
    >>> dh.show_sev(channel=0)

.. image:: documentation/pics/test_sev.png

Once you accomplished this first step into the world of raw data analysis,  start going through the tutorial notebooks,
that demonstrate most of the functionality of Cait.

For producing efficient and fast physics results, check out or hardware- and stream data analysis templates, and the
trigger script! These include the essential steps of the analysis, easily adaptable to any detector.