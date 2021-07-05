*******************************
Estimate Noise Trigger Rate
*******************************

.. code:: python

    """
    estimate_noise_trigger_rate.py

    Run this script with suitable parameters on an x/y file of the pulse heights of your filtered empty baselines to
    estimate the noise trigger rate. The first four lines of the file will always be skipped. Afterward the file
    can have as many columns (separated with tabulator) as wanted, the script will always only extract the first column
    and interpret it as the heights of the individual empty baselines (unbinned) in Volt.

    Usage:

    1)You need the Cait library installed to use the script, follow the steps:
    - git clone https://git.cryocluster.org/fwagner/cait.git
    - cd cait
    - git checkout 1.0.X
    - pip install -e .

    2)Adapt the parameters of the script accordingly.
    - First, set the paths to the file with the data (pulse heights in V, PATH_XY_FILE) and where you want to save
        the plots (PATH_SAVE, e.g. './detecotor_name' --> will produce plots './detector_name_gauss_counts.pdf').
    - You can also load the data from an HDF5 file instead, from the group 'noise', and the data set 'of_ph'. If you
        want to do, so set the parameter IS_HDF5 to True and the CHANNEL to the channel for which you want to calculate
        the threshold.
    - The parameter CUT_PH_VALUE tells which is the upper limit for baselines that you want to exclude from the fit.
        This is to clean some large, coincident pulses. Typically a value between 15 and 30 mV is a good choice.
        The value should be much higher than the peak of the noise blob (at least 4 times higher).
    - RECORD_LENGTH is the number of samples within a record window.
    - SAMPLE_LENGTH is the inverse of the samples frequency.
    - DETECTOR_MASS is the mass of the crystal inside which the scattering happens.
    - INTERVAL_RESTRICTION is the share of the record window inside which the search for a maximum happens. For
        the implementation of the optimum filter in Cait and CAT, this is 0.75 (we exclude the first and last eight)
        to eliminate boundary effects. If you do the fit on the maxima of the unfiltered window, e.g. those one gets
        in CAT with cmp and in Cait with the mainparameters, then the maximum search is not restricted,
        i.e. set INTERVAL_RESTRICTION to the value 1.
    - SIGMA_X0 is the start value for the fit of the baseline resolution in mV. A good start value is around one third
        or one fourth of the majority of the empty baseline maxima. If the fit does not work properly, play around with
        this value!
    - Usually all other values you can leave as is. These mostly influence only the plot ranges and binning,
        but not the calculation.

    3) Run the script and look at the report that are written to your command line and the corresponding plots.
    """

    # imports
    import cait as ai
    import numpy as np
    import h5py

    # --------------------------------------------------
    # Adapt these parameters to your data or try the suggested values!
    # --------------------------------------------------

    # parameters
    # PATH_XY_FILE = 'height_of_empty_baselines.xy'  # path to the file with the data
    PATH_XY_FILE = 'TUM93AL_BLDiff_SpikeCut.xy'
    IS_HDF5 = False
    CHANNEL = 0
    PATH_SAVE = './filename'  # the path to the directory to save the plots
    CUT_PH_VALUE = 0.05  # e.g. 0.05, in V
    RECORD_LENGTH = 16384  # e.g. 16384
    SAMPLE_LENGTH = 0.00004  # e.g. 0.00004
    DETECTOR_MASS = 0.0005  # ~ density/cm^3 * 1 * 2 * 2, in kg
    SIGMA_X0 = 2  # try 2, play with this value in case the fit does not work
    INTERVAL_RESTRICTION = 0.75  # e.g. 0.75 if first and last eight is excluded (typical for OF application)
    UPPER_LIMIT = (CUT_PH_VALUE) * 1000  # e.g.  (CUT_PH_VALUE * 2)*1000, in mV
    LOWER_LIMIT = 0  # e.g. 0, in mV
    ALLOWED_NOISE_TRIGGERS = 1  # e.g. 1
    Y_LOG_SCALE = True  # set true to have logarithmic y scale on the histogram
    BINS = 200  # e.g. 200
    PLOT_RANGE_Y = None  # (5e-1, 10e8)  # the lower and upper limit of the noise trigger plot (second)
    PLOT_RANGE_X = None  # (0, UPPER_LIMIT)  # the lower and upper limit of the plots x axis
    MODELS = [
        'gauss',  # all samples are Gaussian distributed
        'pollution_exponential',  # one sample is exponentially distributed
        #'fraction_exponential',  # all samples have a gaussian/exponential mixture distribution
        #'pollution_gauss',  # one sample is Gaussian but with different mean and sigma
        #'fraction_gauss',  # all samples have a two component gaussian mixture distribution
    ]

    # --------------------------------------------------
    # All below this line does not need to be changed.
    # --------------------------------------------------

    # load the unbinned event heights from xy file
    if IS_HDF5:
        with h5py.open(PATH_XY_FILE, 'r') as f:
            x = f['noise']['of_ph'][CHANNEL]
    else:
        x = ai.data.read_xy_file(PATH_XY_FILE)
        if len(x.shape) > 1:
            x = x[x[:, 0] == x[:, 1], 0]
    print('Nmbr bl: ', x.shape)
    x = x[x < CUT_PH_VALUE]
    print('Nmbr bl after ph cut: ', x.shape)
    x *= 1000  # now in mV

    # make histogram of events heights
    counts_hist, bins_hist = np.histogram(x, bins=BINS, range=(LOWER_LIMIT, UPPER_LIMIT), density=True)

    # fit all models in a loop
    for model in MODELS:
        print('-------------------------------------------------')
        print('Working on {} model.'.format(model))

        only_histogram = False

        pars = ai.fit.get_noise_parameters_unbinned(events=x,
                                                    model=model,
                                                    sigma_x0=SIGMA_X0,
                                                    )

        try:

            x_grid, \
            trigger_window, \
            ph_distribution, \
            polluted_ph_distribution, \
            noise_trigger_rate, \
            polluted_trigger_rate, \
            threshold, \
            nmbr_pollution_triggers = ai.fit.calc_threshold(record_length=RECORD_LENGTH,
                                                            sample_length=SAMPLE_LENGTH,
                                                            detector_mass=DETECTOR_MASS,
                                                            interval_restriction=INTERVAL_RESTRICTION,
                                                            ul=UPPER_LIMIT,
                                                            ll=LOWER_LIMIT,
                                                            model=model,
                                                            pars=pars,
                                                            allowed_noise_triggers=ALLOWED_NOISE_TRIGGERS)

        except IndexError as err:
            print('The threshold is above the upper limit, plotting only the counts and fit.')
            x_grid, \
            trigger_window, \
            ph_distribution, \
            polluted_ph_distribution, \
            noise_trigger_rate, \
            polluted_trigger_rate, \
            threshold, \
            nmbr_pollution_triggers = err.args[1]
            only_histogram = True

        finally:

            if PATH_SAVE is not None:
                save_path = PATH_SAVE + '_' + model
                if Y_LOG_SCALE:
                    save_path += '_log'

            ai.fit.plot_noise_trigger_model(bins_hist=bins_hist,
                                            counts_hist=counts_hist,
                                            x_grid=x_grid,
                                            trigger_window=trigger_window,
                                            ph_distribution=ph_distribution,
                                            model=model,
                                            polluted_ph_distribution=polluted_ph_distribution,
                                            title=model,
                                            xran_hist=None,
                                            noise_trigger_rate=noise_trigger_rate,
                                            polluted_trigger_rate=polluted_trigger_rate,
                                            threshold=threshold,
                                            yran=PLOT_RANGE_Y,
                                            allowed_noise_triggers=ALLOWED_NOISE_TRIGGERS,
                                            nmbr_pollution_triggers=nmbr_pollution_triggers,
                                            xran=PLOT_RANGE_X,
                                            ylog=Y_LOG_SCALE,
                                            only_histogram=only_histogram,
                                            save_path=save_path,
                                            )
