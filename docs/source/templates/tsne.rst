**********************
Interactive t-SNE Plot
**********************

.. code:: python

    """
    tsne.py

    This script produces an interactive t-SNE plot of the events inside an HDF5 file. No changes to the file are
    required, only command line argument. Run the file with the -h flag, to get a description of all available arguments.

    """

    # ---------------------------------------------
    # Imports
    # ---------------------------------------------

    import cait as ai
    import os.path
    from os import path
    import argparse
    import numpy as np
    import matplotlib as mpl

    # ---------------------------------------------
    # Read command line arguments
    # ---------------------------------------------

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-f", "--file_name", type=str, required=True, help="The name of the HDF5 file.")
    ap.add_argument("-c", "--channel", type=int, required=True, help="The channel number within the HDF5 file.")
    ap.add_argument("-t", "--time_series", action='store_true', help="Use the raw time series as data instead of the main parameters.")
    ap.add_argument("-a", "--additional_mainpar", action='store_true', help="Use the additional main parameters as data instead of the main parameters.")
    ap.add_argument("-s", "--test_size", type=float, required=False, default=0.5, help="The share of events used as test set.")
    ap.add_argument("-i", "--highest_idx", type=int, required=False, default=None, help="The highest index that is included in the plot. Use this if there are too many events to handle for the scatter plot.")
    ap.add_argument("-p", "--perplexity", type=int, required=False, default=30, help="The perplexity value, hyperparameter of the t-SNE plot. Recommended value: Between 5 and 50.")
    ap.add_argument("-m", "--matplotlib", action='store_true', help="Start matplotlib for plotting, instead plotly.")
    args = vars(ap.parse_args())

    assert args['time_series'] is False or args['additional_mainpar'] is False, "You cannot activate both time_series and additional_mainpar!"
    assert args['test_size'] > 0 and args['test_size'] < 1, "The test_size must be between 0 and 1!"
    assert path.exists(args['file_name']), f"The file {args['file_name']} does not exist!"
    assert args['channel'] >= 0, 'You cannot choose a negative channel number!'
    if args['highest_idx'] is not None:
        assert args['highest_idx'] >= 1, 'The highest index must be 1 or higher!'
    assert args['perplexity'] > 1, 'The perplexipy must at least be 1. Recommended value: Between 5 and 50.'

    # ---------------------------------------------
    # Main
    # ---------------------------------------------

    if args['time_series']:
        which_data = 'time_series'
    elif args['additional_mainpar']:
        which_data = 'additional_mainpar'
    else:
        which_data = 'mainpar'

    if args['highest_idx'] is not None:
        only_idx = np.arange(args['highest_idx'])
    else:
        only_idx = None

    et = ai.EvaluationTools()
    et.add_events_from_file(file=args['file_name'],
                        channel=args['channel'],
                        which_data=which_data,
                        only_idx=only_idx,
                        )
    et.split_test_train(test_size=float(args['test_size']))

    if args['matplotlib']:
        et.plt_pred_with_tsne(pred_methods=[],
                              what='all',
                              verb=True,
                              perplexity=args['perplexity'],
                             )
    else:
        et.plt_pred_with_tsne_plotly(pred_methods=[],
                      what='all',
                      verb=True,
                      perplexity=args['perplexity'],
                     )