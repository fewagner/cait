Features and Changelog
======================

On this page we assemble the features of current Cait releases and changes from the past versions.

development
~~~~~~~~~~~

- Added dcache file reading support (dcap, WebDAV and XRootD protocol; recommended protocol: XRootD)
- Added pipeline to build containers for released versions and development branch (on CERN gitlab)
- Added `dh.trigger_coincidence` for CRESST doubleTES analysis
- Added short-cut notation for `dh.get`: Now you can slice a DataHandler object using `dh['<group>/<dataset>']`. Additionally, this syntax supports iPython's TAB-completion, i.e. you can start typing `dh[` and hit 'TAB' to preview a list of possible datasets.
- Moved functions to combine/merge hdf5 files to `cait.data.combine_h5` and `cait.data.merge_h5`. The previous implementation `ai.data.merge_h5_sets` has been deprecated.
- Minor improvements
- New features in `cait.versatile`

v.1.2.2
~~~~~~~
- Added support for python versions 3.11 and 3.12
- Added possibility to automatically calculate RMS when applying the optimum filter. The option can be toggled using the `calc_rms` keyword on the `dh.apply_of` method.
- Added tests for general workflow
- New (experimental) features in `cait.versatile`
- Documentation for `cait.versatile` in preparation

v1.2.1
~~~~~~
- SEV fit parameters of extended pulse shape models are now supported by `cait` analysis routines like `show_sev` and `simulate_pulses`. If the dataset `fitpar` in the `stdevent` group has length `6`, it is assumed to be of the form `(t0, An, At, tau_n, tau_in, tau_t)` and the regular `2`-component pulse shape model is used. If it is of length `2(k+1)` for `k=3,4,...`, an extended `k`-component model is used and the parameters are assumed to be in order `(t0, A1, A2, ..., Ak, tau_in, tau_2, ..., tau_k, tau_n)`. Notice, however, that there is currently no function which does the corresponding fit for you due to yet unresolved restrictions in the `cait` source code. Therefore, you have to perform the fit yourself, e.g. by using the model function `cait.fit._templates.pulse_template` (which supports extended models) and any fit routine of your choice. Afterwards, you can include the results in the HDF5 file using `DataHandler.set()`.
- Fixed deprecation issue due to missing `seaborn-paper` style in matplotlib version 3.8.0
- Fixed a bug where the `exclude_tpas` keyword in `PulserModel` and derived functionalities like `calc_calibration` would not correctly exclude small testpulse amplitudes.
- Fixed a minor bug in `PulserModel`'s standard-deviation-estimation.
- Fixed a bug where the voltage trace view in `vizTool` would incorrectly subtract baselines for very short record lengths.
- Fixed a bug where creating an optimum filter with downsampling factor larger than 1 did not work for single-channel-data.
- New (experimental) features in `cait.versatile`.

v1.2.0
~~~~~~

This release aims to make cait slightly more light-weight by dropping some deprecated/unnecessary dependencies. Most importantly, the torch and torch_lightning packages are not automatically installed anymore (if you use a function that requires those packages, you will be prompted to install them). The reason being that few users need those functionalities yet their installation size is comparatively large.
Furthermore, we do not fix versions for visualization packages (plotly/ipywidgets) anymore because we figured out the reason for version mismatch issues (see README).

A number of improvements were implemented for the DataHandler and vizTool classes:

**DataHandler**

- print(dh) now prints useful information about the DataHandler
- dh.content() has been reformatted to be easier to read and with additional informations like dtype and whether the datasets are virtual or not (see below)
- dh.content() got an optional argument “group” in case you only want the contents of a specific group
- methods get_filepath, get_filedirectory and get_filename added to reduce ambiguity and improve maintainability
- dh.drop() can now drop entire groups
- dh.get() now supports indices for all datasets (not just 3d ones)
- dh.include_values() has been deprecated and replaced by dh.set() which is a more powerful and versatile way to include datasets and groups into the hdf5 file
- dh.rename() was added which can rename hdf5 groups and datasets
- dh.repackage() was added to repackage hdf5 file after it was changed to optimize file size (also arguments for appropriate functions were added)
- DataHandler now supports virtual datasets (see merging hdf5 files below)

**vizTool**

- The vizTool has been reworked and comes in a more concise layout. 
- When constructing the vizTool, you can directly hand it a DataHandler objects (you don't need to specify the hdf5 path anymore)
- the datasets argument can also include numpy arrays for quick inspection
- Heatmap functionality has been added (choose between scatter and density). Note that data selection is not possible in this mode due to a plotly limitation.
- a third dropdown menu was included which lets you choose a color scale for the scatter plot (essentially, this replaces color_flag)
- Histograms are now included in the main plot
- preview for event shape was included upon click
- the number of selected datapoints is displayed
- you can choose the plotly template for the vizTool now (e.g. light, dark, seaborn, …)

Additionally, this version introduces a new module called cait.versatile which as of now is still in its early development stages but will eventually provide a set of tools that let you individualize your workflow. Currently, the following functionality can be used (with caution as they might change in an upcoming release):

- use cait.versatile.file.combine to analyze multiple hdf5 files at once. This method makes use of virtual hdf5 datasets to create a file of links, i.e. no data is copied. From the point of view of the DataHandler, the hdf5 file looks as if it was a single file, though.
- cait.versatile.iterators can be used to iterate events in an hdf5 file or triggered events in a cait.versatile.stream file. Using cait.versatile.analysis.apply, you can apply any function to the events returned by the iterator.
- cait.versatile.plot contains classes for plotting data. The most exciting one is cait.versatile.plot.StreamViewer which can be used to inspect stream data (currently only VDAQ2 data)

**Bug Fixes**

- fixed a bug (that was likely to never occur) where specifying the moving average window when calculating main parameters would have no effect
- fixed a bug which occurred when triggering VDAQ2 data without timestamps

v1.1.0
~~~~~~

There were significant chages since v1.0, partially affecting the user interface.

New features:

- Viztool
    The VizTool is a new, interactive Interface to visualize events and their properties and to do interactive
    cuts. For many standard situations, this is the new recommended method to define cuts for standard events, baseline
    resolutions and noise power spectra.

- Augmentation suite
    A class with functionality to augment pulse-shaped events and a wide range of artifacts. This is
    mainly to augment training data for machine learning methods and to test cuts. Several models were trained with
    this data and are deliverey pre-trained with the library.

- Ressources
    A folder to store pre-trained models. Two pre-trained models are delivered with the packe.

- VDAQ functionalities
    Methods to include events from VDAQ2-written `*.bin` files. A trigger method is not included,
    for this we recommend the use of external repositories, to write the time stamps.

New methods to calculate properties of events:

- Array fit
    A robust implementation to fit numerical array (e.g. the numerical SEV) to events, truncation works.

- Correlated pulse height
    Calculate the pulse height with 50 sample moving average, with a dominant channel. The height
    in the other channels is evaluated at the peak position of the first channel.

- CNN model
    A CNN lightning module for event classification.

- Separation LSTM
    An LSTM lightning module to separate pile-up events.

Updates on existing features:

- Memsafe SEV
    Data is not loaded into memory anymore. This is activated per default and introduces some changes in
    the available arguments. However, a fallback is possible, by deactivating the option.

- Energy calibration with interpolation
    A method to use interpolation instead of a polynomial fit is now possible.

- Maximum shift for OF correlated evaluation
    For the corralated evaluation, it is possible that the maxima in
    different channels are a different positions. For this scenario, there is now a shift argument.

- Merge HDF5 for scalars
    The merge does also work for datasets that are single scalar values.

- Triggering without SQL database
    For triggering of csmpl files, the start of files can now be read from the metainfo,
    instead the SQL database.

New utilities:

- Shrink HDF5
    Create a new HDF5 set, which excludes event from the former one, based on a cut flag.

- Metainfo
    Include the information stored in PAR files in the HDF5 group metainfo.

New documentation:

- Tutorial notebooks
    New tutorial notebooks for data augmentation and pile-up separation.


v1.0.0
~~~~~~

This is the first stable, full release of Cait. In this original version, the following features are included:

- Data access:
    - Conversion of raw data file formats to structured HDF5 files.
    - Conversion of Root files to HDF5 files.
    - Im- and Export of arbitrary feature values, standard events, filters, noise power spectra to and from `*.xy` files
    - Import of trigger time stamps from `*.trip` files
- Calculation of features:
    - Main parameters
    - Standard events
    - Noise power spectra
    - Optimum filter
    - Standard event and parametric fits
    - Principal components
    - Baseline fits
- Processing of continuously recorded raw data:
    - Stream (`*.csmpl`) triggering with or without optimum filtering.
    - Synchronisation with hardware triggered data.
    - Simulation of random triggers on the continuous data stream.
- Raw data analysis tools:
    - Logical cuts
    - Estimation of trigger thresholds
    - Rate and stability cuts
    - Energy calibration
    - Calculation of baseline resolution
    - Calculation of total exposure
- Simulation tools:
    - Simulation of raw data with particle pulse, test pulses or exceptional standard event templates (e.g. carrier events) in the linear and saturated regime.
    - Simulation of realistic noise baselines.
    - Simulation of test data in various raw data formats.
- Machine learning:
    - Data module compatible with Pytorch Lightning.
    - Evaluation environment compatible with Scikit-Learn.
- High level analysis:
    - Fit of recoil bands in the energy-light plane.
    - Calculation of dark matter exclusion limits with Yellins Maximum Gap method.
- Various plotting routines.