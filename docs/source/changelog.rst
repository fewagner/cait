***********************
Features and Changelog
***********************

On this page we assemble the features of current Cait realeases and changes from the past versions.

v1.1.0
======

There were significant chages since v1.0, partially affecting the user interface.

New features:

- Viztool: The VizTool is a new, interactive Interface to visualize events and their properties and to do interactive
    cuts. For many standard situations, this is the new recommended method to define cuts for standard events, baseline
    resolutions and noise power spectra.

- Augmentation suite: A class with functionality to augment pulse-shaped events and a wide range of artifacts. This is
    mainly to augment training data for machine learning methods and to test cuts. Several models were trained with
    this data and are deliverey pre-trained with the library.

- Ressources: A folder to store pre-trained models. Two pre-trained models are delivered with the packe.

- VDAQ functionalities: Methods to include events from VDAQ2-written *.bin files. A trigger method is not included,
    for this we recommend the use of external repositories, to write the time stamps.

New methods to calculate properties of events:

- Array fit: A robust implementation to fit numerical array (e.g. the numerical SEV) to events, truncation works.

- Correlated pulse height: Calculate the pulse height with 50 sample moving average, with a dominant channel. The height
    in the other channels is evaluated at the peak position of the first channel.

- CNN model: A CNN lightning module for event classification.

- Separation LSTM: An LSTM lightning module to separate pile-up events.

Updates on existing features:

- Memsafe SEV: Data is not loaded into memory anymore. This is activated per default and introduces some changes in
    the available arguments. However, a fallback is possible, by deactivating the option.

- Energy calibration with interpolation: A method to use interpolation instead of a polynomial fit is now possible.

- Maximum shift for OF correlated evaluation: For the corralated evaluation, it is possible that the maxima in
    different channels are a different positions. For this scenario, there is now a shift argument.

- Merge HDF5 for scalars: The merge does also work for datasets that are single scalar values.

- Triggering without SQL database: For triggering of csmpl files, the start of files can now be read from the metainfo,
    instead the SQL database.

New utilities:

- Shrink HDF5: Create a new HDF5 set, which excludes event from the former one, based on a cut flag.

- Metainfo: Include the information stored in PAR files in the HDF5 group metainfo.

New documentation:

- Tutorial notebooks: New tutorial notebooks for data augmentation and pile-up separation.


v1.0.0
======

This is the first stable, full release of Cait. In this original version, the following features are included:

- Data access:
    - Conversion of raw data file formats to structured HDF5 files.
    - Conversion of Root files to HDF5 files.
    - Im- and Export of arbitrary feature values, standard events, filters, noise power spectra to and from *.xy files
    - Import of trigger time stamps from *.trip files
- Calculation of features:
    - Main parameters
    - Standard events
    - Noise power spectra
    - Optimum filter
    - Standard event and parametric fits
    - Principal components
    - Baseline fits
- Processing of continuously recorded raw data:
    - Stream (*.csmpl) triggering with or without optimum filtering.
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