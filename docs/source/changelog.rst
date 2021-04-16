***********************
Features and Changelog
***********************

On this page we assemble the features of current Cait realeases and changes from the past versions.

v1.0.0
======

This is the first stable, full release of Cait. In this original version, the following features are included:

- Data access:
    - Conversion of raw data file formats to structured HDF5 files.
    - Conversion of Root files to HDF5 files.
- Calculation of features:
    - Main parameters
    - Standard events
    - Noise power spectra
    - Optimum filter
    - Standard event and parametric fits
    - Principal components
    - Baseline fits
- Processing of continuously recorded raw data:
    - Stream triggering with and without optimum filtering
    - Synchronisation with hardware triggered data
    - Simulation of random triggers on the continuous data stream
- Raw data analysis tools:
    - Logical cuts
    - Estimation of trigger thresholds
    - Rate and stability cuts
    - Energy calibration
    - Calculation of baseline resolution
    - Calculation of total exposure
- Simulation tools:
    - Simulation of raw data with particle, test pulse or exceptional standard event templates in the linear and saturated regime
    - Simulation of realistic noise baselines
    - Simulation of test data in various raw data formats
- Machine learning:
    - Data module compatible with Pytorch Lightning
    - Evaluation environment compatible with Scikit-Learn
- High level analysis:
    - Fit of recoil bands in the energy-light plane
    - Calculation of dark matter exclusion limits with Yellins Maximum Gap method