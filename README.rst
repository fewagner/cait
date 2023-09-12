.. -*- mode: rst -*-

|PyPi|_ |DOI|_

.. |PyPi| image:: https://badge.fury.io/py/cait.svg
.. _PyPi: https://badge.fury.io/py/cait

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5091416.svg
.. _DOI: https://doi.org/10.5281/zenodo.5091416

.. image:: https://raw.githubusercontent.com/fewagner/cait/master/docs/source/logo/logo_wide.png
  :width: 256
  :target: https://github.com/fewagner/cait

**Cait** (Cryogenic Artificial Intelligence Tools) is a Python 3 software package for the machine learning based analysis
of raw data from cryogenic dark matter experiments. It is tailored to the needs of the CRESST and COSINUS experiment,
but applicable to other, similar data structures.

**Documentation:** https://cait.readthedocs.io/

**Source Code:** https://github.com/fewagner/cait

**Bug Report:** https://github.com/fewagner/cait/issues


Installation
============

Cait is hosted on the Python package index.

.. code:: console

    $ pip install cait

You can now import the library in Python, e.g.

.. code:: python

    import cait as ai

**Important Note for JupyterHub on computing clusters:**
In the past, many users experienced issues with our interactive plotting tools which are based on plotly and ipywidgets. These problems were due to version mismatches between the plotly/ipywidgets packages and their corresponding JupyterLab extensions (which are automatically installed alongside the packages). 

To not run into such issues in the first place, we recommend one of the following approaches:
- Install cait in the base environment of your computing cluster's JupyterLab. 
- Install it in a virtual environment (because you want or have to) is also possible but you will have to make sure that the same plotly/ipywidgets versions (which are installed as dependencies of cait) are also present in the base environment. A good practice to ensure this is to always pip-upgrade plotly/ipywidgets to the latest version in both environments. Note that you will potentially have to match these versions every time you upgrade cait.

Lastly, remember to **restart JupyterHub completely** (not just the kernel) for the changes to take effect.

Version History
===============

Master branch is on the latest release version and stable.

Previous versions are hosted on the accordingly named Git branch.

The Changelog starts with Version 1.0.

Version numbers follow the segmantic versioning guide (https://semver.org/).

Citations
===============

If you use Cait in your research work, please reference the package accordingly.

Cait uses a number of Python packages. If you use methods that are based on those packages, please consider
referencing them: h5py, numpy, matplotlib, scipy, numba, sklearn, uproot, torch, pytorch-lightning, plotly.

Cait has methods implemented that were used in prior research work. Please consider
referencing them:

- 2020, F. Wagner, Machine Learning Methods for the Raw Data Analysis of cryogenic Dark Matter Experiments", https://doi.org/10.34726/hss.2020.77322 (accessed on the 9.7.2021)
- 2019, D. Schmiedmayer, Calculation of dark-matter exclusions-limits using a maximum Likelihood approach, https://repositum.tuwien.at/handle/20.500.12708/5351 (accessed on the 9.7.2021)
- 2019, CRESST Collaboration et. al., First results from the CRESST-III low-mass dark matter program, doi 10.1103/PhysRevD.100.102002
- 2020, M. Stahlberg, Probing low-mass dark matter with CRESST-III : data analysis and first results, available via https://doi.org/10.34726/hss.2021.45935 (accessed on the 9.7.2021)
- 2019, M. Mancuso et. al., A method to define the energy threshold depending on noise level for rare event searches" (arXiv:1711.11459)
- 2018, N. Ferreiro Iachellini, Increasing the sensitivity to low mass dark matter in cresst-iii witha new daq and signal processing, doi 10.5282/edoc.23762
- 2016, F. Reindl, Exploring Light Dark Matter With CRESST-II Low-Threshold Detectors", available via http://mediatum.ub.tum.de/?id=1294132 (accessed on the 9.7.2021)
- 1995, F. Pröbst et. al., Model for cryogenic particle detectors with superconducting phase transition thermometers, doi 10.1007/BF00753837

We want you ...
===============

... to contribute! We are always happy about any contributions to our software. To coordinate
efforts, please get in touch with felix.wagner(at)oeaw.ac.at such that we can include your
features in the upcoming release. If you have any troubles with the current release, please open an issue in the Bug Report.