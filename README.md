# cait

Cait (abbreviation for "Cryogenic Artificial Intelligence Tools") is a Python 3 software package for the machine learning based analysis
of raw data from cryogenic dark matter experiments. In the current version, it is tailored to the needs of the CRESST
experiment, but also suitable for the COSINUS and NUCLEUS experiment.

- **Documentation:** https://git.cryocluster.org/fwagner/cait/-/wikis/0.-Getting-Started
- **Source Code:** https://git.cryocluster.org/fwagner/cait
- **Bug Report:** https://git.cryocluster.org/fwagner/cait/-/issues

### Installation

For installing Cait, first install and upgrade the following helper libraries:

> pip install -U wheel setuptools twine

Then clone the Git repository. The folder of the repository contains a wheel file:

> dist/*.whl

If there are multiple wheel files, choose the one with the highes version number. 
For installation of the library, run:

> pip install /path/to/wheelfile.whl

You can now import the library in Python, e.g.

> import cait

> from cait import EventInterface

### Version History

Master brunch is on Version 0.1.0-alpha.

Version numbers follow the segmantic versioning guide (https://semver.org/).

### We want you ...

... to contribute! Cait is in a very early stage of development and we are happy about any contributions. To coordinate 
efforts, please get in touch with felix.wagner@oeaw.ac.at such that we can include your
features in the upcoming release. If you have any troubles with the current release, please open an issue in the Bug Report.