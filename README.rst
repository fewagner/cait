****
cait
****

Cait (Cryogenic Artificial Intelligence Tools) is a Python 3 software package for the machine learning based analysis
of raw data from cryogenic dark matter experiments. In the current version, it is tailored to the needs of the CRESST
experiment, but also suitable for the COSINUS and NUCLEUS experiment.

**Documentation:** https://git.cryocluster.org/fwagner/cait/-/wikis/0.-Getting-Started (depricated), preview of new version: https://admiring-booth-ce071b.netlify.app/

**Source Code:** https://git.cryocluster.org/fwagner/cait (for CRYOCLUSTER collaborations), sync on: https://github.com/fewagner/cait

**Bug Report:** https://git.cryocluster.org/fwagner/cait/-/issues (for CRYOCLUSTER collaborations), sync on: https://github.com/fewagner/cait/issues


Installation
============

For installing Cait, first install and upgrade the following helper libraries:

.. code:: console

    $ pip install -U wheel setuptools twine

Then clone the Git repository. The folder of the repository contains a wheel file:

.. code:: console

    $ dist/*.whl

If there are multiple wheel files, choose the one with the highes version number. 
For installation of the library, run:

.. code:: console

    $ pip install /path/to/wheelfile.whl

You can now import the library in Python, e.g.

.. code:: python

    import cait as ai
    from cait import EventInterface

Version History
===============

Master brunch is on Version 0.1.0-beta.

Version numbers follow the segmantic versioning guide (https://semver.org/).

We want you ...
===============

... to contribute! Cait is in a very early stage of development and we are happy about any contributions. To coordinate 
efforts, please get in touch with felix.wagner@oeaw.ac.at such that we can include your
features in the upcoming release. If you have any troubles with the current release, please open an issue in the Bug Report.