from setuptools import find_packages, setup

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='cait',
    version='1.0.0',
    author='Daniel Bartolot, Jens Burkhart, Damir Rizvanovic, Daniel Schmiedmayer, Felix Wagner',
    author_email="felix.wagner@oeaw.ac.at",
    description='Cryogenic Artificial Intelligence Tools - A Python Package for the Data Analysis of Rare Event Search Experiments with Machine Learning.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.cryocluster.org/fwagner/cait",
    packages=find_packages(include=['cait', 'cait.*']),
    install_requires=['setuptools>=47.1.1',
                      'h5py',
                      'pickle-mixin',
                      'numpy',
                      'matplotlib',
                      'scipy',
                      'numba',
                      'sklearn',
                      'uproot',
                      'awkward1',
                      'torch',
                      'torchvision',
                      'pytorch-lightning',
                      'ipywidgets',
                      'tqdm'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
