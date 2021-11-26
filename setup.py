from setuptools import find_packages, setup

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='cait',
    version='1.1.0dev0',
    author='Daniel Bartolot, '
           'Jens Burkhart, '
           'Damir Rizvanovic, '
           'Daniel Schmiedmayer, '
           'Felix Wagner',
    author_email="felix.wagner@oeaw.ac.at",
    description='Cryogenic Artificial Intelligence Tools - A Python Package for the Data Analysis of Rare Event Search Experiments with Machine Learning.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fewagner/cait",
    license='GPLv3',
    packages=find_packages(include=['cait', 'cait.*']),
    install_requires=['setuptools>=47.1.1',
                      'h5py>=3.2.0',
                      'pickle-mixin>=1.0.2',
                      'numpy>=1.19.0',
                      'matplotlib>=3.3.3',
                      'scipy>=1.6.1',
                      'numba>=0.54.1',
                      'scikit-learn>=0.24.0',
                      'uproot>=4.1',
                      'awkward1>=1.0.0',
                      'torch>=1.8.1',
                      'torchvision>=0.8.1',
                      'pytorch-lightning>=1.2.6',
                      'ipywidgets>=7.5.1',
                      'tqdm>=4.62.3',
                      'pandas>=1.1.4',
                      'plotly>=5.3.1',
                      'IPython>=7.19.0',
                      'dash>=2.0.0',
                      'jupyter_dash>=0.4.0',
                      ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
