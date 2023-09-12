from setuptools import find_packages, setup

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='cait',
    version='1.2.0',
    author='Philipp Schreiner, '
           'Felix Wagner',
    author_email="felix.wagner@oeaw.ac.at",
    description='Cryogenic Artificial Intelligence Tools - A Python Package for the Data Analysis of Rare Event Search Experiments with Machine Learning.',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/fewagner/cait",
    license='GPLv3',
    packages=find_packages(include=['cait', 'cait.*']),
    install_requires=[
        'numpy<1.25',
        'scipy>=1.6',
        'pandas>=1.1',
        'h5py>=3.2',
        'uproot>=4.1',
        'ipywidgets>=7.5',
        'dash>=2.0',
        'matplotlib>=3.4',
        'numba>=0.54',
        'tqdm>=4.62',
        'scikit-learn>=0.24'
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
    python_requires='>=3.8, <3.11',
)
