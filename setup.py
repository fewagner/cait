from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cait',
    version='0.1.0a0',
    author='Felix Wagner',
    author_email="felix.wagner@oeaw.ac.at",
    description='Cryogenic Artificial Intelligence Tools',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.cryocluster.org/fwagner/cait",
    packages=find_packages(include=['cait', 'cait.*']),
    install_requires=['h5py',
                      'pickle-mixin',
                      'numpy',
                      'matplotlib',
                      'tsfel',
                      'pandas',
                      'scipy',
                      'numba', ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Dark Matter Researchers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
