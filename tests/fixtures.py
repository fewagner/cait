import pytest
import tempfile
import numpy as np

import cait as ai

RECORD_LENGTH = 2**15
SAMPLE_FREQUENCY = 2e5

RDT_LENGTH = 100

@pytest.fixture(scope="module")
def tempdir():
    d = tempfile.TemporaryDirectory()
    yield d
    d.cleanup()

@pytest.fixture(scope="module")
def datahandler(tempdir):
    dh = ai.DataHandler(record_length=RECORD_LENGTH, 
			   			sample_frequency=SAMPLE_FREQUENCY,
						nmbr_channels=2)
    dh.set_filepath(path_h5=tempdir.name, fname="test_file", appendix=False)
    dh.init_empty()

    return dh

@pytest.fixture(scope="module")
def testdata_1D_2D_3D_s_mus():
    data_1d = np.random.rand(100)
    data_2d = np.random.rand(2, 100)
    data_3d = np.random.rand(2, 100, RECORD_LENGTH)

    data_s = np.random.randint(0, 1000000, size=100, dtype=np.int32)
    data_mus = np.random.randint(0, 1000000, size=100, dtype=np.int32)

    return data_1d, data_2d, data_3d, data_s, data_mus