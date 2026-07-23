import tempfile
import sys
import termios

import numpy as np
import pytest

import cait as ai

RECORD_LENGTH = 2**15
SAMPLE_FREQUENCY = 2e5

RDT_LENGTH = 100

# Sets up a multi-purpose temporary directory and cleans it up after each module
@pytest.fixture(scope="module")
def tempdir():
    d = tempfile.TemporaryDirectory()
    yield d
    d.cleanup()

@pytest.fixture(scope="function")
def tempdir_fnc():
    d = tempfile.TemporaryDirectory()
    yield d
    d.cleanup()

# Sets up an empty DataHandler instance
@pytest.fixture(scope="module")
def datahandler(tempdir):
    dh = ai.DataHandler(record_length=RECORD_LENGTH,
			   			sample_frequency=SAMPLE_FREQUENCY,
						nmbr_channels=2)
    dh.set_filepath(path_h5=tempdir.name, fname="test_file", appendix=False)
    dh.init_empty()

    return dh

# Sets up a DataHandler instance already including test data
@pytest.fixture(scope="module")
def datahandler_testdata(tempdir):
    ai.data.TestData(filepath=tempdir.name+'/mock_001', duration=1000).generate()
    dh = ai.DataHandler(channels=[0,1])
    dh.convert_dataset(path_rdt=tempdir.name, fname='/mock_001', path_h5=tempdir.name)
    dh.set_filepath(path_h5=tempdir.name, fname='mock_001')

    yield dh

# Provides random 1D, 2D and 3D data as well as s and mus timestamps
@pytest.fixture(scope="module")
def testdata_1D_2D_3D_s_mus():
    data_1d = np.random.rand(100)
    data_2d = np.random.rand(2, 100)
    data_3d = np.random.rand(2, 100, RECORD_LENGTH)

    data_s = np.random.randint(0, 1000000, size=100, dtype=np.int32)
    data_mus = np.random.randint(0, 1000000, size=100, dtype=np.int32)

    return data_1d, data_2d, data_3d, data_s, data_mus

# Provides random 1D, 2D and 3D data (converted from int16) as well as s and mus timestamps.
# This is in principle the same as testdata_1D_2D_3D_s_mus (and should be interchangeable),
# but used for testing iterators saved as and loaded from int16, and therefore the "raw" data
# needs to be drawn from int16 as well.
@pytest.fixture(scope="module")
def testdata_1D_2D_3D_s_mus_int16():
    data_1d = ai.data.convert_to_V((np.random.rand(100) * 32768).astype(np.int16))
    data_2d = ai.data.convert_to_V((np.random.rand(2, 100) * 32768).astype(np.int16))
    data_3d = ai.data.convert_to_V((np.random.rand(2, 100, RECORD_LENGTH) * 32768).astype(np.int16))

    data_s = np.random.randint(0, 1000000, size=100, dtype=np.int32)
    data_mus = np.random.randint(0, 1000000, size=100, dtype=np.int32)

    return data_1d, data_2d, data_3d, data_s, data_mus


# Spoof standard input.  Meant to be used for tests with Uniplot.
# If used as an argument to a test, the yielded object can be
# pushed onto to set the next characters read from the spoofed stdin.  If used with
# @pytest.mark.usefixtures (or if used normally, and the buffer is empty when read),
# then this will always return 'q', the key we use to kill a uniplot instance.
# Inspired by tests in readchar: https://github.com/magmax/python-readchar/blob/master/tests/linux/conftest.py
@pytest.fixture(scope='class')
def spoof_stdin():
    class SpoofedStdIn:
        buffer = []

        def read(self, n):
            if not len(self.buffer):
                # For use with our Uniplot implementation, if we try to read from the buffer
                # while it's empty, just exit
                self.buffer.append('q')
            return "".join([self.buffer.pop(x) for x in range(n)])

        def push(self, char):
            [self.buffer.append(x) for x in char]

        def fileno(self):
            return 0

    def spoof_tcgetattr(fd):
        return [0, 0, 0, 0, None, None, None]

    def spoof_tcsetattr(fd, TCSADRAIN, old_settings):
        return None

    spoof = SpoofedStdIn()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys.stdin, "read", spoof.read)
        mp.setattr(sys.stdin, "fileno", spoof.fileno)
        mp.setattr(termios, "tcgetattr", spoof_tcgetattr)
        mp.setattr(termios, "tcsetattr", spoof_tcsetattr)
        yield spoof
