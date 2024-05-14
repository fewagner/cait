import pytest
import numpy as np

import cait as ai
from cait.readers import TextFile, BinaryFile

from .fixtures import tempdir, RDT_LENGTH, RECORD_LENGTH, SAMPLE_FREQUENCY

@pytest.fixture(scope="module")
def testdata(tempdir):
    data = ai.data.TestData(filepath=tempdir.name+'/mock_001',
                            duration=RDT_LENGTH,
                            record_length=RECORD_LENGTH,
                            sample_frequency=SAMPLE_FREQUENCY,
                            channels=[0, 1],
                            start_s=13)
    data.generate()
    par_file = tempdir.name+'/mock_001.par'
    csmpl_file = tempdir.name+'/mock_001_Ch0.csmpl'
    rdt_file = tempdir.name+'/mock_001.rdt'

    yield par_file, csmpl_file, rdt_file

def test_batches_multiCh(testdata):
    par_file, csmpl_file, rdt_file = testdata

    # TEXT FILES
    with TextFile(par_file) as tf: 
        s1 = tf.read()
    with open(par_file, "r") as f:
        s2 = f.read()
    assert s1 == s2

    # BINARY FILES
    # test if elements are the same and still available outside of context
    bf1 = BinaryFile(csmpl_file, dtype=np.dtype(np.int16))[0]
    with BinaryFile(csmpl_file, dtype=np.dtype(np.int16)) as bf:
        bf2 = bf[0]
    assert np.array_equal(bf1, bf2)
    
    # test if offset works
    bf3 = BinaryFile(csmpl_file, dtype=np.dtype(np.int16))[-1]
    bf4 = BinaryFile(csmpl_file, dtype=np.dtype(np.int16), offset=2)[-1]
    assert np.array_equal(bf3, bf4)

    # test if count works
    bf5 = BinaryFile(csmpl_file, dtype=np.dtype(np.int16), count=13)
    assert len(bf5) == 13
    assert len(bf5[:]) == 13
    with bf5 as f: assert len(f[:]) == 13
