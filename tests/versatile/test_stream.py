import pytest

import numpy as np
import cait as ai
import cait.versatile as vai

from ..fixtures import tempdir

LENGTH = 100

@pytest.fixture(scope="module")
def stream_cresst(tempdir):
    data = ai.data.TestData(filepath=tempdir.name+'/mock_001',
                            duration=LENGTH,
                            channels=[0, 1],
                            sample_frequency=25000,
                            start_s=13,
                            tpas=[20, -1.0, 20, 0.3, 0, 20, 0.5, 1, 20, 3, -1, 20, 0, 10])
    data.generate()

    yield vai.Stream('cresst', [tempdir.name+'/mock_001_Ch0.csmpl',
                                   tempdir.name+'/mock_001_Ch1.csmpl',
                                   tempdir.name+'/mock_001.test_stamps',
                                   tempdir.name+'/mock_001.dig_stamps',
                                   tempdir.name+'/mock_001.par'])

def basic_checks(stream):
    # Basic indexing
    k = stream.keys[0]
    stream[k]
    stream[k, :10]
    stream[k, -1]
    stream[k, 10:20, 'as_voltage']

    # Methods
    len(stream)
    stream.get_channel(k)
    stream.get_voltage_trace(k, slice(10, 30))
    stream.start_us
    stream.dt_us
    stream.time
    it1 = stream.get_event_iterator(k, 100, inds=[10,20,30])
    it2 = stream.get_event_iterator(k, 100, timestamps=stream.time[[10,20,30]])
    assert np.array_equal(next(iter(it1)), next(iter(it2)))

def test_cresst_StreamTime(stream_cresst):
    t = stream_cresst.time
    t[0]
    t[-1]
    t[0:10]
    t[[1,3,10]]
    ts = t[1000]
    dt = t.timestamp_to_datetime(ts)
    assert t.timestamp_to_ind(ts) == 1000
    assert t.datetime_to_timestamp(dt) == ts

def test_cresst_basic(tempdir, stream_cresst): # stream_cresst needed for file initialization
    s1 = vai.Stream('cresst', [tempdir.name+'/mock_001_Ch0.csmpl',
                               tempdir.name+'/mock_001.par'])
    
    s2 = vai.Stream('cresst', [tempdir.name+'/mock_001_Ch0.csmpl',
                               tempdir.name+'/mock_001_Ch1.csmpl',
                               tempdir.name+'/mock_001.test_stamps',
                               tempdir.name+'/mock_001.dig_stamps',
                               tempdir.name+'/mock_001.par'])
    
    # Not available if not provided via test_stamps
    with pytest.raises(KeyError): s1.tpas
    with pytest.raises(KeyError): s1.tp_timestamps

    s2.tpas
    s2.tp_timestamps
    it = s2.get_event_iterator("mock_001_Ch0", 2**13, timestamps=s2.tp_timestamps["0"])
    assert len(s2.tpas["0"]) == len(s2.tp_timestamps["0"])
    assert len(it) == len(s2.tp_timestamps["0"])
    
    basic_checks(s1)
    basic_checks(s2)

# TODO
def test_VDAQ2():
    ...

# TODO
def test_VDAQ3():
    ...

# TODO: test either on "real" data or figure out how cait simulates stream files
# to validate the trigger indices
def test_trigger(stream_cresst):
    i, v = vai.trigger_zscore(stream_cresst, "mock_001_Ch0", 2**14)
    i, v = vai.trigger_zscore(stream_cresst, "mock_001_Ch0", 2**14, n_triggers=10)
    i, v = vai.trigger_zscore(stream_cresst, "mock_001_Ch1", 2**14, apply_first=lambda x: -x)