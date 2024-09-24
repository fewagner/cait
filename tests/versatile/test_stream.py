import pytest

import numpy as np
import cait as ai
import cait.versatile as vai

from ..fixtures import tempdir

LENGTH = 100

@pytest.fixture(scope="module")
def stream_csmpl(tempdir):
    data = ai.data.TestData(filepath=tempdir.name+'/mock_001',
                            duration=LENGTH,
                            channels=[0, 1],
                            sample_frequency=25000,
                            start_s=13,
                            tpas=[20, -1.0, 20, 0.3, 0, 20, 0.5, 1, 20, 3, -1, 20, 0, 10])
    data.generate()

    yield vai.Stream('csmpl', [tempdir.name+'/mock_001_Ch0.csmpl',
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
    stream.get_trace(k, slice(10, 30), voltage=True)
    stream.get_trace(k, slice(10, 30), voltage=False)
    stream.start_us
    stream.dt_us
    stream.time
    it1 = stream.get_event_iterator(k, 100, inds=[10,20,30])
    it2 = stream.get_event_iterator(k, 100, timestamps=stream.time[[10,20,30]])
    assert np.array_equal(next(iter(it1)), next(iter(it2)))

def test_csmpl_StreamTime(stream_csmpl):
    t = stream_csmpl.time
    t[0]
    t[-1]
    t[0:10]
    t[[1,3,10]]
    ts = t[1000]
    dt = t.timestamp_to_datetime(ts)
    assert t.timestamp_to_ind(ts) == 1000
    assert t.datetime_to_timestamp(dt) == ts

def test_csmpl_basic(tempdir, stream_csmpl): # stream_csmpl needed for file initialization
    s1 = vai.Stream('csmpl', [tempdir.name+'/mock_001_Ch0.csmpl',
                               tempdir.name+'/mock_001.par'])
    
    s2 = vai.Stream('csmpl', [tempdir.name+'/mock_001_Ch0.csmpl',
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
def test_trigger_basic(stream_csmpl):
    i, v = vai.trigger_zscore(stream_csmpl["Ch0"], 2**14)
    i, v = vai.trigger_zscore(stream_csmpl["Ch0"], 2**14, n_triggers=10)
    i, v = vai.trigger_zscore(stream_csmpl["Ch1"], 2**14, apply_first=lambda x: -x)

def test_trigger_single_samples():
    record_length = 2**15
    of = np.ones(record_length//2+1)

    for trigger in [lambda d: vai.trigger_zscore(d, record_length=record_length),
                    lambda d: vai.trigger_of(d, threshold=0.5, of=of)]:

        # should not find the peak (outside of search area)
        data = np.zeros(record_length*13)
        data[record_length-1] = 1
        i, _ = trigger(data)
        assert len(i)==0

        # should find the peak on first searched sample
        data = np.zeros(record_length*13)
        data[record_length] = 1
        i, _ = trigger(data)
        assert len(i)==1 and i[0]==record_length 

        # should not find the peak (outside of search area)
        data = np.zeros(record_length*13)
        data[-2*record_length] = 1
        i, _ = trigger(data)
        assert len(i)==0 

        # should find the peak on first sample (from back) that is searched
        data = np.zeros(record_length*13)
        data[-2*record_length-1] = 1
        i, _ = trigger(data)
        assert len(i)==1 and i[0]==(len(data) - 2*record_length - 1)

        # should find the peak on first searched sample but not the second one (within record_length/2)
        data = np.zeros(record_length*13)
        data[record_length] = 1
        data[record_length+record_length//2] = 1
        i, _ = trigger(data)
        assert len(i)==1

        # should find both peaks (outside record_length/2)
        data = np.zeros(record_length*13)
        data[record_length] = 1
        data[record_length+record_length//2+1] = 1
        i, _ = trigger(data)
        assert len(i)==2 and i[1]==(record_length+record_length//2+1)