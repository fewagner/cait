import pytest
import numpy as np

import cait as ai
from cait.versatile import Stream, RDTFile, MockData, apply
from cait.versatile.iterators.impl_h5 import H5Iterator
from cait.versatile.iterators.impl_rdt import RDTIterator
from cait.versatile.iterators.impl_stream import StreamIterator
from cait.versatile.iterators.impl_mock import MockIterator

from ..fixtures import datahandler, tempdir, testdata_1D_2D_3D_s_mus, RDT_LENGTH, RECORD_LENGTH, SAMPLE_FREQUENCY

DATA_2D_single_CH = np.random.rand(1, 100)
DATA_3D_single_CH = np.random.rand(1, 100, 16)

@pytest.fixture(scope="module")
def dh(datahandler, testdata_1D_2D_3D_s_mus):
    d1, d2, d3, s, mus = testdata_1D_2D_3D_s_mus
    datahandler.set("events", event=d3, event_single_Ch=DATA_3D_single_CH, 
                             hours=d1, pulse_height=d2, pulse_height_single_Ch=DATA_2D_single_CH)
    datahandler.set("events", time_s=s, time_mus=mus, dtype=np.int32)
    yield datahandler

@pytest.fixture(scope="module")
def testdata(tempdir):
    data = ai.data.TestData(filepath=tempdir.name+'/mock_001',
                            duration=RDT_LENGTH,
                            record_length=RECORD_LENGTH,
                            sample_frequency=SAMPLE_FREQUENCY,
                            channels=[0, 1],
                            start_s=13)
    data.generate()
    stream = Stream('cresst', [tempdir.name+'/mock_001_Ch0.csmpl',
                               tempdir.name+'/mock_001_Ch1.csmpl',
                               tempdir.name+'/mock_001.test_stamps',
                               tempdir.name+'/mock_001.dig_stamps',
                               tempdir.name+'/mock_001.par'])
    rdt_file = RDTFile(tempdir.name+'/mock_001.rdt')

    yield stream, rdt_file

# Used repeatedly for all iterators
def basic_checks(it):
    # Copy to not alter the original iterator
    it = it[:,:]
    # Test adding of processing
    it_new = it.with_processing(lambda x: x**2)
    it.add_processing(lambda x: x**2)

    # Test application of function
    arr1 = apply(lambda x: -x, it)
    arr2 = apply(lambda x: x, it)
    assert np.array_equal(arr1[:,...,0], -arr2[:,...,0])

    # Test timestamps
    ts = it.timestamps

    # Test record_length, record window and time base
    l = it.record_length
    tb = it.dt_us
    t = it.t

    # Test context manager
    S = 0
    with it as i:
        for k in i:
            S += np.max(k)
    
    with it[0] as i:
        for k in i:
            S += np.max(k)

    # Test flag slicing, int slicing and list slicing
    flag = np.zeros(len(it), dtype=bool)
    flag[:20] = True

    next(iter(it[:]))
    next(iter(it[0]))
    next(iter(it[0, :10]))
    next(iter(it[:]))
    next(iter(it[:, flag]))
    next(iter(it[0, 0]))
    next(iter(it[0, [0,1,4]]))
    
    # Test if slicing gives expected number of channels
    assert it[0].n_channels == 1
    assert it[:].n_channels == it.n_channels

    # Test if number of events is preserved when only indexing channels
    assert len(it[0]) == len(it)
    assert len(it[:]) == len(it)

    # Test if number returned by flag sliced iterator is correct
    assert len(it[:, flag]) == 20
    assert len(it[0, flag]) == 20

    # Check if events returned by sliced and unsliced iterator are identical
    assert np.array_equal(next(iter(it)), next(iter(it[:,flag])))
    assert np.array_equal(next(iter(it)), next(iter(it_new)))

    # Test if grab works as intended
    single = it.grab(0)
    single_last = it.grab(-1)
    selected_events = it.grab([0,2,4])

    assert single.shape == ((it.n_channels, it.record_length) if it.n_channels > 1 else (it.record_length,))
    assert selected_events.shape == ((3, it.n_channels, it.record_length) if it.n_channels > 1 else (3, it.record_length))

    assert it[0].grab(0).shape == (it.record_length,)
    assert it[0].grab([0,1,4]).shape == (3, it.record_length)

    # Add iterators and check if first and last event are as expected
    it_combined = it + it_new + it_new
    assert len(it_combined) == len(it)+2*len(it_new)
    assert np.array_equal(next(iter(it_combined)), next(iter(it)))
    assert np.array_equal(next(iter(it_combined[:,-1])), 
                                   next(iter(it_new[:,-1])))
    flag = np.zeros(len(it_combined), dtype=bool)
    flag[:20] = True

    next(iter(it_combined[:]))
    next(iter(it_combined[0]))
    next(iter(it_combined[0, :10:3]))
    next(iter(it_combined[:]))
    next(iter(it_combined[:, flag]))
    next(iter(it_combined[0, 0]))
    next(iter(it_combined[0, [0,1,4]]))

class TestH5Iterator:
    def test_iterator_bs1_ch2(self, dh):
        it = H5Iterator(dh, "events")
        for i in it:
            assert i.shape == (2, dh.record_length)

        with it as opened_it:
            for i in opened_it:
                assert i.shape == (2, dh.record_length)

    def test_iterator_bs10_ch2(self, dh):
        # Note that total length is a multiple of batch size. Remaining batches smaller than the batch size are 
        # checked in a different test
        it = H5Iterator(dh, "events", batch_size=10)
        for i in it:
            assert i.shape == (10, 2, dh.record_length)

        with it as opened_it:
            for i in opened_it:
                assert i.shape == (10, 2, dh.record_length)

    def test_iterator_bs1_ch1(self, dh):
        it = H5Iterator(dh, "events", channels=1)
        for i in it:
            assert i.shape == (dh.record_length,)

        with it as opened_it:
            for i in opened_it:
                assert i.shape == (dh.record_length,)

        # Same but with list instead of integer
        it = H5Iterator(dh, "events", channels=[1])
        for i in it:
            assert i.shape == (dh.record_length,)

        with it as opened_it:
            for i in opened_it:
                assert i.shape == (dh.record_length,)

    def test_iterator_bs10_ch1(self, dh):
        it = H5Iterator(dh, "events", channels=1, batch_size=10)
        for i in it:
            assert i.shape == (10, dh.record_length)

        with it as opened_it:
            for i in opened_it:
                assert i.shape == (10, dh.record_length)

        # Same but with list instead of integer
        it = H5Iterator(dh, "events", channels=[1], batch_size=10)
        for i in it:
            assert i.shape == (10, dh.record_length)

        with it as opened_it:
            for i in opened_it:
                assert i.shape == (10, dh.record_length)

    def test_inds(self, dh):
        inds = [12, 18, 55, 81, 90, 99]
        it1 = H5Iterator(dh, "events", inds=inds)
        it2 = H5Iterator(dh, "events", channels=1, inds=inds)
        it3 = H5Iterator(dh, "events", channels=1, batch_size=3, inds=inds)
        it4 = H5Iterator(dh, "events", batch_size=3, inds=inds)

        for it in [it1, it2, it3, it4]: assert len(it) == len(inds)

        for i in it1: assert i.shape == (2, dh.record_length)
        for i in it2: assert i.shape == (dh.record_length, )
        for i in it3: assert i.shape == (3, dh.record_length)
        for i in it4: assert i.shape == (3, 2, dh.record_length)

    def test_batch_remainder(self, dh):
        it = H5Iterator(dh, "events", channels=1, batch_size=11)
        lens_are = np.array([len(i) for i in it])
        lens_should_be = np.ones_like(lens_are, dtype=int)*11
        lens_should_be[-1] = 1 

        assert np.all(lens_are == lens_should_be)

    def test_add_processing(self, dh):
        it1 = H5Iterator(dh, "events", channels=1, batch_size=11)
        it2 = H5Iterator(dh, "events", batch_size=11)
        it3 = H5Iterator(dh, "events", channels=1)
        it4 = H5Iterator(dh, "events")

        it1.add_processing([lambda ev: ev**2, lambda ev: -np.sqrt(ev)])
        it2.add_processing([lambda ev: ev**2, lambda ev: -np.sqrt(ev)])
        it3.add_processing([lambda ev: ev**2, lambda ev: -np.sqrt(ev)])
        it4.add_processing([lambda ev: ev**2, lambda ev: -np.sqrt(ev)])

        # Test iteration
        next(iter(it1))
        next(iter(it2))
        next(iter(it3))
        next(iter(it4))

        # Check if outputs are equivalent (internally, batched/unbatched iterators
        # and iterators of different numbers of channels are handled differently)
        arr1 = apply(lambda x: x, it1) # Just uses apply method to resolve batches
        arr2 = apply(lambda x: x, it2)
        arr3 = apply(lambda x: x, it3)
        arr4 = apply(lambda x: x, it4)

        assert np.array_equal(arr1, arr3)
        assert np.array_equal(arr2, arr4)

    def test_basic(self, dh):
        it1 = H5Iterator(dh, "events", channels=1, batch_size=11)
        it2 = H5Iterator(dh, "events", batch_size=11)
        it3 = H5Iterator(dh, "events", channels=1)
        it4 = H5Iterator(dh, "events")
  
        basic_checks(it1)
        basic_checks(it2)
        basic_checks(it3)
        basic_checks(it4)

class TestStreamIterator:
    def test_basic(self, testdata):
        stream, *_ = testdata

        inds = stream.time.timestamp_to_ind(stream.tp_timestamps["0"])

        basic_checks(StreamIterator(stream=stream,
                                    keys="mock_001_Ch0",
                                    inds=inds[:25],
                                    record_length=2**13))
        basic_checks(StreamIterator(stream=stream,
                                    keys=["mock_001_Ch0","mock_001_Ch1"],
                                    inds=inds,
                                    record_length=2**14))
        basic_checks(StreamIterator(stream=stream,
                                    keys=["mock_001_Ch0","mock_001_Ch1"],
                                    inds=inds,
                                    record_length=2**15))
        basic_checks(StreamIterator(stream=stream,
                                    keys=["mock_001_Ch0","mock_001_Ch1"],
                                    inds=inds,
                                    record_length=2**15,
                                    alignment=1/2))
        
    def test_batches_singleCh(self, testdata):
        stream, *_ = testdata

        inds = stream.time.timestamp_to_ind(stream.tp_timestamps["0"])
        
        basic_checks(StreamIterator(stream=stream,
                                    keys="mock_001_Ch0",
                                    inds=inds[:25],
                                    record_length=2**13,
                                    batch_size=13))
        
        it = StreamIterator(stream=stream,
                            keys="mock_001_Ch0",
                            inds=inds[:25],
                            record_length=2**13,
                            batch_size=13)
        it2 = StreamIterator(stream=stream,
                            keys="mock_001_Ch0",
                            inds=inds[:25],
                            record_length=2**13)

        for n, i in enumerate(it):
            if n == 0: assert i.shape == (13, 2**13)
            elif n == 1: assert i.shape == (12, 2**13)

        assert np.array_equal(next(iter(it))[0], next(iter(it2)))

    def test_batches_multiCh(self, testdata):
        stream, *_ = testdata

        inds = stream.time.timestamp_to_ind(stream.tp_timestamps["0"])
        
        basic_checks(StreamIterator(stream=stream,
                                    keys=["mock_001_Ch0","mock_001_Ch1"],
                                    inds=inds[:25],
                                    record_length=2**13,
                                    batch_size=13))
        
        it = StreamIterator(stream=stream,
                            keys=["mock_001_Ch0","mock_001_Ch1"],
                            inds=inds[:25],
                            record_length=2**13,
                            batch_size=13)
        it2 = StreamIterator(stream=stream,
                            keys=["mock_001_Ch0","mock_001_Ch1"],
                            inds=inds[:25],
                            record_length=2**13)

        for n, i in enumerate(it):
            if n == 0: assert i.shape == (13, 2, 2**13)
            elif n == 1: assert i.shape == (12, 2, 2**13)

        assert np.array_equal(next(iter(it))[0], next(iter(it2)))

class TestRDTIterator:
    def test_basic(self, testdata):
        _, f, *_ = testdata
        basic_checks(RDTIterator(rdt_channel=f[0]))
        basic_checks(RDTIterator(rdt_channel=f[0], channels=0))
        basic_checks(RDTIterator(rdt_channel=f[(0,1)]))
        basic_checks(RDTIterator(rdt_channel=f[(0,1)], channels=(0,1)))
        basic_checks(RDTIterator(rdt_channel=f[1]))
        basic_checks(RDTIterator(rdt_channel=f[1], channels=1))

    def test_batches_singleCh(self, testdata):
        _, f, *_ = testdata

        basic_checks(RDTIterator(rdt_channel=f[0], batch_size=13))
        
        it = RDTIterator(rdt_channel=f[0], batch_size=13)
        it2 = RDTIterator(rdt_channel=f[0])

        for n, i in enumerate(it):
            if n < it.n_batches-1: 
                assert i.shape == (13, it.record_length)
            elif n == it.n_batches-1: 
                assert i.shape == (len(it)%13, it.record_length)

        assert np.array_equal(next(iter(it))[0], next(iter(it2)))

    def test_batches_multiCh(self, testdata):
        _, f, *_ = testdata

        basic_checks(RDTIterator(rdt_channel=f[(0,1)], batch_size=13))
        
        it = RDTIterator(rdt_channel=f[(0,1)], batch_size=13)
        it2 = RDTIterator(rdt_channel=f[(0,1)])

        for n, i in enumerate(it):
            if n < it.n_batches-1: 
                assert i.shape == (13, 2, it.record_length)
            elif n == it.n_batches-1: 
                assert i.shape == (len(it)%13, 2, it.record_length)

        assert np.array_equal(next(iter(it))[0], next(iter(it2)))
        
class TestIteratorCollection:
    def test_basic(self, dh, testdata):
        _, f, *_ = testdata
        it1 = dh.get_event_iterator("events")
        it2 = f[(0,1)].get_event_iterator()
        it3 = dh.get_event_iterator("events", batch_size=13)

        basic_checks(it1 + it2)
        basic_checks(it1[0] + it2[0])
        basic_checks(it1[0] + it2[0] + it1[0])

        with pytest.raises(ValueError): it1 + it2[0]
        
        with pytest.raises(ValueError): it1 + it3

class TestMockIterator:
    def test_basic(self):
        mock = MockData()

        basic_checks(MockIterator(mock=mock))
        basic_checks(MockIterator(mock=mock, channels=0))
        basic_checks(MockIterator(mock=mock, channels=(0,1)))

    def test_batches_singleCh(self):
        mock = MockData()

        basic_checks(MockIterator(mock=mock, batch_size=13)[0])
        
        it = MockIterator(mock=mock, batch_size=13)[0]
        it2 = MockIterator(mock=mock)[0]

        for n, i in enumerate(it):
            if n < it.n_batches-1: 
                assert i.shape == (13, it.record_length)
            elif n == it.n_batches-1: 
                assert i.shape == (len(it)%13, it.record_length)

        assert np.array_equal(next(iter(it))[0], next(iter(it2)))

    def test_batches_multiCh(self):
        mock = MockData()

        basic_checks(MockIterator(mock=mock, batch_size=13))
        
        it = MockIterator(mock=mock, batch_size=13)
        it2 = MockIterator(mock=mock)

        for n, i in enumerate(it):
            if n < it.n_batches-1: 
                assert i.shape == (13, 2, it.record_length)
            elif n == it.n_batches-1: 
                assert i.shape == (len(it)%13, 2, it.record_length)

        assert np.array_equal(next(iter(it))[0], next(iter(it2)))