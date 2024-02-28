import unittest
import tempfile
import numpy as np

import cait as ai
from cait.versatile.stream import Stream
from cait.versatile.iterators import H5Iterator, RDTIterator, StreamIterator, MockIterator, apply
from cait.versatile import RDTFile, MockData

RECORD_LENGTH = 2**15
SAMPLE_FREQUENCY = 2e5

DATA_1D = np.random.rand(100)
DATA_2D = np.random.rand(2, 100)
DATA_3D = np.random.rand(2, 100, RECORD_LENGTH)

DATA_2D_single_CH = np.random.rand(1, 100)
DATA_3D_single_CH = np.random.rand(1, 100, 16)

DATA_s = np.random.randint(0, 1000000, size=100, dtype=np.int32)
DATA_mus = np.random.randint(0, 1000000, size=100, dtype=np.int32)

RDT_LENGTH = 100

# Used repeatedly for all iterators
def basic_checks(self, it):
    # Copy to not alter the original iterator
    it = it[:,:]
    # Test adding of processing
    it_new = it.with_processing(lambda x: x**2)
    it.add_processing(lambda x: x**2)

    # Test application of function
    arr1 = apply(lambda x: -x, it)
    arr2 = apply(lambda x: x, it)
    self.assertTrue(np.array_equal(arr1[:,...,0], -arr2[:,...,0]))

    # Test timestamps
    ts = it.timestamps

    # Test record_length
    l = it.record_length

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

    next(iter(it[:])),
    next(iter(it[0]))
    next(iter(it[0, :10]))
    next(iter(it[:]))
    next(iter(it[:, flag]))
    next(iter(it[0, 0]))
    next(iter(it[0, [0,1,4]]))
    
    # Test if slicing gives expected number of channels
    self.assertTrue(it[0].n_channels == 1)
    self.assertTrue(it[:].n_channels == it.n_channels)

    # Test if number of events is preserved when only indexing channels
    self.assertTrue(len(it[0]) == len(it))
    self.assertTrue(len(it[:]) == len(it))

    # Test if number returned by flag sliced iterator is correct
    self.assertTrue(len(it[:, flag]) == 20)
    self.assertTrue(len(it[0, flag]) == 20)

    # Check if events returned by sliced and unsliced iterator are identical
    self.assertTrue(np.array_equal(next(iter(it)), next(iter(it[:,flag]))))
    self.assertTrue(np.array_equal(next(iter(it)), next(iter(it_new))))

    # Add iterators and check if first and last event are as expected
    it_combined = it + it_new + it_new
    self.assertTrue(len(it_combined) == len(it)+2*len(it_new))
    self.assertTrue(np.array_equal(next(iter(it_combined)), next(iter(it))))
    self.assertTrue(np.array_equal(next(iter(it_combined[:,-1])), 
                                   next(iter(it_new[:,-1]))))
    flag = np.zeros(len(it_combined), dtype=bool)
    flag[:20] = True

    next(iter(it_combined[:]))
    next(iter(it_combined[0]))
    next(iter(it_combined[0, :10:3]))
    next(iter(it_combined[:]))
    next(iter(it_combined[:, flag]))
    next(iter(it_combined[0, 0]))
    next(iter(it_combined[0, [0,1,4]]))

class TestH5Iterator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.dh = ai.DataHandler(record_length=RECORD_LENGTH, 
			   					 sample_frequency=SAMPLE_FREQUENCY,
								 nmbr_channels=2)
		
        cls.dh.set_filepath(path_h5=cls.dir.name, fname="event_iterator_test_file", appendix=False)
        cls.dh.init_empty()

        cls.dh.set("events", event=DATA_3D, event_single_Ch=DATA_3D_single_CH, 
                             hours=DATA_1D, pulse_height=DATA_2D, pulse_height_single_Ch=DATA_2D_single_CH)
        cls.dh.set("events", time_s=DATA_s, time_mus=DATA_mus, dtype=np.int32)

    @classmethod
    def tearDownClass(cls):			
        cls.dir.cleanup()

    def test_iterator_bs1_ch2(self):
        it = H5Iterator(self.dh.get_filepath(), "events")
        for i in it:
            self.assertTrue(i.shape == (2, RECORD_LENGTH))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (2, RECORD_LENGTH))

    def test_iterator_bs10_ch2(self):
        # Note that total length is a multiple of batch size. Remaining batches smaller than the batch size are 
        # checked in a different test
        it = H5Iterator(self.dh.get_filepath(), "events", batch_size=10)
        for i in it:
            self.assertTrue(i.shape == (10, 2, RECORD_LENGTH))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (10, 2, RECORD_LENGTH))

    def test_iterator_bs1_ch1(self):
        it = H5Iterator(self.dh.get_filepath(), "events", channels=1)
        for i in it:
            self.assertTrue(i.shape == (RECORD_LENGTH,))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (RECORD_LENGTH,))

        # Same but with list instead of integer
        it = H5Iterator(self.dh.get_filepath(), "events", channels=[1])
        for i in it:
            self.assertTrue(i.shape == (RECORD_LENGTH,))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (RECORD_LENGTH,))

    def test_iterator_bs10_ch1(self):
        it = H5Iterator(self.dh.get_filepath(), "events", channels=1, batch_size=10)
        for i in it:
            self.assertTrue(i.shape == (10, RECORD_LENGTH))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (10, RECORD_LENGTH))

        # Same but with list instead of integer
        it = H5Iterator(self.dh.get_filepath(), "events", channels=[1], batch_size=10)
        for i in it:
            self.assertTrue(i.shape == (10, RECORD_LENGTH))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (10, RECORD_LENGTH))

    def test_inds(self):
        inds = [12, 18, 55, 81, 90, 99]
        it1 = H5Iterator(self.dh.get_filepath(), "events", inds=inds)
        it2 = H5Iterator(self.dh.get_filepath(), "events", channels=1, inds=inds)
        it3 = H5Iterator(self.dh.get_filepath(), "events", channels=1, batch_size=3, inds=inds)
        it4 = H5Iterator(self.dh.get_filepath(), "events", batch_size=3, inds=inds)

        for it in [it1, it2, it3, it4]: self.assertTrue(len(it) == len(inds))

        for i in it1: self.assertTrue(i.shape == (2, RECORD_LENGTH))
        for i in it2: self.assertTrue(i.shape == (RECORD_LENGTH, ))
        for i in it3: self.assertTrue(i.shape == (3, RECORD_LENGTH))
        for i in it4: self.assertTrue(i.shape == (3, 2, RECORD_LENGTH))

    def test_batch_remainder(self):
        it = H5Iterator(self.dh.get_filepath(), "events", channels=1, batch_size=11)
        lens_are = np.array([len(i) for i in it])
        lens_should_be = np.ones_like(lens_are, dtype=int)*11
        lens_should_be[-1] = 1 

        self.assertTrue(np.all(lens_are == lens_should_be))

    def test_add_processing(self):
        it1 = H5Iterator(self.dh.get_filepath(), "events", channels=1, batch_size=11)
        it2 = H5Iterator(self.dh.get_filepath(), "events", batch_size=11)
        it3 = H5Iterator(self.dh.get_filepath(), "events", channels=1)
        it4 = H5Iterator(self.dh.get_filepath(), "events")

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

        self.assertTrue(np.array_equal(arr1, arr3))
        self.assertTrue(np.array_equal(arr2, arr4))

    def test_basic(self):
        it1 = H5Iterator(self.dh.get_filepath(), "events", channels=1, batch_size=11)
        it2 = H5Iterator(self.dh.get_filepath(), "events", batch_size=11)
        it3 = H5Iterator(self.dh.get_filepath(), "events", channels=1)
        it4 = H5Iterator(self.dh.get_filepath(), "events")
  
        basic_checks(self, it1)
        basic_checks(self, it2)
        basic_checks(self, it3)
        basic_checks(self, it4)

class TestStreamIterator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        data = ai.data.TestData(filepath=cls.dir.name+'/mock_001', 
                                 duration=RDT_LENGTH,
                                 channels=[0, 1],
                                 sample_frequency=25000,
                                 start_s=13)

        data.generate()

        cls.stream = Stream('cresst', [cls.dir.name+'/mock_001_Ch0.csmpl',
                                        cls.dir.name+'/mock_001_Ch1.csmpl',
                                        cls.dir.name+'/mock_001.test_stamps',
                                        cls.dir.name+'/mock_001.dig_stamps',
                                        cls.dir.name+'/mock_001.par'])

    @classmethod
    def tearDownClass(cls):			
        cls.dir.cleanup()

    def test_basic(self):
        inds = self.stream.time.timestamp_to_ind(self.stream.tp_timestamps["0"])

        basic_checks(self, 
                     StreamIterator(stream=self.stream,
                                    keys="mock_001_Ch0",
                                    inds=inds[:25],
                                    record_length=2**13))
        basic_checks(self, 
                     StreamIterator(stream=self.stream,
                                    keys=["mock_001_Ch0","mock_001_Ch1"],
                                    inds=inds,
                                    record_length=2**14))
        basic_checks(self, 
                     StreamIterator(stream=self.stream,
                                    keys=["mock_001_Ch0","mock_001_Ch1"],
                                    inds=inds,
                                    record_length=2**15))
        basic_checks(self, 
                     StreamIterator(stream=self.stream,
                                    keys=["mock_001_Ch0","mock_001_Ch1"],
                                    inds=inds,
                                    record_length=2**15,
                                    alignment=1/2))

class TestRDTIterator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        test_data = ai.data.TestData(filepath=cls.dir.name+'/test_data', duration=RDT_LENGTH)
        test_data.generate()

        cls.f = RDTFile(cls.dir.name+'/test_data.rdt')

    @classmethod
    def tearDownClass(cls):			
        cls.dir.cleanup()

    def test_basic(self):
        basic_checks(self, 
                     RDTIterator(rdt_channel=self.f[0]))
        basic_checks(self, 
                     RDTIterator(rdt_channel=self.f[0], channels=0))
        basic_checks(self, 
                     RDTIterator(rdt_channel=self.f[(0,1)]))
        basic_checks(self, 
                     RDTIterator(rdt_channel=self.f[(0,1)], channels=(0,1)))
        basic_checks(self, 
                     RDTIterator(rdt_channel=self.f[1]))
        basic_checks(self, 
                     RDTIterator(rdt_channel=self.f[1], channels=1))
        
class TestIteratorCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.dh = ai.DataHandler(record_length=RECORD_LENGTH, 
			   					 sample_frequency=SAMPLE_FREQUENCY,
								 nmbr_channels=2)
		
        cls.dh.set_filepath(path_h5=cls.dir.name, fname="H5test", appendix=False)
        cls.dh.init_empty()

        cls.dh.set("events", event=DATA_3D)
        cls.dh.set("events", time_s=DATA_s, time_mus=DATA_mus, dtype=np.int32)
        
        test_data = ai.data.TestData(filepath=cls.dir.name+'/test_data', 
                                     duration=RDT_LENGTH,
                                     record_length=RECORD_LENGTH)
        test_data.generate()

        cls.f = RDTFile(cls.dir.name+'/test_data.rdt')

    @classmethod
    def tearDownClass(cls):			
        cls.dir.cleanup()

    def test_basic(self):
        it1 = self.dh.get_event_iterator("events")
        it2 = self.f[(0,1)].get_event_iterator()
        it3 = self.dh.get_event_iterator("events", batch_size=13)

        basic_checks(self, it1 + it2)
        basic_checks(self, it1[0] + it2[0])
        basic_checks(self, it1[0] + it2[0] + it1[0])

        with self.assertRaises(ValueError):
            it1 + it2[0]
        
        with self.assertRaises(ValueError):
            it1 + it3

class TestMockIterator(unittest.TestCase):
    def test_basic(self):
        mock = MockData()

        basic_checks(self, 
                     MockIterator(mock=mock))
        basic_checks(self, 
                     MockIterator(mock=mock, channels=0))
        basic_checks(self, 
                     MockIterator(mock=mock, channels=(0,1)))
        
if __name__ == '__main__':
    unittest.main()