import unittest
import tempfile
import numpy as np

import cait as ai
from cait.versatile.iterators import H5Iterator, apply

RECORD_LENGTH = 2**15
SAMPLE_FREQUENCY = 2e5

DATA_1D = np.random.rand(100)
DATA_2D = np.random.rand(2, 100)
DATA_3D = np.random.rand(2, 100, RECORD_LENGTH)

DATA_2D_single_CH = np.random.rand(1, 100)
DATA_3D_single_CH = np.random.rand(1, 100, 16)

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

    @classmethod
    def tearDownClass(cls):			
        cls.dir.cleanup()

    def test_iterator_bs1_ch2(self):
        it = H5Iterator(self.dh.get_filepath(), "events", "event")
        for i in it:
            self.assertTrue(i.shape == (2, RECORD_LENGTH))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (2, RECORD_LENGTH))

    def test_iterator_bs10_ch2(self):
        # Note that total length is a multiple of batch size. Remaining batches smaller than the batch size are 
        # checked in a different test
        it = H5Iterator(self.dh.get_filepath(), "events", "event", batch_size=10)
        for i in it:
            self.assertTrue(i.shape == (10, 2, RECORD_LENGTH))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (10, 2, RECORD_LENGTH))

    def test_iterator_bs1_ch1(self):
        it = H5Iterator(self.dh.get_filepath(), "events", "event", channels=1)
        for i in it:
            self.assertTrue(i.shape == (RECORD_LENGTH,))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (RECORD_LENGTH,))

        # Same but with list instead of integer
        it = H5Iterator(self.dh.get_filepath(), "events", "event", channels=[1])
        for i in it:
            self.assertTrue(i.shape == (RECORD_LENGTH,))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (RECORD_LENGTH,))

    def test_iterator_bs10_ch1(self):
        it = H5Iterator(self.dh.get_filepath(), "events", "event", channels=1, batch_size=10)
        for i in it:
            self.assertTrue(i.shape == (10, RECORD_LENGTH))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (10, RECORD_LENGTH))

        # Same but with list instead of integer
        it = H5Iterator(self.dh.get_filepath(), "events", "event", channels=[1], batch_size=10)
        for i in it:
            self.assertTrue(i.shape == (10, RECORD_LENGTH))

        with it as opened_it:
            for i in opened_it:
                self.assertTrue(i.shape == (10, RECORD_LENGTH))

    def test_inds(self):
        inds = [12, 18, 55, 81, 90, 99]
        it1 = H5Iterator(self.dh.get_filepath(), "events", "event", inds=inds)
        it2 = H5Iterator(self.dh.get_filepath(), "events", "event", channels=1, inds=inds)
        it3 = H5Iterator(self.dh.get_filepath(), "events", "event", channels=1, batch_size=3, inds=inds)
        it4 = H5Iterator(self.dh.get_filepath(), "events", "event", batch_size=3, inds=inds)

        for it in [it1, it2, it3, it4]: self.assertTrue(len(it) == len(inds))

        for i in it1: self.assertTrue(i.shape == (2, RECORD_LENGTH))
        for i in it2: self.assertTrue(i.shape == (RECORD_LENGTH, ))
        for i in it3: self.assertTrue(i.shape == (3, RECORD_LENGTH))
        for i in it4: self.assertTrue(i.shape == (3, 2, RECORD_LENGTH))

    def test_batch_remainder(self):
        it = H5Iterator(self.dh.get_filepath(), "events", "event", channels=1, batch_size=11)
        lens_are = np.array([len(i) for i in it])
        lens_should_be = np.ones_like(lens_are, dtype=int)*11
        lens_should_be[-1] = 1 

        self.assertTrue(np.all(lens_are == lens_should_be))

    def test_add_processing(self):
        it1 = H5Iterator(self.dh.get_filepath(), "events", "event", channels=1, batch_size=11)
        it2 = H5Iterator(self.dh.get_filepath(), "events", "event", batch_size=11)
        it3 = H5Iterator(self.dh.get_filepath(), "events", "event", channels=1)
        it4 = H5Iterator(self.dh.get_filepath(), "events", "event")

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

    #TODO: add testing for slicing

if __name__ == '__main__':
    unittest.main()