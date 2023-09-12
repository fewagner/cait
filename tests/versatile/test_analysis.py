import unittest
import tempfile
import numpy as np

import cait as ai
import cait.versatile as vai

RECORD_LENGTH = 2**15
SAMPLE_FREQUENCY = 2e5

DATA_3D = np.random.rand(2, 100, RECORD_LENGTH)

def func1(ev):
    return np.max(ev)

def func2(ev):
    return (np.min(ev), np.max(ev), np.mean(ev))

def func1_multi(ev):
    return np.max(ev[0])

def func2_multi(ev):
    return (np.max(ev[0]), np.max(ev[1]))

def func3_multi(ev):
    return (np.max(ev[0]), np.max(ev[1]), np.mean(ev[0]))

def func4_multi(ev):
    return ([np.min(ev[0]), np.max(ev[0]), np.mean(ev[0])], [np.min(ev[1]), np.max(ev[1]), np.mean(ev[1])])

class TestApply(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.dh = ai.DataHandler(record_length=RECORD_LENGTH, 
			   					 sample_frequency=SAMPLE_FREQUENCY,
								 nmbr_channels=2)
		
        cls.dh.set_filepath(path_h5=cls.dir.name, fname="event_iterator_test_file", appendix=False)
        cls.dh.init_empty()

        cls.dh.set("events", event=DATA_3D)

    @classmethod
    def tearDownClass(cls):			
        cls.dir.cleanup()

    def test_scalar_out_single_ch(self):
        it1 = self.dh.get_event_iterator("events", 0)
        it2 = self.dh.get_event_iterator("events", 0, batch_size=11)
        arr1 = vai.analysis.apply(func1, it1)
        arr2 = vai.analysis.apply(func1, it2)

        self.assertTrue(arr1.shape == arr2.shape)
        self.assertTrue(arr1.shape == (100,))

    def test_vector_out_single_ch(self):
        it1 = self.dh.get_event_iterator("events", 0)
        it2 = self.dh.get_event_iterator("events", 0, batch_size=11)
        arr1 = vai.analysis.apply(func2, it1)
        arr2 = vai.analysis.apply(func2, it2)

        self.assertTrue(arr1.shape == arr2.shape)
        self.assertTrue(arr1.shape == (100, 3))

    def test_scalar_out_multi_ch(self):
        it1 = self.dh.get_event_iterator("events", 0)
        it2 = self.dh.get_event_iterator("events", 0, batch_size=11)
        arr1 = vai.analysis.apply(func1_multi, it1)
        arr2 = vai.analysis.apply(func1_multi, it2)

        self.assertTrue(arr1.shape == arr2.shape)
        self.assertTrue(arr1.shape == (100,))

    def test_vector2_out_multi_ch(self):
        it1 = self.dh.get_event_iterator("events", 0)
        it2 = self.dh.get_event_iterator("events", 0, batch_size=11)
        arr1 = vai.analysis.apply(func2_multi, it1)
        arr2 = vai.analysis.apply(func2_multi, it2)

        self.assertTrue(arr1.shape == arr2.shape)
        self.assertTrue(arr1.shape == (100, 2))

    def test_vector3_out_multi_ch(self):
        it1 = self.dh.get_event_iterator("events", 0)
        it2 = self.dh.get_event_iterator("events", 0, batch_size=11)
        arr1 = vai.analysis.apply(func3_multi, it1)
        arr2 = vai.analysis.apply(func3_multi, it2)

        self.assertTrue(arr1.shape == arr2.shape)
        self.assertTrue(arr1.shape == (100, 3))

    def test_tensor_out_multi_ch(self):
        it1 = self.dh.get_event_iterator("events", 0)
        it2 = self.dh.get_event_iterator("events", 0, batch_size=11)
        arr1 = vai.analysis.apply(func4_multi, it1)
        arr2 = vai.analysis.apply(func4_multi, it2)

        self.assertTrue(arr1.shape == arr2.shape)
        self.assertTrue(arr1.shape == (100, 2, 3))

    def test_unpack(self):
        it1 = self.dh.get_event_iterator("events", 0)
        it2 = self.dh.get_event_iterator("events", 0, batch_size=11)

        out1, out2 = vai.analysis.apply(func2_multi, it1, unpack=True)
        out3, out4 = vai.analysis.apply(func2_multi, it2, unpack=True)

        self.assertTrue(out1.shape == (100,))
        self.assertTrue(out2.shape == (100,))
        self.assertTrue(out3.shape == (100,))
        self.assertTrue(out4.shape == (100,))

if __name__ == '__main__':
    unittest.main()