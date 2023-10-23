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

def func_events(ev):
    return ev

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
        out1 = vai.apply(func1, it1)
        out2 = vai.apply(func1, it2)

        self.assertTrue(out1.shape == out2.shape)
        self.assertTrue(out1.shape == (100,))

    def test_vector_out_single_ch(self):
        it1 = self.dh.get_event_iterator("events", 0)
        it2 = self.dh.get_event_iterator("events", 0, batch_size=11)
        out11, out12, out13 = vai.apply(func2, it1)
        out21, out22, out23 = vai.apply(func2, it2)

        self.assertTrue(out11.shape == out21.shape)
        self.assertTrue(out12.shape == out22.shape)
        self.assertTrue(out13.shape == out23.shape)
        self.assertTrue(out11.shape == (100,))
        self.assertTrue(out12.shape == (100,))
        self.assertTrue(out13.shape == (100,))

    def test_scalar_out_multi_ch(self):
        it1 = self.dh.get_event_iterator("events")
        it2 = self.dh.get_event_iterator("events", batch_size=11)
        out1 = vai.apply(func1_multi, it1)
        out2 = vai.apply(func1_multi, it2)

        self.assertTrue(out1.shape == out2.shape)
        self.assertTrue(out1.shape == (100, ))

    def test_vector2_out_multi_ch(self):
        it1 = self.dh.get_event_iterator("events")
        it2 = self.dh.get_event_iterator("events", batch_size=11)
        out11, out12 = vai.apply(func2_multi, it1)
        out21, out22 = vai.apply(func2_multi, it2)

        self.assertTrue(out11.shape == out21.shape)
        self.assertTrue(out12.shape == out22.shape)
        self.assertTrue(out11.shape == (100,))
        self.assertTrue(out12.shape == (100,))

    def test_vector3_out_multi_ch(self):
        it1 = self.dh.get_event_iterator("events")
        it2 = self.dh.get_event_iterator("events", batch_size=11)
        out11, out12, out13 = vai.apply(func3_multi, it1)
        out21, out22, out23 = vai.apply(func3_multi, it2)

        self.assertTrue(out11.shape == out21.shape)
        self.assertTrue(out12.shape == out22.shape)
        self.assertTrue(out13.shape == out23.shape)
        self.assertTrue(out11.shape == (100,))
        self.assertTrue(out12.shape == (100,))
        self.assertTrue(out13.shape == (100,))

    def test_tensor_out_multi_ch(self):
        it1 = self.dh.get_event_iterator("events")
        it2 = self.dh.get_event_iterator("events", batch_size=11)
        out11, out12 = vai.apply(func4_multi, it1)
        out21, out22 = vai.apply(func4_multi, it2)

        self.assertTrue(out11.shape == out21.shape)
        self.assertTrue(out12.shape == out22.shape)
        self.assertTrue(out11.shape == (100, 3))
        self.assertTrue(out12.shape == (100, 3))

    def test_event_size_output(self):
        it1 = self.dh.get_event_iterator("events")
        it2 = self.dh.get_event_iterator("events", batch_size=11)
        it3 = self.dh.get_event_iterator("events", 0)
        it4 = self.dh.get_event_iterator("events", 0, batch_size=11)
        out1 = vai.apply(func_events, it1)
        out2 = vai.apply(func_events, it2)
        out3 = vai.apply(func_events, it3)
        out4 = vai.apply(func_events, it4)

        self.assertTrue(out1.shape == out2.shape)
        self.assertTrue(out3.shape == out4.shape)
        self.assertTrue(out1.shape == (100, 2, RECORD_LENGTH))
        self.assertTrue(out3.shape == (100, RECORD_LENGTH))

if __name__ == '__main__':
    unittest.main()