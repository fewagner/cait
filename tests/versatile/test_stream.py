import unittest
import tempfile

import numpy as np
import cait as ai
import cait.versatile as vai

LENGTH = 100

def basic_checks(self, stream):
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
    self.assertTrue(np.array_equal(next(iter(it1)), next(iter(it2))))

class TestStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        data = ai.data.TestData(filepath=cls.dir.name+'/mock_001', 
                                 duration=LENGTH,
                                 channels=[0, 1],
                                 sample_frequency=25000,
                                 start_s=13,
                                 tpas=[20, -1.0, 20, 0.3, 0, 20, 0.5, 1, 20, 3, -1, 20, 0, 10])

        data.generate()

        cls.stream = vai.Stream('cresst', [cls.dir.name+'/mock_001_Ch0.csmpl',
                                           cls.dir.name+'/mock_001_Ch1.csmpl',
                                           cls.dir.name+'/mock_001.test_stamps',
                                           cls.dir.name+'/mock_001.dig_stamps',
                                           cls.dir.name+'/mock_001.par'])

    @classmethod
    def tearDownClass(cls):			
        cls.dir.cleanup()

    def test_StreamTime(self):
        t = self.stream.time
        t[0]
        t[-1]
        t[0:10]
        t[[1,3,10]]
        ts = t[1000]
        dt = t.timestamp_to_datetime(ts)
        self.assertTrue(t.timestamp_to_ind(ts) == 1000)
        self.assertTrue(t.datetime_to_timestamp(dt) == ts)

    def test_CRESST(self):
        s1 = vai.Stream('cresst', [self.dir.name+'/mock_001_Ch0.csmpl',
                                    self.dir.name+'/mock_001.par'])
        
        s2 = vai.Stream('cresst', [self.dir.name+'/mock_001_Ch0.csmpl',
                                    self.dir.name+'/mock_001_Ch1.csmpl',
                                    self.dir.name+'/mock_001.test_stamps',
                                    self.dir.name+'/mock_001.dig_stamps',
                                    self.dir.name+'/mock_001.par'])
        
        # Not available if not provided via test_stamps
        with self.assertRaises(KeyError):
            s1.tpas
        with self.assertRaises(KeyError):
            s1.tp_timestamps

        s2.tpas
        s2.tp_timestamps
        it = s2.get_event_iterator("mock_001_Ch0", 2**13, timestamps=s2.tp_timestamps["0"])
        self.assertTrue(len(s2.tpas["0"]) == len(s2.tp_timestamps["0"]))
        self.assertTrue(len(it) == len(s2.tp_timestamps["0"]))
        
        basic_checks(self, s1)
        basic_checks(self, s2)

    # TODO
    def test_VDAQ2(self):
        ...

    # TODO
    def test_VDAQ3(self):
        ...

    # TODO: test either on "real" data or figure out how cait simulates stream files
    # to validate the trigger indices
    def test_trigger(self):
        i, v = vai.trigger(self.stream, "mock_001_Ch0", 0.1, 2**14)
        i, v = vai.trigger(self.stream, "mock_001_Ch0", 0.1, 2**14, n_triggers=10)
        i, v = vai.trigger(self.stream, "mock_001_Ch1", 0.1, 2**14, preprocessing=lambda x: -x)

if __name__ == '__main__':
    unittest.main()