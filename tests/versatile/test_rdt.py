import unittest
import tempfile

import cait as ai
from cait.versatile.rdt import RDTFile, PARFile

RDT_LENGTH = 100

class TestRDT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        data1 = ai.data.TestData(filepath=cls.dir.name+'/data1', 
                                 duration=RDT_LENGTH,
                                 channels=[0, 1],
                                 sample_frequency=25000,
                                 start_s=13)
        data2 = ai.data.TestData(filepath=cls.dir.name+'/data2', 
                                 duration=RDT_LENGTH,
                                 channels=[1],
                                 sample_frequency=50000,
                                 start_s=20349,
                                 include_carriers=False)

        data1.generate()
        data2.generate()

        cls.f1 = RDTFile(cls.dir.name+'/data1.rdt')
        cls.f2 = RDTFile(cls.dir.name+'/data2.rdt')

    @classmethod
    def tearDownClass(cls):			
        cls.dir.cleanup()

    def test_par(self):
        p1 = PARFile(self.dir.name+'/data1.par')
        p2 = PARFile(self.dir.name+'/data2.par')

        self.assertTrue(p1.start_s == 13)
        self.assertTrue(p2.start_s == 20349)

        self.assertTrue(p1.stop_s == 13 + RDT_LENGTH)
        self.assertTrue(p2.stop_s == 20349 + RDT_LENGTH)

    def test_constructor(self):
        self.assertTrue(self.f1.sample_frequency == 25000)
        self.assertTrue(self.f2.sample_frequency == 50000)

        self.assertTrue(self.f1.keys == [0, 1, (0,1)])
        self.assertTrue(self.f2.keys == [1])

    def test_channels(self):
        self.assertTrue(self.f1[0].n_channels == 1)
        self.assertTrue(self.f1[1].n_channels == 1)
        self.assertTrue(self.f1[(0,1)].n_channels == 2)

        self.assertTrue(self.f2[1].n_channels == 1)

        tpas = [-1.0, 0.0, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]
        self.assertTrue([float(str(x)) for x in self.f1[0].unique_tpas] == tpas)
        self.assertTrue([float(str(x)) for x in self.f1[(0,1)].unique_tpas] == tpas)
        self.assertTrue([float(str(x)) for x in self.f2[1].unique_tpas] == tpas)

    def test_methods_rdt(self):
        self.assertTrue(self.f1.default_channels == (0,1))
        self.assertTrue(self.f2.default_channels == 1)

        self.f1.time_base_us
        self.f1.measuring_time_h
        self.f1.keys
        self.f1.get_voltage_trace([0,1,2])

        self.f2.time_base_us
        self.f2.measuring_time_h
        self.f2.keys
        self.f2.get_voltage_trace([0,1,2])

    def test_methods_channels(self):
        self.f1[0].key
        self.f1[(0,1)].key
        self.f2[1].key

        self.f1[0].timestamps
        self.f1[(0,1)].timestamps
        self.f2[1].timestamps

        self.f1[0].tpas
        self.f1[(0,1)].tpas
        self.f2[1].tpas

        it1 = self.f1[0].get_event_iterator()
        it2 = self.f1[(0,1)].get_event_iterator()
        it3 = self.f2[1].get_event_iterator()

        self.assertTrue(it1.n_channels == 1)
        self.assertTrue(it2.n_channels == 2)
        self.assertTrue(it3.n_channels == 1)

        next(iter(it1))
        next(iter(it2))
        next(iter(it3))

if __name__ == '__main__':
    unittest.main()