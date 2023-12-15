import unittest
import tempfile

import numpy as np
import cait as ai
from cait.versatile import SEV, NPS, OF

RECORD_LENGTH = 2**15
SAMPLE_FREQUENCY = 2e5
DATA_3D = np.random.rand(2, 100, RECORD_LENGTH)
DATA_2D = np.random.rand(2, RECORD_LENGTH)
DATA_1D = np.random.rand(1, RECORD_LENGTH)
DATA_1D_flat = DATA_1D.flatten()

def basic_checks(self, obj, k):
    # Indexing
    n_channels = obj.ndim
    self.assertTrue(obj[:].ndim == n_channels)
    self.assertTrue(obj[list(range(n_channels))].ndim == n_channels)
    if n_channels > 1:
        self.assertTrue(obj[0].ndim == 1)

    # numpy slicing and assignment
    if n_channels > 1:
        obj[0,:10]
        obj[:,0]
        obj[0,0]
        obj[:,0] = 0
    else:
        obj[:10]
        obj[:]
        obj[0]
        obj[0] = 0

    # Methods
    obj.show()
    obj.show(dt=5)
    appendix = f"_{obj.__class__.__name__}_{k}"
    obj.to_file(fname="test"+appendix, out_dir=self.dir.name)
    obj.__class__().from_file(fname="test"+appendix, src_dir=self.dir.name)
    obj.to_dh(self.dh, group=f"test_group"+appendix, dataset="test_ds")
    obj.__class__().from_dh(self.dh, group=f"test_group"+appendix, dataset="test_ds")

class TestArraysWithBenefits(unittest.TestCase): #TODO: Finish test cases
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.dh = ai.DataHandler(record_length=RECORD_LENGTH, 
			   					 sample_frequency=SAMPLE_FREQUENCY,
								 nmbr_channels=2)
		
        cls.dh.set_filepath(path_h5=cls.dir.name, fname="events_with_benefits", appendix=False)
        cls.dh.init_empty()

        cls.dh.set("events", event=DATA_3D)
        cls.dh.set("noise", event=DATA_3D)
        cls.dh.calc_mp()
        cls.dh.calc_sev()
        cls.dh.calc_nps()
        cls.dh.calc_of()

    @classmethod
    def tearDownClass(cls):			
        cls.dir.cleanup()

    def test_SEV(self):
        # Empty creation
        sev1 = SEV()

        # Creation from arrays
        sev2 = SEV(DATA_2D.copy())
        sev3 = SEV(DATA_1D.copy())
        sev4 = SEV(DATA_1D_flat.copy())
        self.assertTrue(np.array_equal(sev3, sev4))

        # Check consistency with vanilla cait
        sev5 = SEV().from_dh(self.dh)

        # Creation from iterator
        it = self.dh.get_event_iterator("events")
        sev6 = SEV(it)
        sev7 = SEV(it[0])
        self.assertTrue(np.array_equal(sev6[0], sev7))

        basic_checks(self, sev2, 0)
        basic_checks(self, sev3, 1)
        basic_checks(self, sev4, 2)
        basic_checks(self, sev5, 3)
        basic_checks(self, sev6, 4)
        basic_checks(self, sev7, 5)
        basic_checks(self, sev2[0], 6)
        basic_checks(self, sev5[0], 7)

    def test_NPS(self):
        # Empty creation
        nps1 = NPS()

        # Creation from arrays
        nps2 = NPS(DATA_2D.copy())
        nps3 = NPS(DATA_1D.copy())
        nps4 = NPS(DATA_1D_flat.copy())
        self.assertTrue(np.array_equal(nps3, nps4))

        # Check consistency with vanilla cait
        nps5 = NPS().from_dh(self.dh)

        # Creation from iterator
        it = self.dh.get_event_iterator("noise")
        nps6 = NPS(it)
        nps7 = NPS(it[0])
        self.assertTrue(np.array_equal(nps6[0], nps7))

        basic_checks(self, nps2, 0)
        basic_checks(self, nps3, 1)
        basic_checks(self, nps4, 2)
        basic_checks(self, nps5, 3)
        basic_checks(self, nps6, 4)
        basic_checks(self, nps7, 5)
        basic_checks(self, nps2[0], 6)
        basic_checks(self, nps5[0], 7)

    def test_OF(self):
        # Empty creation
        of1 = OF()

        # Creation from arrays
        of2 = OF(DATA_2D.copy())
        of3 = OF(DATA_1D.copy())
        of4 = OF(DATA_1D_flat.copy())
        self.assertTrue(np.array_equal(of3, of4))

        # Check consistency with vanilla cait
        of5 = OF().from_dh(self.dh)

        # Creation from NPS/SEV
        of6 = OF(SEV().from_dh(self.dh), NPS().from_dh(self.dh))
        of7 = OF(SEV(self.dh.get_event_iterator("events")), 
                 NPS(self.dh.get_event_iterator("noise")))
        of8 = OF(NPS(self.dh.get_event_iterator("noise")),
                  SEV(self.dh.get_event_iterator("events")))
        self.assertTrue(np.array_equal(of7, of8))

        basic_checks(self, of2, 0)
        basic_checks(self, of3, 1)
        basic_checks(self, of4, 2)
        basic_checks(self, of5, 3)
        basic_checks(self, of6, 4)
        basic_checks(self, of7, 5)
        basic_checks(self, of8, 6)
        basic_checks(self, of2[0], 9)
        basic_checks(self, of5[0], 10)
        basic_checks(self, of8[0], 11)

if __name__ == '__main__':
    unittest.main()