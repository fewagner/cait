import unittest
import tempfile
import h5py
import os
import numpy as np
import cait as ai

RECORD_LENGTH = 2**15
SAMPLE_FREQUENCY = 2e5
DATA_1D = np.random.rand(100)
DATA_2D = np.random.rand(2, 100)
DATA_3D = np.random.rand(2, 100, RECORD_LENGTH)

class TestDataHandler(unittest.TestCase):
	# tests all methods directly defined in DataHandler (not within mixins) except:
	# get_event_iterator, import_labels, import_predictions, drop_raw_data
	# downsample_raw_data, truncate_raw_data, generate_startstop, record_window
	# methods are also not tested for virtual datasets
	@classmethod
	def setUpClass(cls):
		cls.dir = tempfile.TemporaryDirectory()
		cls.dh = ai.DataHandler(record_length=RECORD_LENGTH, 
			   					 sample_frequency=SAMPLE_FREQUENCY,
								 nmbr_channels=2)
		
		cls.dh.set_filepath(path_h5=cls.dir.name, fname="test_file", appendix=False)
		cls.dh.init_empty()

	@classmethod
	def tearDownClass(cls):			
		cls.dir.cleanup()

	def test_get_fileinfos(self):
		dh = ai.DataHandler(nmbr_channels=2)

		# test if exception is raised when filepath is not set
		with self.assertRaises(Exception):
			dh.get_filepath()

		dh.set_filepath(path_h5=self.dir.name, fname="this_does_not_exist", appendix=False)

		# test if exception is raised when filepath does not exist
		with self.assertRaises(FileNotFoundError):
			dh.get_filepath()

		dh.init_empty()

		# test output of get_filepath
		self.assertEqual(dh.get_filepath(absolute=True),
		   				 os.path.join(self.dir.name, "this_does_not_exist.h5"))
		# test output of get_filename
		self.assertEqual(dh.get_filename(), "this_does_not_exist")
		# test output of get_filedirectory
		self.assertEqual(dh.get_filedirectory(absolute=True), self.dir.name)
		
	def test_get_filehandle(self):
		with h5py.File(self.dh.get_filepath(), "r") as f:
			with self.dh.get_filehandle(mode="r") as fh:
				self.assertEqual(f, fh)

		mock_file = os.path.join(self.dir.name, "mock_file.h5")
		with h5py.File(mock_file, "a") as f:
			with self.dh.get_filehandle(path=mock_file, mode="a") as fh:
				self.assertEqual(f, fh)

	def test_manipulations(self):
		# test info methods before manipulation
		self.info_methods()

		for name in dir(self): # dir is sorted
			if name.startswith("manipulations"):
				getattr(self, name)()

		# test info methods again after manipulation
		self.info_methods()

	def info_methods(self):
		print(str(self.dh))
		self.dh.keys()
		self.dh.content()

	def manipulations1(self): # testing set() for non-virtual datasets
		# create an empty group
		self.dh.set(group="group1")

		# create a group and write datasets directly
		self.dh.set(group="group2", 
	      			ds1=DATA_1D, 
					ds2=DATA_2D, 
					ds3=DATA_3D,
					dtype=DATA_1D.dtype)
		
		# create a group and write datasets in channel mode
		self.dh.set(group="group3",
	      			n_channels=2,
					channel=0,
					ds1d=DATA_1D,
					ds2d=DATA_2D)
		self.dh.set(group="group3",
	      			n_channels=2,
					channel=1,
					ds1d=10*DATA_1D,
					ds2d=100*DATA_2D,
					change_existing=True)
		
		# overwrite existing datasets 
		self.dh.set(group="group4",
					ds=DATA_1D)
		self.dh.set(group="group4",
	      			overwrite_existing=True,
					ds=DATA_2D)
		
		with h5py.File(self.dh.get_filepath(), 'r') as h5f:
			# check if groups are present
			self.assertTrue( all( [g in list(h5f.keys()) for g in ["group1", "group2", "group3"]] ))
			# check if datasets are present in group2
			self.assertTrue( all( [ds in list(h5f["group2"].keys()) for ds in ["ds1", "ds2", "ds3"]] ))
			# check if datasets in group2 have correct shape
			self.assertTrue( all( [h5f["group2"][ds].shape == ref.shape for ds, ref in zip(["ds1", "ds2", "ds3"], [DATA_1D, DATA_2D, DATA_3D])] ))
			# check if datasets in group2 have correct dtype
			self.assertTrue( all( [h5f["group2"][ds].dtype == ref.dtype for ds, ref in zip(["ds1", "ds2", "ds3"], [DATA_1D, DATA_2D, DATA_3D])] ))

			# check if datasets in group3 have correct shape
			self.assertEqual(h5f["group3/ds1d"].shape, (2, DATA_1D.shape[0]))
			self.assertEqual(h5f["group3/ds2d"].shape, (2, DATA_2D.shape[0], DATA_2D.shape[1]))

			# check if dataset was overwritten
			self.assertEqual( h5f["group4/ds"].shape, DATA_2D.shape)

		# check warnings for incorrect use
		self.assertWarns(UserWarning, self.dh.set, # n_channels without channel
		   				 group="group",
						 n_channels=2)
		self.assertWarns(UserWarning, self.dh.set, # channel without n_channels
		   				 group="group",
						 channel=0)
		self.assertWarns(UserWarning, self.dh.set, # no change_existing
		   				 group="group2",
						 ds1=DATA_1D)
		self.assertWarns(UserWarning, self.dh.set, # no overwrite_existing
		   				 group="group2",
						 change_existing=True,
						 ds1=DATA_1D)
		self.assertWarns(UserWarning, self.dh.set, # no overwrite_existing (2 channels exist)
		   				 group="group3",
						 n_channels=3,
						 channel=1,
						 change_existing=True,
						 ds1d=DATA_1D)
			
	def manipulations2(self): # testing get() for non-virtual datasets
		# check if written datasets match (within numerical precision)
		diffs = list()
		diffs.append( self.dh.get("group2", "ds1") - DATA_1D )
		diffs.append( self.dh.get("group2", "ds2") - DATA_2D )
		diffs.append( self.dh.get("group2", "ds3") - DATA_3D )
		
		diffs.append( self.dh.get("group3", "ds1d") - np.array([DATA_1D, 10*DATA_1D]) )
		diffs.append( self.dh.get("group3", "ds2d") - np.array([DATA_2D, 100*DATA_2D]) )

		diffs.append( self.dh.get("group4", "ds") - DATA_2D )

		# check if slicing works
		diffs.append( self.dh.get("group3", "ds1d", 0) - DATA_1D)
		diffs.append( self.dh.get("group3", "ds1d", 1) - 10*DATA_1D)
		diffs.append( self.dh.get("group3", "ds2d", 0, None, 3) - DATA_2D[:,3])
		diffs.append( self.dh.get("group3", "ds2d", 1, [1]) - 100*DATA_2D[1])

		self.assertTrue( all( [np.all(np.abs(d) < 1e-5) for d in diffs] ) )

	def manipulations3(self): # testing drop, repackage and rename
		self.dh.drop("group1") # drop empty group
		self.dh.drop("group4") # drop non-empty group
		self.dh.drop("group2", "ds1") # drop dataset
		self.dh.drop("group2", "ds2", repackage=True) # drop dataset and repackage
		self.dh.repackage() # repackage
		
		# check if groups/datasets are gone
		with h5py.File(self.dh.get_filepath(), 'r') as h5f:
			still_exist = ["group1" in list(h5f.keys()),
		  				   "group4" in list(h5f.keys()),
						   "ds1" in list(h5f["group2"].keys()),
						   "ds2" in list(h5f["group2"].keys())
		  				  ]
		self.assertFalse( any(still_exist) )

		# test rename
		self.dh.rename(group3="group3_new")
		self.dh.rename(group="group3_new", ds1d="ds1d_new")

		# check if names were changed and values stayed the same
		with h5py.File(self.dh.get_filepath(), 'r') as h5f:
			self.assertFalse( "group3" in list(h5f.keys()) )
			self.assertTrue( "group3_new" in list(h5f.keys()) )
			self.assertFalse( "ds1d" in list(h5f["group3_new"].keys()) )
			self.assertTrue( "ds1d_new" in list(h5f["group3_new"].keys()) )

		d = self.dh.get("group3_new", "ds1d_new") - np.array([DATA_1D, 10*DATA_1D])
		self.assertTrue( np.all(np.abs(d) < 1e-5))

if __name__ == '__main__':
    unittest.main()