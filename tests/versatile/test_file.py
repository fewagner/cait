import unittest
import tempfile
import numpy as np
import h5py

import cait as ai
import cait.versatile as vai

RECORD_LENGTH = 2**15
SAMPLE_FREQUENCY = 2e5
DATA_1D = np.random.rand(100)
DATA_2D = np.random.rand(2, 100)
DATA_3D = np.random.rand(2, 100, RECORD_LENGTH)

N_FILES = 5 # for both combine and merge

def relative_res(a, b):
    return np.abs((a-b)/a)

class TestCombineMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.filenames = list()
        for n in range(2*N_FILES): # N_FILES for both combine and merge separately
            fname = f"file{n}"
            cls.filenames.append(fname)

            dh = ai.DataHandler(record_length=RECORD_LENGTH, 
			   					 sample_frequency=SAMPLE_FREQUENCY,
								 nmbr_channels=2)

		
            dh.set_filepath(path_h5=cls.dir.name, fname=fname, appendix=False)
            dh.init_empty()

            # to have consistent file content for both sets of files
            # (i.e. for both merge and combine)
            exp = np.floor(n/2) 

            # write data to files
            dh.set(group="group1", ds1=DATA_1D*10**exp, ds2=DATA_2D*10**exp, ds3=DATA_3D*10**exp)
            dh.set(group="group2", ds1=DATA_1D*10**(2*exp), ds2=DATA_2D*10**(2*exp), ds3=DATA_3D*10**(2*exp))
            dh.set(group="testgroup", ds=np.ones_like(DATA_1D))

    @classmethod
    def tearDownClass(cls):			
        cls.dir.cleanup()

    def test_file_consistency(self):
        # test if exception is raised in case the dtypes don't agree
        files = ["2combine1_dtype", "2combine2_dtype"]
        dtypes = ["float32", "float64"]
        for file, dtype in zip(files, dtypes):
            dh = ai.DataHandler(record_length=RECORD_LENGTH, 
			   				    sample_frequency=SAMPLE_FREQUENCY,
							    nmbr_channels=2)
            dh.set_filepath(path_h5=self.dir.name, fname=file, appendix=False)
            dh.init_empty()

            dh.set(group="group1", ds1=DATA_1D, ds2=DATA_2D, ds3=DATA_3D, dtype=dtype)
            dh.set(group="group2", ds1=DATA_1D, ds2=DATA_2D, ds3=DATA_3D, dtype=dtype)
            dh.set(group="testgroup", ds=np.ones_like(DATA_1D), dtype=dtype)

        with self.assertRaises(AssertionError):
            vai.file.check_file_consistency(files=files, 
                                            src_dir=self.dir.name,
                                            groups_combine=["group1","group2"], 
                                            groups_include=["testgroup"])

        # test if exception is raised in case the shapes don't agree
        files = ["2combine1_shape", "2combine2_shape"]
        for file in files:
            dh = ai.DataHandler(record_length=RECORD_LENGTH, 
			   				    sample_frequency=SAMPLE_FREQUENCY,
							    nmbr_channels=2)
            dh.set_filepath(path_h5=self.dir.name, fname=file, appendix=False)
            dh.init_empty()

            if file == "2combine2_shape":
                dh.set(group="group1", ds1=DATA_1D, ds2=np.transpose(DATA_2D), ds3=DATA_3D)
            else:
                dh.set(group="group1", ds1=DATA_1D, ds2=DATA_2D, ds3=DATA_3D)
            dh.set(group="group2", ds1=DATA_1D, ds2=DATA_2D, ds3=DATA_3D)
            dh.set(group="testgroup", ds=np.ones_like(DATA_1D))

        with self.assertRaises(AssertionError):
            vai.file.check_file_consistency(files=files, 
                                            src_dir=self.dir.name,
                                            groups_combine=["group1","group2"], 
                                            groups_include=["testgroup"])
            
        # test if exception is raised in case the group/dataset is not present
        files = ["2combine1_ds", "2combine2_ds"]
        for file in files:
            dh = ai.DataHandler(record_length=RECORD_LENGTH, 
			   				    sample_frequency=SAMPLE_FREQUENCY,
							    nmbr_channels=2)
            dh.set_filepath(path_h5=self.dir.name, fname=file, appendix=False)
            dh.init_empty()

            if file == "2combine2_ds":
                dh.set(group="group1", ds1=DATA_1D, ds2=DATA_2D, ds3=DATA_3D)
            if file == "2combine2_ds":
                dh.set(group="group2", ds1=DATA_1D, ds2=DATA_2D, ds3=DATA_3D)
            else:
                dh.set(group="group2", ds1=DATA_1D, ds3=DATA_3D)
            dh.set(group="testgroup", ds=np.ones_like(DATA_1D))

        with self.assertRaises(AssertionError): # not all groups present in all files
            vai.file.check_file_consistency(files=files, 
                                            src_dir=self.dir.name,
                                            groups_combine=["group1","group2"], 
                                            groups_include=["testgroup"])
        with self.assertRaises(AssertionError): # ds2 in group2 missing from second file
            vai.file.check_file_consistency(files=files, 
                                            src_dir=self.dir.name,
                                            groups_combine=["group2"], 
                                            groups_include=["testgroup"])
        with self.assertRaises(AssertionError): # include group not present in either file
            vai.file.check_file_consistency(files=files, 
                                            src_dir=self.dir.name,
                                            groups_combine=[], 
                                            groups_include=["non_existent_group"])

    def test_combine(self):
        # combine files
        vai.file.combine(fname="output_combine",
                         files=self.filenames[::2], # every second starting from 0th
                         src_dir=self.dir.name,
                         out_dir=self.dir.name, 
                         groups_combine=["group1","group2"], 
                         groups_include=["testgroup"])
        
        # validate shapes and dataset contents
        dh = self.validate("output_combine")

        # check if all datasets are virtual
        with h5py.File(dh.get_filepath(), 'r') as h5f:
            for group in list(h5f.keys()):
                for ds in h5f[group]:
                    self.assertTrue(h5f[group][ds].is_virtual)

        # check set() method with virtual datasets
        # even with change_existing=True, warning should be given
        self.assertWarns(UserWarning, dh.set, 
                         group="group1", 
                         change_existing=True, 
                         ds1=DATA_1D)
        # even with overwrite_existing=True, warning should be given
        self.assertWarns(UserWarning, dh.set,
                         group="group1", 
                         change_existing=True,
                         overwrite_existing=True, 
                         ds1=DATA_1D)
        # same as above but in channel mode
        self.assertWarns(UserWarning, dh.set,
                         group="group1", 
                         n_channels=2,
                         channel=0,
                         change_existing=True, 
                         ds2=DATA_2D[0,:])
        self.assertWarns(UserWarning, dh.set,
                         group="group1", 
                         n_channels=2,
                         channel=0,
                         change_existing=True, 
                         overwrite_existing=True,
                         ds2=DATA_2D[0,:])
        
        # test if writing works
        old_ds = dh.get("group1", "ds1")
        dh.set(group="group1", ds1=old_ds*10, write_to_virtual=True)
        self.assertTrue(np.all(relative_res(old_ds*10, dh.get("group1", "ds1")) < 1e-5))
        with h5py.File(dh.get_filepath(), 'r') as h5f:
            self.assertTrue(h5f["group1/ds1"].is_virtual) 

        # test if OVER-writing works
        dh.set(group="group1", ds1=old_ds*10, write_to_virtual=False)
        self.assertTrue(np.all(relative_res(old_ds*10, dh.get("group1", "ds1")) < 1e-5))
        with h5py.File(dh.get_filepath(), 'r') as h5f:
            self.assertFalse(h5f["group1/ds1"].is_virtual) 

    def test_merge(self):
        # merge files
        vai.file.merge(fname="output_merge",
                         files=self.filenames[1::2], # every second starting from first
                         src_dir=self.dir.name,
                         out_dir=self.dir.name, 
                         groups_merge=["group1","group2"], 
                         groups_include=["testgroup"])
        
        # validate shapes and dataset contents
        dh = self.validate("output_merge")

        # check if none of the datasets is virtual
        with h5py.File(dh.get_filepath(), 'r') as h5f:
            for group in list(h5f.keys()):
                for ds in h5f[group]:
                    self.assertFalse(h5f[group][ds].is_virtual)
        
    def validate(self, name):
        # check if merge was successful
        dh = ai.DataHandler(record_length=RECORD_LENGTH, 
			   				sample_frequency=SAMPLE_FREQUENCY,
							nmbr_channels=2)
        dh.set_filepath(path_h5=self.dir.name, fname=name, appendix=False)

        # check if shapes are as expected
        self.assertTrue( dh.get("group1", "ds1").shape == (DATA_1D.shape[0]*N_FILES,))
        self.assertTrue( dh.get("group1", "ds2").shape == (DATA_2D.shape[0], DATA_2D.shape[1]*N_FILES))
        self.assertTrue( dh.get("group1", "ds3").shape == (DATA_3D.shape[0], DATA_3D.shape[1]*N_FILES, DATA_3D.shape[2]))

        # check content
        diffs = list()
        diffs.append( relative_res(dh.get("group1", "ds1"), np.hstack([DATA_1D*10**n for n in range(N_FILES)])) )
        diffs.append( relative_res(dh.get("group1", "ds2"), np.hstack([DATA_2D*10**n for n in range(N_FILES)])) )
        diffs.append( relative_res(dh.get("group1", "ds3"), np.hstack([DATA_3D*10**n for n in range(N_FILES)])) )

        diffs.append( relative_res(dh.get("group2", "ds1"), np.hstack([DATA_1D*10**(2*n) for n in range(N_FILES)])) )
        diffs.append( relative_res(dh.get("group2", "ds2"), np.hstack([DATA_2D*10**(2*n) for n in range(N_FILES)])) )
        diffs.append( relative_res(dh.get("group2", "ds3"), np.hstack([DATA_3D*10**(2*n) for n in range(N_FILES)])) )

        diffs.append( relative_res(dh.get("testgroup", "ds"), np.ones_like(DATA_1D)) )

        self.assertTrue( all( [np.all(d < 1e-5) for d in diffs] ) )

        return dh

class TestEventIterator(unittest.TestCase):
    # TODO
    def test_iterator(self):
        ...
if __name__ == '__main__':
    unittest.main()