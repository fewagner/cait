import pytest

import os
import h5py
import numpy as np
import cait as ai

from .fixtures import tempdir, datahandler, testdata_1D_2D_3D_s_mus, RECORD_LENGTH

class TestDataHandler:
    # tests all methods directly defined in DataHandler (not within mixins) except:
    # import_labels, import_predictions, drop_raw_data
    # downsample_raw_data, truncate_raw_data, generate_startstop, record_window
    # methods are also not tested for virtual datasets
    def test_get_fileinfos(self, tempdir):
        dh = ai.DataHandler(nmbr_channels=2)

        # test if exception is raised when filepath is not set
        with pytest.raises(Exception): dh.get_filepath()

        dh.set_filepath(path_h5=tempdir.name, fname="this_does_not_exist", appendix=False)

        # test if exception is raised when filepath does not exist
        with pytest.raises(FileNotFoundError): dh.get_filepath()

        dh.init_empty()

        # test output of get_filepath
        assert dh.get_filepath(absolute=True) == os.path.join(tempdir.name, "this_does_not_exist.h5")
        # test output of get_filename
        assert dh.get_filename() == "this_does_not_exist"
        # test output of get_filedirectory
        assert dh.get_filedirectory(absolute=True) == tempdir.name
        
    def test_get_filehandle(self, datahandler, tempdir):
        with h5py.File(datahandler.get_filepath(), "r") as f:
            with datahandler.get_filehandle(mode="r") as fh:
                assert f == fh

        mock_file = os.path.join(tempdir.name, "mock_file.h5")
        with h5py.File(mock_file, "a") as f:
            with datahandler.get_filehandle(path=mock_file, mode="a") as fh:
                assert f == fh

    def test_manipulations(self, datahandler, testdata_1D_2D_3D_s_mus):
        d1, d2, d3, *_ = testdata_1D_2D_3D_s_mus

        # test info methods before manipulation
        print(str(datahandler))
        repr(datahandler)
        datahandler.keys()
        datahandler.content()

        ############### PART 1 ####################
        ### (test set for non-virtual datasets) ###
        ###########################################
        # create an empty group
        datahandler.set(group="group1")

        # create a group and write datasets directly
        datahandler.set(group="group2", ds1=d1, ds2=d2, ds3=d3, dtype=d1.dtype)
        
        # create a group and write datasets in channel mode
        datahandler.set(group="group3", n_channels=2, channel=0, ds1d=d1, ds2d=d2)
        datahandler.set(group="group3", n_channels=2, channel=1, 
                        ds1d=10*d1, ds2d=100*d2, change_existing=True)
        
        # overwrite existing datasets 
        datahandler.set(group="group4", ds=d1)
        datahandler.set(group="group4", overwrite_existing=True, ds=d2)
        
        with h5py.File(datahandler.get_filepath(), 'r') as h5f:
            # check if groups are present
            assert all( [g in list(h5f.keys()) for g in ["group1", "group2", "group3"]] )
            # check if datasets are present in group2
            assert all( [ds in list(h5f["group2"].keys()) for ds in ["ds1", "ds2", "ds3"]] )
            # check if datasets in group2 have correct shape
            assert all( [h5f["group2"][ds].shape == ref.shape for ds, ref in zip(["ds1", "ds2", "ds3"], [d1, d2, d3])] )
            # check if datasets in group2 have correct dtype
            assert all( [h5f["group2"][ds].dtype == ref.dtype for ds, ref in zip(["ds1", "ds2", "ds3"], [d1, d2, d3])] )

            # check if datasets in group3 have correct shape
            assert h5f["group3/ds1d"].shape == (2, d1.shape[0])
            assert h5f["group3/ds2d"].shape == (2, d2.shape[0], d2.shape[1])

            # check if dataset was overwritten
            assert h5f["group4/ds"].shape == d2.shape

        # check warnings for incorrect use
        # n_channels without channel
        with pytest.warns(UserWarning): datahandler.set(group="group", n_channels=2)
        # channel without n_channels
        with pytest.warns(UserWarning): datahandler.set(group="group", channel=0)
        # no change_existing
        with pytest.warns(UserWarning): datahandler.set(group="group2", ds1=d1)
        # no overwrite_existing
        with pytest.warns(UserWarning): 
            datahandler.set(group="group2", change_existing=True, ds1=d1)
        # no overwrite_existing (2 channels exist)
        with pytest.warns(UserWarning): 
            datahandler.set(group="group3", n_channels=3, channel=1, change_existing=True, ds1d=d1)
              
        ############### PART 2 ####################
        ### (test get for non-virtual datasets) ###
        ###########################################
        # check if written datasets match (within numerical precision)
        diffs = list()
        diffs.append( datahandler.get("group2", "ds1") - d1 )
        diffs.append( datahandler.get("group2", "ds2") - d2 )
        diffs.append( datahandler.get("group2", "ds3") - d3 )
        diffs.append( datahandler.get("group3", "ds1d") - np.array([d1, 10*d1]) )
        diffs.append( datahandler.get("group3", "ds2d") - np.array([d2, 100*d2]) )
        diffs.append( datahandler.get("group4", "ds") - d2 )
        # check if slicing works
        diffs.append( datahandler.get("group3", "ds1d", 0) - d1)
        diffs.append( datahandler.get("group3", "ds1d", 1) - 10*d1)
        diffs.append( datahandler.get("group3", "ds2d", 0, None, 3) - d2[:,3])
        diffs.append( datahandler.get("group3", "ds2d", 1, [1]) - 100*d2[1])

        assert all( [np.all(np.abs(d) < 1e-5) for d in diffs] )

        ############### PART 3 ####################
        ### (testing drop, repackage and rename) ##
        ###########################################
        datahandler.drop("group1") # drop empty group
        datahandler.drop("group4") # drop non-empty group
        datahandler.drop("group2", "ds1") # drop dataset
        datahandler.drop("group2", "ds2", repackage=True) # drop dataset and repackage
        datahandler.repackage() # repackage
        
        # check if groups/datasets are gone
        with h5py.File(datahandler.get_filepath(), 'r') as h5f:
            still_exist = ["group1" in list(h5f.keys()),
                           "group4" in list(h5f.keys()),
                           "ds1" in list(h5f["group2"].keys()),
                           "ds2" in list(h5f["group2"].keys())
                           ]
        assert not any(still_exist)

        # test rename
        datahandler.rename(group3="group3_new")
        datahandler.rename(group="group3_new", ds1d="ds1d_new")

        # check if names were changed and values stayed the same
        with h5py.File(datahandler.get_filepath(), 'r') as h5f:
            assert not "group3" in list(h5f.keys())
            assert "group3_new" in list(h5f.keys())
            assert not "ds1d" in list(h5f["group3_new"].keys())
            assert "ds1d_new" in list(h5f["group3_new"].keys())

        d = datahandler.get("group3_new", "ds1d_new") - np.array([d1, 10*d1])
        assert np.all(np.abs(d) < 1e-5)

        # test info methods again after manipulation
        print(str(datahandler))
        repr(datahandler)
        datahandler.keys()
        datahandler.content()
 
    # The correct functioning of EventIterator is tested in the respective test file
    def test_event_iterator(self, datahandler, testdata_1D_2D_3D_s_mus): 
        *_ , d3, s, mus = testdata_1D_2D_3D_s_mus

        # Prepare datasets
        datahandler.set("iterator_testing", event=d3)
        datahandler.set(group="iterator_testing", time_s=s, time_mus=mus, dtype=np.int32)

        it1 = datahandler.get_event_iterator("iterator_testing")
        it2 = datahandler.get_event_iterator("iterator_testing", batch_size=13)
        it3 = datahandler.get_event_iterator("iterator_testing", 0)
        it4 = datahandler.get_event_iterator("iterator_testing", 0, batch_size=13)

        for it in [it1, it2]:
            datahandler.include_event_iterator("iterator_testing_out", it)
            assert datahandler.get("iterator_testing_out", "event").shape == (2, 100, RECORD_LENGTH)
            datahandler.drop("iterator_testing_out")

        for it in [it3, it4]:
            datahandler.include_event_iterator("iterator_testing_out", it)
            assert datahandler.get("iterator_testing_out", "event").shape == (1, 100, RECORD_LENGTH)
            datahandler.drop("iterator_testing_out")

        with pytest.raises(Exception): # already existing dataset
            it = datahandler.get_event_iterator("iterator_testing")
            datahandler.include_event_iterator("iterator_testing", it)