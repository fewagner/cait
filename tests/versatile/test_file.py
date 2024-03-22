import pytest
import tempfile
import numpy as np
import h5py

import cait as ai
import cait.versatile as vai
from cait.versatile.functions.file import check_file_consistency

from ..fixtures import tempdir, testdata_1D_2D_3D_s_mus, RECORD_LENGTH, SAMPLE_FREQUENCY

DATA_2D_single_CH = np.random.rand(1, 100)
DATA_3D_single_CH = np.random.rand(1, 100, 16)

N_FILES = 5 # for both combine and merge

def relative_res(a, b):
    return np.abs((a-b)/a)

@pytest.fixture(scope="module")
def dhs(tempdir, testdata_1D_2D_3D_s_mus):
    d1, d2, d3, *_ = testdata_1D_2D_3D_s_mus

    dhs = list()
    for n in range(2*N_FILES): # N_FILES for both combine and merge separately
        dhs.append(ai.DataHandler(record_length=RECORD_LENGTH,
                                  sample_frequency=SAMPLE_FREQUENCY,
                                  nmbr_channels=2))

        dhs[-1].set_filepath(path_h5=tempdir.name, fname=f"file{n}", appendix=False)
        dhs[-1].init_empty()

        # to have consistent file content for both sets of files
        # (i.e. for both merge and combine)
        exp = np.floor(n/2) 

        # write data to files
        dhs[-1].set(group="group1", ds1=d1*10**exp, ds2=d2*10**exp, ds3=d3*10**exp)
        dhs[-1].set(group="group2", ds1=d1*10**(2*exp), ds2=d2*10**(2*exp), ds3=d3*10**(2*exp))
        dhs[-1].set(group="testgroup", ds=np.ones_like(d1))

    return dhs

def validate(name, tempdir, testdata_1D_2D_3D_s_mus):
    d1, d2, d3, *_ = testdata_1D_2D_3D_s_mus
    # check if merge was successful
    dh = ai.DataHandler(record_length=RECORD_LENGTH, 
                        sample_frequency=SAMPLE_FREQUENCY,
                        nmbr_channels=2)
    dh.set_filepath(path_h5=tempdir.name, fname=name, appendix=False)

    # check if shapes are as expected
    assert dh.get("group1", "ds1").shape == (d1.shape[0]*N_FILES,)
    assert dh.get("group1", "ds2").shape == (d2.shape[0], d2.shape[1]*N_FILES)
    assert dh.get("group1", "ds3").shape == (d3.shape[0], d3.shape[1]*N_FILES, d3.shape[2])

    # check content
    diffs = list()
    diffs.append( relative_res(dh.get("group1", "ds1"), np.hstack([d1*10**n for n in range(N_FILES)])) )
    diffs.append( relative_res(dh.get("group1", "ds2"), np.hstack([d2*10**n for n in range(N_FILES)])) )
    diffs.append( relative_res(dh.get("group1", "ds3"), np.hstack([d3*10**n for n in range(N_FILES)])) )

    diffs.append( relative_res(dh.get("group2", "ds1"), np.hstack([d1*10**(2*n) for n in range(N_FILES)])) )
    diffs.append( relative_res(dh.get("group2", "ds2"), np.hstack([d2*10**(2*n) for n in range(N_FILES)])) )
    diffs.append( relative_res(dh.get("group2", "ds3"), np.hstack([d3*10**(2*n) for n in range(N_FILES)])) )

    diffs.append( relative_res(dh.get("testgroup", "ds"), np.ones_like(d1)) )

    assert all( [np.all(d < 1e-5) for d in diffs] )

    return dh

def test_file_consistency(tempdir, testdata_1D_2D_3D_s_mus):
    d1, d2, d3, *_ = testdata_1D_2D_3D_s_mus
    # test if exception is raised in case the dtypes don't agree
    files = ["2combine1_dtype", "2combine2_dtype"]
    dtypes = ["float32", "float64"]
    for f, dtype in zip(files, dtypes):
        dh = ai.DataHandler(record_length=RECORD_LENGTH, 
                            sample_frequency=SAMPLE_FREQUENCY,
                            nmbr_channels=2)
        dh.set_filepath(path_h5=tempdir.name, fname=f, appendix=False)
        dh.init_empty()

        dh.set(group="group1", ds1=d1, ds2=d2, ds3=d3, dtype=dtype)
        dh.set(group="group2", ds1=d1, ds2=d2, ds3=d3, dtype=dtype)
        dh.set(group="testgroup", ds=np.ones_like(d1), dtype=dtype)

    with pytest.raises(AssertionError):
        check_file_consistency(files=files, 
                                src_dir=tempdir.name,
                                groups_combine=["group1","group2"], 
                                groups_include=["testgroup"])

    # test if exception is raised in case the shapes don't agree
    files = ["2combine1_shape", "2combine2_shape"]
    for f in files:
        dh = ai.DataHandler(record_length=RECORD_LENGTH, 
                            sample_frequency=SAMPLE_FREQUENCY,
                            nmbr_channels=2)
        dh.set_filepath(path_h5=tempdir.name, fname=f, appendix=False)
        dh.init_empty()

        if f == "2combine2_shape":
            dh.set(group="group1", ds1=d1, ds2=np.transpose(d2), ds3=d3)
        else:
            dh.set(group="group1", ds1=d1, ds2=d2, ds3=d3)
        dh.set(group="group2", ds1=d1, ds2=d2, ds3=d3)
        dh.set(group="testgroup", ds=np.ones_like(d1))

    with pytest.raises(AssertionError):
        check_file_consistency(files=files, 
                                src_dir=tempdir.name,
                                groups_combine=["group1","group2"], 
                                groups_include=["testgroup"])
        
    # test if exception is raised in case the group/dataset is not present
    files = ["2combine1_ds", "2combine2_ds"]
    for f in files:
        dh = ai.DataHandler(record_length=RECORD_LENGTH, 
                            sample_frequency=SAMPLE_FREQUENCY,
                            nmbr_channels=2)
        dh.set_filepath(path_h5=tempdir.name, fname=f, appendix=False)
        dh.init_empty()

        if f == "2combine2_ds":
            dh.set(group="group1", ds1=d1, ds2=d2, ds3=d3)
        if f == "2combine2_ds":
            dh.set(group="group2", ds1=d1, ds2=d2, ds3=d3)
        else:
            dh.set(group="group2", ds1=d1, ds3=d3)
        dh.set(group="testgroup", ds=np.ones_like(d1))

    with pytest.raises(AssertionError): # not all groups present in all files
        check_file_consistency(files=files, 
                                src_dir=tempdir.name,
                                groups_combine=["group1","group2"], 
                                groups_include=["testgroup"])
    with pytest.raises(AssertionError): # ds2 in group2 missing from second file
        check_file_consistency(files=files, 
                                src_dir=tempdir.name,
                                groups_combine=["group2"], 
                                groups_include=["testgroup"])
    with pytest.raises(AssertionError): # include group not present in either file
        check_file_consistency(files=files, 
                                src_dir=tempdir.name,
                                groups_combine=[], 
                                groups_include=["non_existent_group"])

def test_combine(tempdir, testdata_1D_2D_3D_s_mus, dhs):
    d1, d2, *_ = testdata_1D_2D_3D_s_mus

    # combine files
    files2combine = [dh.get_filename() for dh in dhs[::2]] # every second starting from 0th
    vai.combine(fname="output_combine",
                files=files2combine, 
                src_dir=tempdir.name,
                out_dir=tempdir.name,
                groups_combine=["group1","group2"],
                groups_include=["testgroup"])
    
    # validate shapes and dataset contents
    dh = validate("output_combine", tempdir, testdata_1D_2D_3D_s_mus)

    # check if all datasets are virtual
    with h5py.File(dh.get_filepath(), 'r') as h5f:
        for group in list(h5f.keys()):
            for ds in h5f[group]:
                assert h5f[group][ds].is_virtual

    # check set() method with virtual datasets
    # even with change_existing=True, warning should be given
    with pytest.warns(UserWarning):
        dh.set(group="group1", change_existing=True, ds1=d1)

    # even with overwrite_existing=True, warning should be given
    with pytest.warns(UserWarning):
        dh.set(group="group1", change_existing=True, overwrite_existing=True, ds1=d1)

    # same as above but in channel mode
    with pytest.warns(UserWarning):
        dh.set(group="group1", n_channels=2, channel=0, change_existing=True, ds2=d2[0,:])
    
    with pytest.warns(UserWarning):
        dh.set(group="group1", n_channels=2, channel=0, change_existing=True, overwrite_existing=True, ds2=d2[0,:])
    
    # test if writing works
    old_ds = dh.get("group1", "ds1")
    dh.set(group="group1", ds1=old_ds*10, write_to_virtual=True)
    assert np.all(relative_res(old_ds*10, dh.get("group1", "ds1")) < 1e-5)
    with h5py.File(dh.get_filepath(), 'r') as h5f:
        assert h5f["group1/ds1"].is_virtual

    # test if OVER-writing works
    dh.set(group="group1", ds1=old_ds*10, write_to_virtual=False)
    assert np.all(relative_res(old_ds*10, dh.get("group1", "ds1")) < 1e-5)
    with h5py.File(dh.get_filepath(), 'r') as h5f:
        assert not h5f["group1/ds1"].is_virtual

def test_merge(tempdir, testdata_1D_2D_3D_s_mus, dhs):
    # merge files
    files2merge = [dh.get_filename() for dh in dhs[1::2]] # every second starting from first
    vai.merge(fname="output_merge",
                files=files2merge, # every second starting from first
                src_dir=tempdir.name,
                out_dir=tempdir.name, 
                groups_merge=["group1","group2"], 
                groups_include=["testgroup"])
    
    # validate shapes and dataset contents
    dh = validate("output_merge", tempdir, testdata_1D_2D_3D_s_mus)

    # check if none of the datasets is virtual
    with h5py.File(dh.get_filepath(), 'r') as h5f:
        for group in list(h5f.keys()):
            for ds in h5f[group]:
                assert not h5f[group][ds].is_virtual