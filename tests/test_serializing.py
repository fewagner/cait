import numpy as np
import pytest

import cait as ai
import cait.versatile as vai
from cait.data import combine_h5, merge_h5
from cait.serialize import dump, dumps, load, loads
from cait.versatile.iterators.impl_h5 import H5Iterator

from .fixtures import (RDT_LENGTH, RECORD_LENGTH, SAMPLE_FREQUENCY,
                       datahandler, tempdir)


def load_unload(obj):
    assert isinstance(dump(obj), dict)
    assert isinstance(dumps(obj), str)
    assert isinstance(load(dump(obj)), obj.__class__)
    assert isinstance(loads(dumps(obj)), obj.__class__)

@pytest.fixture(scope="module")
def testdata(tempdir):
    data = ai.data.TestData(filepath=tempdir.name+'/mock_001',
                            duration=RDT_LENGTH,
                            record_length=RECORD_LENGTH,
                            sample_frequency=SAMPLE_FREQUENCY,
                            channels=[0, 1],
                            start_s=13)
    data.generate()
    stream = vai.Stream('csmpl', [tempdir.name+'/mock_001_Ch0.csmpl',
                               tempdir.name+'/mock_001_Ch1.csmpl',
                               tempdir.name+'/mock_001.test_stamps',
                               tempdir.name+'/mock_001.dig_stamps',
                               tempdir.name+'/mock_001.par'])
    rdt_file = vai.RDTFile(tempdir.name+'/mock_001.rdt')

    dh = ai.DataHandler(channels=[0, 1], record_length=RECORD_LENGTH)
    dh.set_filepath(tempdir.name, "mock_001")
    dh.init_empty()
    dh.include_event_iterator("events", rdt_file[(0, 1)].get_event_iterator())

    yield stream, rdt_file, dh

def test_datasources(testdata):
    stream, rdt_file, dh = testdata
    mock = vai.MockData()

    for obj in [
        stream,
        rdt_file,
        rdt_file[0],
        rdt_file[(0, 1)],
        dh,
        mock,
    ]:
        load_unload(obj)

    with pytest.raises(AttributeError):
        # Should raise error if filepath not set
        dump(ai.DataHandler(channels=[0, 1]))

    with pytest.raises(KeyError):
        # Missing 'kwargs' key
        load({"class": "DataHandler", "args": []})

    with pytest.raises(KeyError):
        # Unknown class
        load({"class": "SomeUnknownClass", "args": [], "kwargs": {}})

    with pytest.raises(TypeError):
        # String is not a dictionary
        loads("not_a_dictionary")

def test_iterators(testdata):
    stream, rdt_file, dh = testdata
    mock = vai.MockData()

    stream_it1 = stream.get_event_iterator("Ch0", 100, [1000, 2000, 3000, 4000])
    stream_it2 = stream.get_event_iterator("Ch0", 100, np.array([1000, 2000, 3000, 4000]))
    rdt_it = rdt_file[(0,1)].get_event_iterator()
    dh_it = dh.get_event_iterator("events")
    mock_it = mock.get_event_iterator(batch_size=20)

    for obj in [
        stream_it1,
        stream_it2,
        stream_it1 + stream_it2,
        rdt_it,
        dh_it,
        mock_it,
    ]:
        load_unload(obj)

        assert len(obj) == len(load(dump(obj)))
        assert len(obj) == len(loads(dumps(obj)))

        for attr in ["dt_us", "record_length", "uses_batches", "ds_start_us", "n_channels"]:
            assert getattr(obj, attr) == getattr(load(dump(obj)), attr)
            assert getattr(obj, attr) == getattr(loads(dumps(obj)), attr)
        
def test_externally_store_iterator_in_datahandler(testdata):
    stream, rdt_file, dh = testdata
    mock = vai.MockData(record_length=RECORD_LENGTH)

    stream_it = stream.get_event_iterator("Ch0", 
                                          RECORD_LENGTH, 
                                          [3*RECORD_LENGTH, 5*RECORD_LENGTH, 10*RECORD_LENGTH, 13*RECORD_LENGTH]
                                          )
    rdt_it = rdt_file[(0,1)].get_event_iterator()
    dh_it = dh.get_event_iterator("events")
    mock_it = mock.get_event_iterator(batch_size=13)

    # Check if reference remains even after dataset is dropped
    dh.include_event_iterator("external_storage_test_group1", rdt_it, copy_events=True)
    it = dh.get_event_iterator("external_storage_test_group1")
    assert isinstance(it, H5Iterator)
    dh.drop("external_storage_test_group1", "event")
    it = dh.get_event_iterator("external_storage_test_group1")
    assert isinstance(it, rdt_it.__class__)
    # Check if accessing events via dh.get() works
    dh.get(f"external_storage_test_group1", "event")

    # Check if reference is deleted when entire group is dropped
    dh.include_event_iterator("external_storage_test_group2", rdt_it, copy_events=True)
    dh.get_event_iterator("external_storage_test_group2")
    dh.drop("external_storage_test_group2")
    with pytest.raises(KeyError):
        dh.get_event_iterator("external_storage_test_group2")

    # Check if reference works if data is not copied for multiple iterators
    for i, it in enumerate([stream_it, rdt_it, dh_it, mock_it, stream_it+stream_it], start=3):
        dh.include_event_iterator(f"external_storage_test_group{i}", it, copy_events=False)
        recovered_it = dh.get_event_iterator(f"external_storage_test_group{i}")
        assert isinstance(recovered_it, it.__class__)

        # Check if accessing events via dh.get() works and gives the correct shape
        assert (it.n_channels, len(it), it.record_length) == dh.get(f"external_storage_test_group{i}", "event").shape

def test_combine_h5(tempdir):
    h5_fnames = [
        "combine_test1", 
        "combine_test2", 
        "combine_test3", 
        "combine_test4",
        "combine_test5",
        "combine_test6"]
    its = [
        vai.MockData(n_events=113, record_length=RECORD_LENGTH).get_event_iterator(),
        vai.MockData(n_events=213, record_length=RECORD_LENGTH).get_event_iterator(),
        vai.MockData(n_events=313, record_length=RECORD_LENGTH).get_event_iterator()[0],
        vai.MockData(n_events=413, record_length=2*RECORD_LENGTH).get_event_iterator(),
        vai.RDTFile(tempdir.name+'/mock_001.rdt')[(0,1)].get_event_iterator()[0],
        vai.RDTFile(tempdir.name+'/mock_001.rdt')[(0,1)].get_event_iterator()[1]
    ]
    for fname, it in zip(h5_fnames, its):
        dh = ai.DataHandler(channels=[0, 1])
        dh.set_filepath(tempdir.name, fname, appendix=False)
        dh.init_empty()
        dh.include_event_iterator("events", it, copy_events=False)

    # Does 'combining' single file still work?
    # MockIterator (cannot be checked for 
    # values because mock data currently is still random every time)
    combine_h5(
        fname="combined_test1",
        files=[h5_fnames[0]],
        src_dir=tempdir.name,
        out_dir=tempdir.name,
        groups_combine=["events"],
    )

    # Does 'combining' single file still work?
    # RDTIterator (this can be checked further)
    combine_h5(
        fname="combined_test2",
        files=[h5_fnames[4]],
        src_dir=tempdir.name,
        out_dir=tempdir.name,
        groups_combine=["events"],
    )
    # Does it 'combine' correctly?
    dh = ai.DataHandler(channels=[0])
    dh.set_filepath(tempdir.name, "combined_test2", appendix=False)
    it = dh.get_event_iterator("events")
    assert len(it) == len(its[4]), "Combined iterator has wrong length"
    for ev1, ev2 in zip(its[4], it):
        assert np.array_equal(ev1, ev2), "Combined iterator returned wrong events."

    # Does combining consistent files work?
    # MockIterator
    combine_h5(
        fname="combined_test3",
        files=h5_fnames[:2],
        src_dir=tempdir.name,
        out_dir=tempdir.name,
        groups_combine=["events"],
    )

    # Does combining consistent files work?
    # RDTIterator
    combine_h5(
        fname="combined_test4",
        files=h5_fnames[4:6],
        src_dir=tempdir.name,
        out_dir=tempdir.name,
        groups_combine=["events"],
    )
    # Does it combine correctly?
    dh = ai.DataHandler(channels=[0])
    dh.set_filepath(tempdir.name, "combined_test4", appendix=False)
    it = dh.get_event_iterator("events")
    assert len(it) == len(its[4]+its[5]), "Combined iterator has wrong length"
    for ev1, ev2 in zip(its[4]+its[5], it):
        assert np.array_equal(ev1, ev2), "Combined iterator returned wrong events."

    # Does combining INconsistent files raise Exception?
    # (inconsistent number of channels)
    with pytest.raises(AssertionError):
        combine_h5(
            fname="combined_test5",
            files=h5_fnames[:3],
            src_dir=tempdir.name,
            out_dir=tempdir.name,
            groups_combine=["events"],
        )

    # Does combining INconsistent files raise Exception?
    # (inconsistent record length)
    with pytest.raises(AssertionError):
        combine_h5(
            fname="combined_test6",
            files=h5_fnames[:2] + [h5_fnames[3]],
            src_dir=tempdir.name,
            out_dir=tempdir.name,
            groups_combine=["events"],
        )

def test_merge_h5(tempdir):
    h5_fnames = [
        "merge_test1", 
        "merge_test2",
    ]
    its = [
        vai.RDTFile(tempdir.name+'/mock_001.rdt')[(0,1)].get_event_iterator()[0],
        vai.RDTFile(tempdir.name+'/mock_001.rdt')[(0,1)].get_event_iterator()[1]
    ]
    for fname, it in zip(h5_fnames, its):
        dh = ai.DataHandler(channels=[0, 1])
        dh.set_filepath(tempdir.name, fname, appendix=False)
        dh.init_empty()
        dh.include_event_iterator("events", it, copy_events=False)

    # Does 'merging' single file work?
    merge_h5(
        fname="merged_test1",
        files=[h5_fnames[0]],
        src_dir=tempdir.name,
        out_dir=tempdir.name,
        groups_merge=["events"],
    )
    # Does it 'merge' correctly?
    dh = ai.DataHandler(channels=[0])
    dh.set_filepath(tempdir.name, "merged_test1", appendix=False)
    it = dh.get_event_iterator("events")
    assert len(it) == len(its[0]), "Merged iterator has wrong length"
    for ev1, ev2 in zip(its[0], it):
        assert np.array_equal(ev1, ev2), "Merged iterator returned wrong events."

    # Does merging multiple files work?
    merge_h5(
        fname="merged_test2",
        files=h5_fnames[:2],
        src_dir=tempdir.name,
        out_dir=tempdir.name,
        groups_merge=["events"],
    )
    # Does it merge correctly?
    dh = ai.DataHandler(channels=[0])
    dh.set_filepath(tempdir.name, "merged_test2", appendix=False)
    it = dh.get_event_iterator("events")
    assert len(it) == len(its[0]+its[1]), "Combined iterator has wrong length"
    for ev1, ev2 in zip(its[0]+its[1], it):
        assert np.array_equal(ev1, ev2), "Combined iterator returned wrong events."