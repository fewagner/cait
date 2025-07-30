import numpy as np
import pytest

import cait as ai
import cait.versatile as vai
from cait.serialize import dump, dumps, load, loads

from .fixtures import (RDT_LENGTH, RECORD_LENGTH, SAMPLE_FREQUENCY,
                       datahandler, tempdir, testdata_1D_2D_3D_s_mus)


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

    dh = ai.DataHandler(channels=[0, 1])
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
        
    