import json

import numpy as np
import pytest

import cait as ai
import cait.versatile as vai
from cait.versatile.datasources import PARFile

from ..fixtures import tempdir

LENGTH = 100


@pytest.fixture(scope="module")
def stream_csmpl(tempdir):
    data = ai.data.TestData(
        filepath=tempdir.name + "/mock_001",
        duration=LENGTH,
        channels=[0, 1],
        sample_frequency=25000,
        start_s=13,
        tpas=[20, -1.0, 20, 0.3, 0, 20, 0.5, 1, 20, 3, -1, 20, 0, 10],
    )
    data.generate()

    yield vai.Stream(
        "csmpl",
        [
            tempdir.name + "/mock_001_Ch0.csmpl",
            tempdir.name + "/mock_001_Ch1.csmpl",
            tempdir.name + "/mock_001.test_stamps",
            tempdir.name + "/mock_001.dig_stamps",
            tempdir.name + "/mock_001.par",
        ],
    )


def basic_checks(stream):
    # Basic indexing
    k = stream.keys[0]
    stream[k]
    stream[k, :10]
    stream[k, -1]
    stream[k, 10:20, "as_voltage"]

    # Methods
    len(stream)
    stream.get_channel(k)
    stream.get_trace(k, slice(10, 30), voltage=True)
    stream.get_trace(k, slice(10, 30), voltage=False)
    stream.start_us
    stream.dt_us
    stream.time
    it1 = stream.get_event_iterator(k, 100, inds=[10, 20, 30])
    it2 = stream.get_event_iterator(k, 100, timestamps=stream.time[[10, 20, 30]])
    assert np.array_equal(next(iter(it1)), next(iter(it2)))


def test_multi_channel(stream_csmpl):
    ch0 = stream_csmpl["Ch0"]
    ch1 = stream_csmpl["Ch1"]
    mch = stream_csmpl[["Ch0", "Ch1"]]

    assert len(mch) == len(ch0)
    assert np.array_equal(mch[:100][0], ch0[:100])
    assert np.array_equal(mch[:100][1], ch1[:100])
    assert mch[:100].shape == (2, 100)

    # Check potential errors when printing
    print(ch0)
    print(mch)


def test_csmpl_StreamTime(stream_csmpl):
    t = stream_csmpl.time
    t[0]
    t[-1]
    t[0:10]
    t[[1, 3, 10]]
    ts = t[1000]
    dt = t.timestamp_to_datetime(ts)
    assert t.timestamp_to_ind(ts) == 1000
    assert t.datetime_to_timestamp(dt) == ts


def test_csmpl_basic(
    tempdir, stream_csmpl
):  # stream_csmpl needed for file initialization
    s1 = vai.Stream(
        "csmpl", [tempdir.name + "/mock_001_Ch0.csmpl", tempdir.name + "/mock_001.par"]
    )

    s2 = vai.Stream(
        "csmpl",
        [
            tempdir.name + "/mock_001_Ch0.csmpl",
            tempdir.name + "/mock_001_Ch1.csmpl",
            tempdir.name + "/mock_001.test_stamps",
            tempdir.name + "/mock_001.dig_stamps",
            tempdir.name + "/mock_001.par",
        ],
    )

    # Not available if not provided via test_stamps
    with pytest.raises(KeyError):
        s1.tpas
    with pytest.raises(KeyError):
        s1.tp_timestamps

    s2.tpas
    s2.tp_timestamps
    it = s2.get_event_iterator("mock_001_Ch0", 2**13, timestamps=s2.tp_timestamps["0"])
    assert len(s2.tpas["0"]) == len(s2.tp_timestamps["0"])
    assert len(it) == len(s2.tp_timestamps["0"])

    basic_checks(s1)
    basic_checks(s2)

def test_par_json(tempdir):
    # Write json file to tempdir
    parfile = PARFile(tempdir.name+"/mock_001.par")
    json_data = {
            "start_s": parfile.start_s,
            "start_us": parfile.start_us,
            "time_base_us": parfile.time_base_us,
            }
    with open(tempdir.name+"/mock_001.json", "w") as f:
        json.dump(json_data, f)

    stream_json = vai.Stream("csmpl", [tempdir.name+'/mock_001_Ch0.csmpl',
                               tempdir.name+'/mock_001_Ch1.csmpl',
                               tempdir.name+'/mock_001.test_stamps',
                               tempdir.name+'/mock_001.dig_stamps',
                               tempdir.name+'/mock_001.json'])

    stream_par = vai.Stream("csmpl", [tempdir.name+'/mock_001_Ch0.csmpl',
                               tempdir.name+'/mock_001_Ch1.csmpl',
                               tempdir.name+'/mock_001.test_stamps',
                               tempdir.name+'/mock_001.dig_stamps',
                               tempdir.name+'/mock_001.par'])

    assert len(stream_json) == len(stream_par)
    assert stream_json.start_us == stream_par.start_us
    assert stream_json.dt_us == stream_par.dt_us


# TODO
def test_VDAQ2(): ...


# TODO
def test_VDAQ3():
    ...
