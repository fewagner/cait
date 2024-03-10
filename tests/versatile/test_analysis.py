import pytest
import numpy as np

import cait.versatile as vai

from ..fixtures import datahandler, tempdir, testdata_1D_2D_3D_s_mus

###### Functions used for test tests #######
def func1(ev):
    return np.max(ev)

def func2(ev):
    return (np.min(ev), np.max(ev), np.mean(ev))

def func1_multi(ev):
    return np.max(ev[0])

def func2_multi(ev):
    return (np.max(ev[0]), np.max(ev[1]))

def func3_multi(ev):
    return (np.max(ev[0]), np.max(ev[1]), np.mean(ev[0]))

def func4_multi(ev):
    return ([np.min(ev[0]), np.max(ev[0]), np.mean(ev[0])], [np.min(ev[1]), np.max(ev[1]), np.mean(ev[1])])

def func_events(ev):
    return ev

@pytest.fixture(scope="module")
def dh(datahandler, testdata_1D_2D_3D_s_mus):
    _, _, d3, *_ = testdata_1D_2D_3D_s_mus
    datahandler.set("events", event=d3)

    yield datahandler

def test_apply_scalar_out_single_ch(dh):
    it1 = dh.get_event_iterator("events", 0)
    it2 = dh.get_event_iterator("events", 0, batch_size=11)
    out1 = vai.apply(func1, it1)
    out2 = vai.apply(func1, it2)

    assert out1.shape == out2.shape
    assert out1.shape == (100,)

def test_apply_vector_out_single_ch(dh):
    it1 = dh.get_event_iterator("events", 0)
    it2 = dh.get_event_iterator("events", 0, batch_size=11)
    out11, out12, out13 = vai.apply(func2, it1)
    out21, out22, out23 = vai.apply(func2, it2)

    assert out11.shape == out21.shape
    assert out12.shape == out22.shape
    assert out13.shape == out23.shape
    assert out11.shape == (100,)
    assert out12.shape == (100,)
    assert out13.shape == (100,)

def test_apply_scalar_out_multi_ch(dh):
    it1 = dh.get_event_iterator("events")
    it2 = dh.get_event_iterator("events", batch_size=11)
    out1 = vai.apply(func1_multi, it1)
    out2 = vai.apply(func1_multi, it2)

    assert out1.shape == out2.shape
    assert out1.shape == (100, )

def test_apply_vector2_out_multi_ch(dh):
    it1 = dh.get_event_iterator("events")
    it2 = dh.get_event_iterator("events", batch_size=11)
    out11, out12 = vai.apply(func2_multi, it1)
    out21, out22 = vai.apply(func2_multi, it2)

    assert out11.shape == out21.shape
    assert out12.shape == out22.shape
    assert out11.shape == (100,)
    assert out12.shape == (100,)

def test_apply_vector3_out_multi_ch(dh):
    it1 = dh.get_event_iterator("events")
    it2 = dh.get_event_iterator("events", batch_size=11)
    out11, out12, out13 = vai.apply(func3_multi, it1)
    out21, out22, out23 = vai.apply(func3_multi, it2)

    assert out11.shape == out21.shape
    assert out12.shape == out22.shape
    assert out13.shape == out23.shape
    assert out11.shape == (100,)
    assert out12.shape == (100,)
    assert out13.shape == (100,)

def test_apply_tensor_out_multi_ch(dh):
    it1 = dh.get_event_iterator("events")
    it2 = dh.get_event_iterator("events", batch_size=11)
    out11, out12 = vai.apply(func4_multi, it1)
    out21, out22 = vai.apply(func4_multi, it2)

    assert out11.shape == out21.shape
    assert out12.shape == out22.shape
    assert out11.shape == (100, 3)
    assert out12.shape == (100, 3)

def test_apply_event_size_output(dh):
    it1 = dh.get_event_iterator("events")
    it2 = dh.get_event_iterator("events", batch_size=11)
    it3 = dh.get_event_iterator("events", 0)
    it4 = dh.get_event_iterator("events", 0, batch_size=11)
    out1 = vai.apply(func_events, it1)
    out2 = vai.apply(func_events, it2)
    out3 = vai.apply(func_events, it3)
    out4 = vai.apply(func_events, it4)

    assert out1.shape == out2.shape
    assert out3.shape == out4.shape
    assert out1.shape == (100, 2, dh.record_length)
    assert out3.shape == (100, dh.record_length)