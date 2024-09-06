import pytest
import tempfile

import numpy as np
import cait as ai
from cait.versatile import SEV, NPS, OF

from ..fixtures import datahandler, tempdir, testdata_1D_2D_3D_s_mus

@pytest.fixture(scope="module")
def dh(datahandler, testdata_1D_2D_3D_s_mus):
    _, _, d3, *_ = testdata_1D_2D_3D_s_mus
    datahandler.set("events", event=d3)
    datahandler.set("noise", event=d3)
    datahandler.calc_mp()
    datahandler.calc_sev()
    datahandler.calc_nps()
    datahandler.calc_of()

    yield datahandler

def basic_checks(dh, obj, k):
    # Indexing
    n_channels = obj.ndim
    assert obj[:].ndim == n_channels
    assert obj[list(range(n_channels))].ndim == n_channels
    if n_channels > 1: assert obj[0].ndim == 1

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
    obj.show(backend="plotly")
    obj.show(dt_us=5, backend="plotly")
    appendix = f"_{obj.__class__.__name__}_{k}"
    obj.to_file(fname="test"+appendix, out_dir=dh.get_filedirectory())
    obj.__class__.from_file(fname="test"+appendix, src_dir=dh.get_filedirectory())
    obj.to_dh(dh, group=f"test_group"+appendix, dataset="test_ds")
    obj.__class__.from_dh(dh, group=f"test_group"+appendix, dataset="test_ds")

def test_SEV(dh, testdata_1D_2D_3D_s_mus):
    d1, d2, *_ = testdata_1D_2D_3D_s_mus
    # Empty creation
    sev1 = SEV()

    # Creation from arrays
    sev2 = SEV(d2.copy())
    sev3 = SEV(d1.copy())
    sev4 = SEV(d1.flatten().copy())
    assert np.array_equal(sev3, sev4)

    # Check consistency with vanilla cait
    sev5 = SEV.from_dh(dh)

    # Creation from iterator
    it = dh.get_event_iterator("events")
    sev6 = SEV(it)
    sev7 = SEV(it[0])
    assert np.array_equal(sev6[0], sev7)

    it = dh.get_event_iterator("events", batch_size=13)
    sev8 = SEV(it)
    sev9 = SEV(it[0])
    assert np.array_equal(sev6, sev8)
    assert np.array_equal(sev7, sev9)

    basic_checks(dh, sev2, 0)
    basic_checks(dh, sev3, 1)
    basic_checks(dh, sev4, 2)
    basic_checks(dh, sev5, 3)
    basic_checks(dh, sev6, 4)
    basic_checks(dh, sev7, 5)
    basic_checks(dh, sev2[0], 6)
    basic_checks(dh, sev5[0], 7)

def test_NPS(dh, testdata_1D_2D_3D_s_mus):
    d1, d2, *_ = testdata_1D_2D_3D_s_mus
    # Empty creation
    nps1 = NPS()

    # Creation from arrays
    nps2 = NPS(d2.copy())
    nps3 = NPS(d1.copy())
    nps4 = NPS(d1.flatten().copy())
    assert np.array_equal(nps3, nps4)

    # Check consistency with vanilla cait
    nps5 = NPS.from_dh(dh)

    # Creation from iterator
    it = dh.get_event_iterator("noise")
    nps6 = NPS(it)
    nps7 = NPS(it[0])
    assert np.array_equal(nps6[0], nps7)

    it = dh.get_event_iterator("noise", batch_size=13)
    nps8 = NPS(it)
    nps9 = NPS(it[0])
    assert np.array_equal(nps6, nps8)
    assert np.array_equal(nps7, nps9)

    basic_checks(dh, nps2, 0)
    basic_checks(dh, nps3, 1)
    basic_checks(dh, nps4, 2)
    basic_checks(dh, nps5, 3)
    basic_checks(dh, nps6, 4)
    basic_checks(dh, nps7, 5)
    basic_checks(dh, nps2[0], 6)
    basic_checks(dh, nps5[0], 7)

def test_OF(dh, testdata_1D_2D_3D_s_mus):
    d1, d2, *_ = testdata_1D_2D_3D_s_mus
    # Empty creation
    of1 = OF()

    # Creation from arrays
    of2 = OF(d2.copy())
    of3 = OF(d1.copy())
    of4 = OF(d1.flatten().copy())
    assert np.array_equal(of3, of4)

    # Check consistency with vanilla cait
    of5 = OF.from_dh(dh)

    # Creation from NPS/SEV
    of6 = OF(SEV.from_dh(dh), NPS.from_dh(dh))
    of7 = OF(SEV(dh.get_event_iterator("events")), NPS(dh.get_event_iterator("noise")))
    of8 = OF(NPS(dh.get_event_iterator("noise")), SEV(dh.get_event_iterator("events")))
    assert np.array_equal(of7, of8)

    basic_checks(dh, of2, 0)
    basic_checks(dh, of3, 1)
    basic_checks(dh, of4, 2)
    basic_checks(dh, of5, 3)
    basic_checks(dh, of6, 4)
    basic_checks(dh, of7, 5)
    basic_checks(dh, of8, 6)
    basic_checks(dh, of2[0], 9)
    basic_checks(dh, of5[0], 10)
    basic_checks(dh, of8[0], 11)