from functools import partial

import numpy as np
import pytest
import scipy as sp

from cait.versatile import (BoxCarSmoothing, CalcMP, Downsample, FitBaseline,
                            MainParameters, MockData, OptimumFiltering,
                            Preview, RemoveBaseline, TriggerSurvival,
                            TukeyWindow, apply, trigger_of, trigger_zscore)
from cait.versatile.iterators import PulseSimIterator

RECORD_LENGTH = 2**14
N_EVENTS = 100

mock = MockData(record_length=RECORD_LENGTH, n_events=N_EVENTS)
it1 = mock.get_event_iterator(batch_size=1)
it2 = mock.get_event_iterator(batch_size=13)
it3 = mock.get_event_iterator(batch_size=N_EVENTS-1)

bcs = BoxCarSmoothing()
ds = Downsample(2)
rmbl = RemoveBaseline()
tf = TukeyWindow()
calcmp = CalcMP()

calcmp_scalar = CalcMP(dt_us=mock.dt_us)
fbl = FitBaseline()
mp = MainParameters()
mp_scalar = MainParameters(dt_us=mock.dt_us)

@pytest.mark.parametrize("fnc", [bcs, ds, rmbl, tf, calcmp, mp])
def test_batches_processing(fnc):
    # Double channel
    out1 = np.array(apply(fnc, it1))
    out2 = np.array(apply(fnc, it2))
    out3 = np.array(apply(fnc, it3))

    assert out1.shape == out2.shape
    assert out2.shape == out3.shape

    # Single channel
    out1 = np.array(apply(fnc, it1[0]))
    out2 = np.array(apply(fnc, it2[0]))
    out3 = np.array(apply(fnc, it3[0]))

    assert out1.shape == out2.shape
    assert out2.shape == out3.shape

def test_batches_of():
    of1 = OptimumFiltering(mock.of[0])
    of2 = OptimumFiltering(mock.of)

    # Double channel
    out1 = apply(of2, it1)
    out2 = apply(of2, it2)
    out3 = apply(of2, it3)

    assert out1.shape == out2.shape
    assert out2.shape == out3.shape

    # Single channel
    out1 = apply(of1, it1[0])
    out2 = apply(of1, it2[0])
    out3 = apply(of1, it3[0])

    assert out1.shape == out2.shape
    assert out2.shape == out3.shape

@pytest.mark.parametrize("fnc", [calcmp_scalar, fbl, mp_scalar])
def test_batches_scalar(fnc):
    # Double channel
    out1 = apply(fnc, it1)
    out2 = apply(fnc, it2)
    out3 = apply(fnc, it3)

    assert len(out1) == len(out2)
    assert len(out2) == len(out3)

    # Single channel
    out1 = apply(fnc, it1[0])
    out2 = apply(fnc, it2[0])
    out3 = apply(fnc, it3[0])

    assert len(out1) == len(out2)
    assert len(out2) == len(out3)

def test_trigger_survival():
    mock = MockData(record_length=RECORD_LENGTH)
    sev, of = mock.sev[0], mock.of[0]
    it = MockData(record_length=6*RECORD_LENGTH).get_event_iterator()[0]

    padded_sev = np.zeros(6*RECORD_LENGTH)
    padded_sev[3*RECORD_LENGTH:4*RECORD_LENGTH] = np.array(sev)

    sim_phs = sp.stats.uniform.rvs(size=len(it))
    it2 = PulseSimIterator(it, sev=padded_sev, pulse_heights=sim_phs)

    f1 = TriggerSurvival(
        trigger_fnc=partial(trigger_of, of=of, threshold=0.1),
        target_ind=np.argmax(padded_sev),
        tolerance_samples=10
    )
    f2 = TriggerSurvival(
        trigger_fnc=partial(trigger_zscore, record_length=RECORD_LENGTH),
        target_ind=np.argmax(padded_sev),
        tolerance_samples=10
    )

    Preview(it2, f1, backend="plotly")
    Preview(it2, f2, backend="plotly")

    out1 = np.array(apply(f1, it2))
    out2 = np.array(apply(f1, it2.with_batchsize(13)))
    out3 = np.array(apply(f2, it2))
    out4 = np.array(apply(f2, it2.with_batchsize(13)))

    assert all(x.shape==y.shape for x,y in zip(out1, out2))
    assert all(x.shape==y.shape for x,y in zip(out3, out4))