import pytest

from cait.versatile import apply, MockData, BoxCarSmoothing, Downsample, OptimumFiltering, RemoveBaseline, TukeyFiltering, CalcMP, FitBaseline

RECORD_LENGTH = 2**14
N_EVENTS = 100

mock = MockData(record_length=RECORD_LENGTH, n_events=N_EVENTS)
it1 = mock.get_event_iterator(batch_size=1)
it2 = mock.get_event_iterator(batch_size=13)
it3 = mock.get_event_iterator(batch_size=N_EVENTS-1)

bcs = BoxCarSmoothing()
ds = Downsample(2)
rmbl = RemoveBaseline()
tf = TukeyFiltering()
calcmp = CalcMP()

calcmp_scalar = CalcMP(dt_us=mock.dt_us)
fbl = FitBaseline()

@pytest.mark.parametrize("fnc", [bcs, ds, rmbl, tf, calcmp])
def test_batches_processing(fnc):
    # Double channel
    out1 = apply(fnc, it1)
    out2 = apply(fnc, it2)
    out3 = apply(fnc, it3)

    assert out1.shape == out2.shape
    assert out2.shape == out3.shape

    # Single channel
    out1 = apply(fnc, it1[0])
    out2 = apply(fnc, it2[0])
    out3 = apply(fnc, it3[0])

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

@pytest.mark.parametrize("fnc", [calcmp_scalar, fbl])
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