"""Tests vai.OptimumFiltering and vai.OFPulseHeight in more detail."""

import numpy as np
import pytest

import cait.versatile as vai


@pytest.mark.parametrize("method", ["circular", "linear_pad"])
def test_filtering_circ_lin_pad(method):
    md_short = vai.MockData(n_events=113, record_length=2**12, dt_us=10)
    md_long = vai.MockData(n_events=113, record_length=2**13, dt_us=10)

    it_short = md_short.get_event_iterator()
    it_long = md_long.get_event_iterator()
    of = md_short.of

    f1 = vai.OptimumFiltering(of[0], method=method)
    f2 = vai.OptimumFiltering(of, method=method)

    # Single channel (short)
    expected_shapes = [
        (it_short[0].grab(0), (it_short.record_length,)),
        (it_short[0].grab([0, 1, 2]), (3, it_short.record_length)),
    ]
    for ev, shape in expected_shapes:
        assert f1(ev).shape == shape

    # Single channel (long)
    expected_shapes = [
        (it_long[0].grab(0), (it_long.record_length,)),
        (it_long[0].grab([0, 1, 2]), (3, it_long.record_length)),
    ]
    for ev, shape in expected_shapes:
        if method == "circular":
            with pytest.raises(ValueError):
                f1(ev)
        else:
            assert f1(ev).shape == shape

    # Multi channel (short)
    expected_shapes = [
        (it_short.grab(0), (it_short.n_channels, it_short.record_length)),
        (it_short.grab([0, 1, 2]), (3, it_short.n_channels, it_short.record_length)),
    ]
    for ev, shape in expected_shapes:
        assert f2(ev).shape == shape

    # Multi channel (long)
    expected_shapes = [
        (it_long.grab(0), (it_long.n_channels, it_long.record_length)),
        (it_long.grab([0, 1, 2]), (3, it_long.n_channels, it_long.record_length)),
    ]
    for ev, shape in expected_shapes:
        if method == "circular":
            with pytest.raises(ValueError):
                f2(ev)
        else:
            assert f2(ev).shape == shape

def test_filtering_lin():
    record_length = 2**14
    md_short = vai.MockData(n_events=113, record_length=record_length, dt_us=10)
    of = md_short.of

    s = vai.MockStream(rate_Hz=1, seed=137)
    it_short = s.get_event_iterator(["Ch0", "Ch1"], record_length=record_length, inds=[2**16, 2**17, 2**18, 2**19])
    it_long = it_short.with_extended_window()

    f1 = vai.OptimumFiltering(of[0], method="linear")
    f2 = vai.OptimumFiltering(of, method="linear")

    # Raise error when 
    for ev in [
        it_short[0].grab(0),
        it_short[0].grab([0, 1, 2]),
    ]:
        with pytest.raises(ValueError):
            f1(ev)
    for ev in [
        it_short.grab(0),
        it_short.grab([0, 1, 2]),
    ]:
        with pytest.raises(ValueError):
            f2(ev)

    # Single channel
    expected_shapes = [
        (it_long[0].grab(0), (record_length,)),
        (it_long[0].grab([0, 1, 2]), (3, record_length)),
    ]
    for ev, shape in expected_shapes:
        assert f1(ev).shape == shape

    # Multi channel
    expected_shapes = [
        (it_long.grab(0), (it_long.n_channels, record_length)),
        (it_long.grab([0, 1, 2]), (3, it_long.n_channels, record_length)),
    ]
    for ev, shape in expected_shapes:
        assert f2(ev).shape == shape

    # Check if the maxima are aligned for 'circular' and 'linear' with stream extended windows.
    ev1 = np.zeros_like(it_short[0].grab(0))
    ev2 = np.zeros_like(it_long[0].grab(0))

    ev1[:] = md_short.sev[0]
    ev2[record_length:2*record_length] = md_short.sev[0]

    ind1 = np.argmax(vai.OptimumFiltering(of[0], method="circular")(ev1))
    ind2 = np.argmax(vai.OptimumFiltering(of[0], method="linear")(ev2))
    assert ind1 == ind2