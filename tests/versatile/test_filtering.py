"""Tests vai.OptimumFiltering and vai.OFPulseHeight in more detail."""

import numpy as np
import pytest

import cait.versatile as vai


# Test vai.OptimumFiltering
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

    # Test apply, also for batched iterator
    vai.apply(f1, it_short[0])
    vai.apply(f1, it_short[0].with_batchsize(7))
    vai.apply(f2, it_short)
    vai.apply(f2, it_short.with_batchsize(7))

    if method == "linear_pad":
        vai.apply(f1, it_long[0])
        vai.apply(f1, it_long[0].with_batchsize(7))
        vai.apply(f2, it_long)
        vai.apply(f2, it_long.with_batchsize(7))

# Test vai.OptimumFiltering
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

# Test vai.OFPulseHeight
def test_ofpulseheight_docstring_example():
    # Just check that they are running in different configurations.
    # From the docstring example:
    record_length = 2**14

    md = vai.MockData(record_length=record_length)
    it = md.get_event_iterator()
    sev, of = md.sev, md.of

    f1 = vai.OFPulseHeight(
        of=of[0],
        sev=sev[0],
    )

    vai.Preview(it[0], f1, backend="plotly")

    vai.apply(f1, it[0])
    of_res = vai.apply(f1, it[0].with_batchsize(7))
    of_res_dict = {k: v for k, v in zip(f1.names, of_res)}
    # ------------------------- #

    f2 = vai.OFPulseHeight(
        of=of,
        sev=sev,
        relative_to=[None, 0],
        max_search=[(0.2, 0.4), (-300, -100)],
    )
    vai.apply(f2, it)
    vai.apply(f2, it.with_batchsize(7))

    vai.Preview(it, f2, backend="plotly")
    # ------------------------- #

    s = vai.MockStream(
        rate_Hz=1,
        seed=137,
        pulse_shape=[[0.5, 0.5, 0.3, 0.1, 10.0], [0.3, 0.5, 0.3, 0.01, 4.0], [0.3, 0.5, 0.3, 0.01, 4.0]],
        tp_shape=[[0.5, 0.5, 0.3, 0.01, 20.0], [0.5, 0.5, 0.3, 0.01, 10.0], [0.5, 0.5, 0.3, 0.01, 10.0]],
        baseline_sig=[0.005, 0.001, 0.002],
    )
    stream_it = s.get_event_iterator(
        ["Ch0", "Ch1", "Ch2"],
        record_length=record_length,
        inds=[(4+x)*record_length for x in range(10)],
    )
    f3 = vai.OFPulseHeight(
        of=of,
        sev=sev,
        method="linear",
    )
    vai.apply(f3, stream_it[:2].with_extended_window())
    vai.apply(f3, stream_it[:2].with_extended_window().with_batchsize(7))

    vai.Preview(stream_it[:2].with_extended_window(), f3, backend="plotly")
    # ------------------------- #

    of2d_norm = np.max(vai.OptimumFiltering2D([of[0], of[1]])(np.array([sev[0], sev[1]])))
    f4 = vai.OFPulseHeight(
        of=[of[0]/of2d_norm, of[1]/of2d_norm, of[1]],
        sev=[sev[0], sev[1], sev[1]],
        filter_groups=[(0, 1), 2],
        relative_to=[None, 0],
        max_search=[(0.2, 0.4), (-0.1, 0)],
    )
    vai.apply(f4, stream_it)
    vai.apply(f4, stream_it.with_batchsize(7))

    vai.Preview(stream_it, f4, backend="plotly")
    # ------------------------- #

    f5 = vai.OFPulseHeight(
        of=[of[0]/of2d_norm, of[1]/of2d_norm],
        sev=[sev[0], sev[1]],
        filter_groups=[(0, 1)],
        relative_to=[None],
        max_search=[(0., 1.)],
    )
    vai.apply(f5, stream_it[:2])
    vai.apply(f5, stream_it[:2].with_batchsize(7))

    vai.Preview(stream_it[:2], f5, backend="plotly")

# Test vai.OFPulseHeight
# Check if evaluation for various function configurations
# returns ~0 RMS when evaluated on the (shifted) SEV.
def test_rms():
    record_length = 2**14
    md = vai.MockData(record_length=record_length, n_events=10)
    it = md.get_event_iterator()
    sev, of = md.sev, md.of

    s = vai.MockStream(
        rate_Hz=1,
        seed=137,
        pulse_shape=[[0.5, 0.5, 0.3, 0.1, 10.0], [0.3, 0.5, 0.3, 0.01, 4.0], [0.3, 0.5, 0.3, 0.01, 4.0]],
        tp_shape=[[0.5, 0.5, 0.3, 0.01, 20.0], [0.5, 0.5, 0.3, 0.01, 10.0], [0.5, 0.5, 0.3, 0.01, 10.0]],
        baseline_sig=[0.005, 0.001, 0.002],
    )
    it_stream = s.get_event_iterator(
        ["Ch0", "Ch1", "Ch2"],
        record_length=record_length,
        inds=[(4+x)*record_length for x in range(10)],
    )

    f1 = vai.OFPulseHeight(
        of=of[0],
        sev=sev[0],
    )
    f2 = vai.OFPulseHeight(
        of=of,
        sev=sev,
        relative_to=[None, 0],
        max_search=[(0.2, 0.4), (-300, -100)],
    )
    f3 = vai.OFPulseHeight(
        of=of,
        sev=sev,
        method="linear",
    )
    of2d_norm = np.max(vai.OptimumFiltering2D([of[0], of[1]])(np.array([sev[0], sev[1]])))
    f4 = vai.OFPulseHeight(
        of=[of[0]/of2d_norm, of[1]/of2d_norm, of[1]],
        sev=[sev[0], sev[1], sev[1]],
        filter_groups=[(0, 1), 2],
        relative_to=[None, 0],
        max_search=[(0.2, 0.4), (-0.1, 0)],
    )
    f5 = vai.OFPulseHeight(
        of=[of[0]/of2d_norm, of[1]/of2d_norm],
        sev=[sev[0], sev[1]],
        filter_groups=[(0, 1)],
        relative_to=[None],
        max_search=[(0.2, 0.4)],
    )

    for f, ev in [
        (f1, sev[0]),
        (f2, sev),
        (f3, np.pad(sev, [(0, 0), (record_length, record_length)])),
        (f4, np.array([sev[0], sev[1], sev[1]])),
        (f5, sev),
    ]:
        for shift in [-13, -10, -5, 0, 5, 10, 13]:
            out = f(np.roll(ev, shift, axis=-1))
            of_dict = {k: v for k, v in zip(f.names, out)}
            assert np.allclose([of_dict["of_rms"], of_dict["of_peak_rms"]], 0)