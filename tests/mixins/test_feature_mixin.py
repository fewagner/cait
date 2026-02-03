import numpy as np
import pytest

import cait as ai
import cait.versatile as vai

from ..fixtures import tempdir_fnc

# Mock data
record_length = 2**14

md = vai.MockData(record_length=record_length)
it = md.get_event_iterator()
sev, of = md.sev, md.of

s = vai.MockStream(
    duration_h=0.1,
    rate_Hz=1,
    seed=137,
    pulse_shape=[
        [0.5, 0.5, 0.3, 0.1, 10.0], 
        [0.3, 0.5, 0.3, 0.01, 4.0], 
        [0.3, 0.5, 0.3, 0.01, 4.0],
    ],
    tp_shape=[
        [0.5, 0.5, 0.3, 0.01, 20.0], 
        [0.5, 0.5, 0.3, 0.01, 10.0], 
        [0.5, 0.5, 0.3, 0.01, 10.0],
    ],
    baseline_sig=[0.005, 0.001, 0.002],
)

def create_dh(filepath, name, record_length, sample_frequency, n_ch):
    dh = ai.DataHandler(
        record_length=record_length,
        sample_frequency=sample_frequency,
        nmbr_channels=n_ch,
    )
    dh.set_filepath(filepath, name, appendix=False)
    dh.init_empty()

    return dh

# Test dh.apply_ofilter
@pytest.mark.parametrize(
        "tag,batch_size,peak_rms_width,max_search", 
        [
            ("", 10, 5, (0.2, 0.4)),
            ("test1", 10, 5, (0.2, 0.4)),
            ("test2", 13, 50, (100, 200)),
            ("test3", 23, 0, (100, 0.5)),
            ("test4", 23, 0, 0.5),
            ("test5", 10, 1, 1000),
        ]
    )
def test_apply_ofilter_single_ch(
    tempdir_fnc, 
    tag, 
    batch_size,
    peak_rms_width,
    max_search,
    ):
    dh1 = create_dh(
        tempdir_fnc.name, 
        name="dh1",
        record_length=record_length, 
        sample_frequency=s.sample_frequency,
        n_ch=1,
    )

    dh1.trigger_zscore(
        s, 
        trigger_channels=["Ch0"],
        testpulse_channels=["TP0"],
    )
    n_ev = len(dh1.get_event_iterator("events"))
    n_tp = len(dh1.get_event_iterator("testpulses"))

    dh1.apply_ofilter(
        "events", 
        of=of[0], 
        sev=sev[0],
        tag=tag,
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        max_search=max_search,
    )
    dh1.apply_ofilter(
        "testpulses", 
        of=of[0], 
        sev=sev[0],
        tag=tag,
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        max_search=max_search,
        with_processing=vai.RemoveBaseline(),
    )
    ds = f"of_ph-{tag}" if tag else "of_ph"
    
    assert dh1[f"events/{ds}"].shape == (1, n_ev)
    assert dh1[f"testpulses/{ds}"].shape == (1, n_tp)

    dh1.apply_ofilter(
        "events", 
        of=of[0], 
        sev=sev[0],
        tag=tag+"_on_stream",
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        max_search=max_search,
        on_stream=True,
    )
    ds = f"of_ph-{tag+'_on_stream'}" if tag else "of_ph-_on_stream"
    assert dh1[f"events/{ds}"].shape == (1, n_ev)

    dh2 = create_dh(
        tempdir_fnc.name, 
        name="dh2",
        record_length=record_length, 
        sample_frequency=s.sample_frequency,
        n_ch=1,
    )
    dh3 = create_dh(
        tempdir_fnc.name, 
        name="dh3",
        record_length=record_length, 
        sample_frequency=s.sample_frequency,
        n_ch=1,
    )
    for dh in [dh2, dh3]:
        dh.trigger_zscore(
            s, 
            trigger_channels=["Ch0"],
            testpulse_channels=["TP0"],
        )

    ai.data.combine_h5(
        "combined", 
        ["dh2", "dh3"], 
        src_dir=tempdir_fnc.name,
        out_dir=tempdir_fnc.name,
        groups_combine=["events", "testpulses"],
    )
    dh_combined = ai.DataHandler(
        record_length=dh1.record_length,
        sample_frequency=dh1.sample_frequency,
        nmbr_channels=1
    )
    dh_combined.set_filepath(tempdir_fnc.name, "combined", appendix=False)

    n_ev = len(dh_combined.get_event_iterator("events"))
    n_tp = len(dh_combined.get_event_iterator("testpulses"))
    
    dh_combined.apply_ofilter(
        "events", 
        of=of[0], 
        sev=sev[0],
        tag=tag+"_combined",
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        max_search=max_search,
        on_stream=True,
    )
    dh_combined.apply_ofilter(
        "events", 
        of=of[0], 
        sev=sev[0],
        tag=tag+"_combined_with_processing",
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        max_search=max_search,
        on_stream=True,
        with_processing=[vai.RemoveBaseline()],
    )

# Test dh.apply_ofilter
@pytest.mark.parametrize(
        "tag,batch_size,peak_rms_width,max_search,relative_to", 
        [
            ("", 10, 5, (0.2, 0.4), None),
            ("test1", 10, 5, [(0.2, 0.4), -10], [None, 0]),
            ("test2", 13, 50, [-100, (0.2, 0.4)], [1, None]),
        ]
    )
def test_apply_ofilter_double_ch(
    tempdir_fnc, 
    tag, 
    batch_size,
    peak_rms_width,
    max_search,
    relative_to,
    ):
    # One datahandler that is used to store triggered data.
    dh1 = create_dh(
        tempdir_fnc.name, 
        name="dh1",
        record_length=record_length, 
        sample_frequency=s.sample_frequency,
        n_ch=2,
    )

    dh1.trigger_zscore(
        s, 
        trigger_channels=["Ch0"],
        passive_channels=["Ch1"],
        testpulse_channels=["TP0", "TP1"],
    )
    n_ev = len(dh1.get_event_iterator("events"))
    n_tp = len(dh1.get_event_iterator("testpulses"))

    # Apply filter to both channels (events).
    dh1.apply_ofilter(
        "events", 
        of=[of[0], of[1]], 
        sev=[sev[0], sev[1]],
        tag=tag,
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        max_search=max_search,
        relative_to=relative_to,
    )
    # Apply filter to both channels (testpulses).
    dh1.apply_ofilter(
        "testpulses", 
        of=[of[0], of[1]], 
        sev=[sev[0], sev[1]],
        tag=tag,
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        max_search=max_search,
        relative_to=relative_to,
        with_processing=[vai.RemoveBaseline()]
    )
    ds = f"of_ph-{tag}" if tag else "of_ph"
    
    assert dh1[f"events/{ds}"].shape == (2, n_ev)
    assert dh1[f"testpulses/{ds}"].shape == (2, n_tp)

    # Apply filter to both channels (extended to stream).
    dh1.apply_ofilter(
        "events", 
        of=[of[0], of[1]], 
        sev=[sev[0], sev[1]],
        tag=tag+"_on_stream",
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        max_search=max_search,
        on_stream=True,
        relative_to=relative_to,
    )
    ds = f"of_ph-{tag+'_on_stream'}" if tag else "of_ph-_on_stream"
    assert dh1[f"events/{ds}"].shape == (2, n_ev)

    # Apply filter to channel 0 only.
    dh1.apply_ofilter(
        "events", 
        of=of[0], 
        sev=sev[0],
        tag=tag+"_single0",
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        only_channels=0,
    )
    ds = f"of_ph-{tag+'_single0'}" if tag else "of_ph-_single0"
    assert dh1[f"events/{ds}"].shape == (2, n_ev)
    assert np.allclose(dh1[f"events/{ds}"][1], -404)

    # Apply filter to channel 1 only.
    dh1.apply_ofilter(
        "events", 
        of=of[1], 
        sev=sev[1],
        tag=tag+"_single1",
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        only_channels=1,
    )
    ds = f"of_ph-{tag+'_single1'}" if tag else "of_ph-_single1"
    assert dh1[f"events/{ds}"].shape == (2, n_ev)
    assert np.allclose(dh1[f"events/{ds}"][0], -404)

    # Create two new datahandlers to check if combination also works
    dh2 = create_dh(
        tempdir_fnc.name, 
        name="dh2",
        record_length=record_length, 
        sample_frequency=s.sample_frequency,
        n_ch=2,
    )
    dh3 = create_dh(
        tempdir_fnc.name, 
        name="dh3",
        record_length=record_length, 
        sample_frequency=s.sample_frequency,
        n_ch=2,
    )
    for dh in [dh2, dh3]:
        dh.trigger_zscore(
            s, 
            trigger_channels=["Ch0"],
            passive_channels=["Ch1"],
            testpulse_channels=["TP0", "TP1"],
        )

    ai.data.combine_h5(
        "combined", 
        ["dh2", "dh3"], 
        src_dir=tempdir_fnc.name,
        out_dir=tempdir_fnc.name,
        groups_combine=["events", "testpulses"],
    )
    dh_combined = ai.DataHandler(
        record_length=dh1.record_length,
        sample_frequency=dh1.sample_frequency,
        nmbr_channels=2,
    )
    dh_combined.set_filepath(tempdir_fnc.name, "combined", appendix=False)

    n_ev = len(dh_combined.get_event_iterator("events"))
    n_tp = len(dh_combined.get_event_iterator("testpulses"))
    
    dh_combined.apply_ofilter(
        "events", 
        of=[of[0], of[1]], 
        sev=[sev[0], sev[1]],
        tag=tag+"_combined",
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        max_search=max_search,
        on_stream=True,
        relative_to=relative_to,
    )
    dh_combined.apply_ofilter(
        "events", 
        of=[of[0], of[1]], 
        sev=[sev[0], sev[1]],
        tag=tag+"_combined_with_processing",
        batch_size=batch_size,
        peak_rms_width=peak_rms_width,
        max_search=max_search,
        on_stream=True,
        relative_to=relative_to,
        with_processing=vai.RemoveBaseline(),
    )

# Test dh.apply_ofilter
def test_apply_ofilter_triple_ch(tempdir_fnc):
    dh = create_dh(
        tempdir_fnc.name, 
        name="dh1",
        record_length=record_length, 
        sample_frequency=s.sample_frequency,
        n_ch=3,
    )

    dh.trigger_zscore(
        s, 
        trigger_channels=["Ch0"],
        passive_channels=["Ch1", "Ch2"],
    )
    n_ev = len(dh.get_event_iterator("events"))

    # All three channels independently.
    dh.apply_ofilter(
        "events", 
        of=[of[0], of[1], of[1]], 
        sev=[sev[0], sev[1], sev[1]],
        tag="independent",
        max_search=(0.2, 0.4),
        relative_to=None,
    )
    out = dh["events/of_ph-independent"]
    assert out.shape == (3, n_ev)

    # First two 2d, last relative.
    dh.apply_ofilter(
        "events", 
        of=[of[0], of[1], of[1]], 
        sev=[sev[0], sev[1], sev[1]],
        tag="2d_1d",
        max_search=[(0.2, 0.4), 100],
        relative_to=[None, 0],
        filter_groups=[(0, 1), 2]
    )
    out = dh["events/of_ph-2d_1d"]
    assert out.shape == (3, n_ev)
    assert np.array_equal(out[0], out[1])
    assert not np.array_equal(out[1], out[2])

    # Only last two 2d.
    dh.apply_ofilter(
        "events", 
        of=[of[0], of[1]], 
        sev=[sev[0], sev[1]],
        tag="2d",
        filter_groups=[(0, 1)],
        only_channels=[1, 2],
    )
    out = dh["events/of_ph-2d"]
    assert out.shape == (3, n_ev)
    assert np.array_equal(out[1], out[2])
    assert np.allclose(out[0], -404)

    # First and last (inverted order), second relative.
    dh.apply_ofilter(
        "events", 
        of=[of[0], of[1], of[1]], 
        sev=[sev[0], sev[1], sev[1]],
        tag="2d_chaos",
        filter_groups=[1, (2, 0)],
        relative_to=[1, None],
        max_search=[-13, (0.2, 0.4)],
    )
    out = dh["events/of_ph-2d_chaos"]
    assert out.shape == (3, n_ev)
    assert np.array_equal(out[0], out[2])
    assert not np.array_equal(out[0], out[1])

    # Only single channel.
    dh.apply_ofilter(
        "events", 
        of=of[0], 
        sev=sev[0],
        tag="single",
        only_channels=2,
    )
    out = dh["events/of_ph-single"]
    assert out.shape == (3, n_ev)
    assert np.array_equal(out[0], out[1])
    assert np.allclose(out[0], -404)
    assert not np.allclose(out[2], -404)