import numpy as np
import pytest
import scipy as sp

import cait as ai
import cait.versatile as vai

from ..fixtures import tempdir

LENGTH = 10
RECORD_LENGTH = 2**14

#####################################
######## GENERATE TEST DATA #########
#####################################
stream = vai.MockStream(
    seed=137, 
    rate_Hz=2,
    pulse_shape=[[0.5, 0.5, 0.3, 0.1, 10.0], [0.3, 0.5, 0.3, 0.01, 4.0], [0.9, 0.5, 0.3, 0.01, 10.0]],
    tp_shape=[[0.5, 0.5, 0.3, 0.01, 20.0], [0.5, 0.5, 0.3, 0.01, 10.0], [0.8, 0.5, 0.3, 0.01, 100.0]],
    baseline_sig=[0.005, 0.001, 0.0002]
    )
md = vai.MockData()
sev, of = md.sev, md.of
t = md.get_event_iterator().t

sev = vai.SEV([sev[0], sev[1], sev[1]], dt_us=sev.dt_us)

sev_fitpars = [
    [0, 0.5, 0.5, 0.3*stream.dt_us, 0.1*stream.dt_us, 1*stream.dt_us], 
    [0, 0.5, 0.5, 0.3, 0.1, 10.0],
    [0, 0.9, 0.5, 0.3, 0.1, 10.0],
]

fit_sev = [ai.fit.pulse_template(t, *pars) for pars in sev_fitpars]
for i in range(len(sev_fitpars)):
    sev_fitpars[i][0] -= t[np.argmax(fit_sev[0])]

sim_ts = sp.stats.randint.rvs(
    stream.time[0], stream.time[-1], size=LENGTH, random_state=42,
    )
sim_phs = [
    sp.stats.uniform.rvs(size=len(sim_ts), random_state=43),
    sp.stats.uniform.rvs(size=len(sim_ts), random_state=44),
    sp.stats.uniform.rvs(size=len(sim_ts), random_state=45),
]
shifts = [
    np.zeros(len(sim_ts)),
    sp.stats.randint.rvs(-300, 300, size=len(sim_ts)),
    sp.stats.randint.rvs(-10, 10, size=len(sim_ts)),
]

#####################################
############## TESTS ################
#####################################
@pytest.fixture(scope="module")
def dh_test(tempdir):
    dh = ai.DataHandler(record_length=RECORD_LENGTH, nmbr_channels=len(stream.keys))
    dh.set_filepath(path_h5=tempdir.name, fname="test_efficiency", appendix=False)
    dh.init_empty()

    yield dh

@pytest.mark.parametrize(
        "kwargs", 
        [
            dict(trigger_channels="Ch0", 
                 testpulse_channels="TP0", 
                 tag="test1"),
            dict(trigger_channels="Ch0", 
                 testpulse_channels="TP0",
                 shift_samples=shifts[-1],
                 tag="test2"),
            dict(trigger_channels=["Ch0", "Ch1"], 
                 testpulse_channels=["TP0", "TP1"],
                 shift_samples=shifts[:2],
                 tag="test3"),
            dict(trigger_channels="Ch0",
                 passive_channels=["Ch1"],
                 testpulse_channels=["TP0", "TP1"],
                 shift_samples=shifts[:2],
                 tag="test4"),
            dict(trigger_channels=["Ch0", "Ch1"],
                 passive_channels=["Ch2"],
                 testpulse_channels=["TP0", "TP1", "TP2"],
                 shift_samples=shifts,
                 tag="test5"),
        ]
)
def test_trigger_efficiency_of(dh_test, kwargs):
    # Infer the number of trigger and passive channels.
    # This is used to slice the correct number of channels
    # from sev/sim_phs/of/sev_fitpars (otherwise we'd have
    # to specify all of them in the parametrize).
    tc = kwargs["trigger_channels"]
    pc = kwargs.get("passive_channels", [])
    tc = [tc] if isinstance(tc, str) else tc
    pc = [pc] if isinstance(pc, str) else pc

    n_trig = len(tc)
    n_tot = len(tc) + len(pc)

    # With SEV input
    dh_test.efficiency_sim_trigger_of(
        stream=stream, 
        threshold=0.1,
        sim_ts=sim_ts,
        sim_phs=sim_phs[:n_tot],
        of=of[:n_trig],
        sev=sev[:n_tot],
        tolerance_samples=10,
        n_record_lens=8,
        record_placement=3,
        **kwargs,
    )

    # With SEV fitpars input
    kwargs["tag"] += ".1"
    dh_test.efficiency_sim_trigger_of(
        stream=stream, 
        threshold=0.1,
        sim_ts=sim_ts,
        sim_phs=sim_phs[:n_tot],
        of=of[:n_trig],
        sev_fitpars=sev_fitpars[:n_tot],
        tolerance_samples=10,
        n_record_lens=8,
        record_placement=3,
        **kwargs,
    )

@pytest.mark.parametrize(
        "kwargs", 
        [
            dict(trigger_channels=[("Ch0", "Ch1")], 
                 tag="test2dof1",
                 sim_phs=sim_phs[:2],
                 of=of,
                 sev=sev[:2],
                 ),
            dict(trigger_channels=[("Ch0", "Ch1")],
                 testpulse_channels=["TP0", "TP1"],
                 tag="test2dof2",
                 sim_phs=sim_phs[:2],
                 of=of,
                 sev=sev[:2],
                 ),
            dict(trigger_channels=[("Ch0", "Ch1")],
                 testpulse_channels=["TP0", "TP1"],
                 tag="test2dof3",
                 sim_phs=sim_phs[:2],
                 shift_samples=shifts[:2],
                 of=of,
                 sev=sev[:2],
                 ),
            dict(trigger_channels=[("Ch0", "Ch1")],
                 passive_channels="Ch2",
                 testpulse_channels=["TP0", "TP1", "TP2"],
                 tag="test2dof4",
                 sim_phs=sim_phs,
                 shift_samples=shifts,
                 of=of,
                 sev=sev,
                 ),
        ]
)
def test_trigger_efficiency_of_2dof(dh_test, kwargs):
    dh_test.efficiency_sim_trigger_of(
        stream=stream, 
        threshold=0.1,
        sim_ts=sim_ts,
        tolerance_samples=10,
        n_record_lens=8,
        record_placement=3,
        **kwargs,
    )

@pytest.mark.parametrize(
        "kwargs", 
        [
            dict(trigger_channels="Ch0", 
                 testpulse_channels="TP0", 
                 sim_ts=sim_ts,
                 sim_phs=sim_phs,
                 of=of[0],
                 sev=sev[0],
                 tag="test_error1"),
            dict(trigger_channels="Ch0", 
                 testpulse_channels="TP0",
                 sim_ts=sim_ts[1:],
                 sim_phs=sim_phs[0],
                 of=of[0],
                 sev=sev[0],
                 tag="test_error2"),
            dict(trigger_channels=["Ch0", "Ch1"], 
                 testpulse_channels=["TP0", "TP1"],
                 sim_ts=sim_ts,
                 sim_phs=sim_phs,
                 of=of[0],
                 sev=sev,
                 tag="test_error3"),
            dict(trigger_channels=["Ch0", "Ch1"], 
                 testpulse_channels=["TP0", "TP1"],
                 sim_ts=sim_ts,
                 sim_phs=sim_phs,
                 of=of,
                 sev=sev[0],
                 tag="test_error4"),
            dict(trigger_channels=["Ch0", "Ch1"], 
                 testpulse_channels=["TP0", "TP1"],
                 sim_ts=sim_ts,
                 sim_phs=sim_phs,
                 of=of,
                 sev=sev,
                 shift_samples=shifts[0],
                 tag="test_error5"),
            dict(trigger_channels=["Ch0", "Ch1"], 
                 testpulse_channels=["TP0", "TP1"],
                 sim_ts=sim_ts,
                 sim_phs=sim_phs,
                 of=of,
                 sev=sev,
                 shift_samples=np.array(shifts)[:,1:],
                 tag="test_error6"),
        ]
)
def test_trigger_efficiency_of_errors(dh_test, kwargs):
    # With SEV input
    with pytest.raises(ValueError):
        dh_test.efficiency_sim_trigger_of(
            stream=stream, 
            threshold=0.1,
            tolerance_samples=10,
            n_record_lens=8,
            record_placement=3,
            **kwargs,
        )