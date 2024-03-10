# THIS FILE TESTS IF ALL FUNCTIONS IN A GENERAL WORKFLOW WORK.
# THE TESTS ARE NOT VERY DETAILED BUT SHOULD CATCH MAJOR ISSUES INTRODUCED BY UPDATED 3RD-PARTY DEPENDENCIES
import pytest
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings

import cait as ai
import cait.versatile as vai

from .fixtures import tempdir
from .fixtures import datahandler_testdata as dh

# Suppress pop up plots during tests
plt.switch_backend("Agg")
# This will raise a warning which we suppress (because we know it should be raised)
@pytest.mark.filterwarnings('ignore:Matplotlib is currently using agg')
@pytest.mark.filterwarnings('ignore:FigureCanvasAgg is non-interactive')
def test_workflow(dh, tempdir):
    # Calculate MP and additional MP
    dh.calc_mp("events")
    dh.calc_mp("noise")
    dh.calc_mp("testpulses")
    dh.calc_additional_mp("events", no_of=True)
    dh.calc_additional_mp("noise", no_of=True)
    dh.calc_additional_mp("testpulses", no_of=True)

    # Create quick cut for SEV
    sev_cuts = ai.cuts.LogicalCut()
    sev_cuts.add_condition(dh.get('events', 'rise_time', 0) > 1.3)
    sev_cuts.add_condition(dh.get('events', 'pulse_height', 0) < 4)
    print('Survived: {}/{}, {} %'.format(sev_cuts.counts(), sev_cuts.total(), sev_cuts.counts()/sev_cuts.total()))

    # Calculate SEV
    dh.calc_sev("events", use_idx=sev_cuts.get_idx())
    dh.show_sev()
    dh.calc_sev("testpulses")

    # Calculate NPS
    dh.calc_nps()
    dh.show_nps()   

    # Calculate OF
    dh.calc_of()
    dh.calc_of(name_appendix='_tp')
    dh.show_of()

    # Estimate optimal trigger threshold
    dh.apply_of("noise")
    dh.estimate_trigger_threshold(channel=0, detector_mass=20e-3, sigma_x0=1, xran=(0,10), xran_hist=(0,10))
    dh.estimate_trigger_threshold(channel=1, detector_mass=10e-3, sigma_x0=0.5, xran=(0,10), xran_hist=(0,10))

    # Triggering
    dh_stream = ai.DataHandler(channels=[0,1])
    dh_stream.set_filepath(path_h5=tempdir.name, fname='stream_001')
    dh_stream.init_empty()
    dh_stream.include_metainfo(path_par=tempdir.name+'/mock_001.par')

    # We also directly include the SEV, NPS and OF that we just calculated in the new DataHandler as well for later use
    vai.SEV().from_dh(dh).to_dh(dh_stream)
    vai.SEV().from_dh(dh, group="stdevent_tp").to_dh(dh_stream, group="stdevent_tp")
    vai.NPS().from_dh(dh).to_dh(dh_stream)
    vai.OF().from_dh(dh).to_dh(dh_stream)
    vai.OF().from_dh(dh, group="optimumfilter_tp").to_dh(dh_stream, group="optimumfilter_tp")

    of = vai.OF().from_dh(dh_stream)

    # Those are the thresholds we just found above
    thresholds = [3.061e-3, 2.391e-3]

    # Perform actual triggering
    dh_stream.include_csmpl_triggers(
        csmpl_paths=[tempdir.name+'/mock_001_Ch0.csmpl', 
                     tempdir.name+'/mock_001_Ch1.csmpl'],
        thresholds=thresholds, 
        of=of,
        path_dig=tempdir.name+'/mock_001.dig_stamps')

    # Includes information needed to distinguish testpulses from particle hits
    dh_stream.include_test_stamps(
        path_teststamps=tempdir.name+'/mock_001.test_stamps',
        path_dig_stamps=tempdir.name+'/mock_001.dig_stamps',
        fix_offset=True)

    # Includes voltage traces
    dh_stream.include_triggered_events(
        csmpl_paths=[tempdir.name+'/mock_001_Ch0.csmpl', 
                     tempdir.name+'/mock_001_Ch1.csmpl'], 
        exclude_tp=True)
    
    dh_stream.calc_mp("events")
    dh_stream.calc_mp("testpulses")
    dh_stream.calc_additional_mp("events")
    dh_stream.calc_additional_mp("testpulses")

    dh_stream.calc_testpulse_stability(channel=0)
    dh_stream.calc_testpulse_stability(channel=1)

    dh_stream.apply_of("events")
    dh_stream.apply_of("testpulses")

    # Energy calibration
    pm = dh_stream.calc_calibration(starts_saturation=[5, 5], 
                         cpe_factor=[1,1], 
                         only_stable=True,
                         method='of', 
                         return_pulser_models=True, 
                         plot=False)
    
    pm[0].plot()

    tpa_equiv_heights = dh_stream.get("events", "tpa_equivalent")

    # Restrict fit to smaller interval
    fit_lower, fit_upper = 0.7, 1.2
    flag_phonon = np.logical_and(tpa_equiv_heights[0]>fit_lower, tpa_equiv_heights[0]<fit_upper)
    flag_light = np.logical_and(tpa_equiv_heights[1]>fit_lower, tpa_equiv_heights[1]<fit_upper)

    # Use scipy to fit (this will be part of cait in the near future)
    mu_phonon, sigma_phonon = sp.stats.norm.fit(tpa_equiv_heights[0, flag_phonon])
    mu_light, sigma_light = sp.stats.norm.fit(tpa_equiv_heights[1, flag_light])

    # Calculate CPE factors (assuming 55Fe calibration source)
    cpe_factor_phonon = 5.9/mu_phonon
    cpe_factor_light = 5.9/mu_light

    tpa_equiv_heights[0,:] = cpe_factor_phonon*tpa_equiv_heights[0,:]
    tpa_equiv_heights[1,:] = cpe_factor_light*tpa_equiv_heights[1,:]
    dh_stream.set("events", recoil_energy=tpa_equiv_heights, overwrite_existing=True)

    # BASELINE RESOLUTION
    # Required before superimposing pulses on baselines
    dh.calc_bl_coefficients()

    # Create new HDF5 file with simulated events
    dh.simulate_pulses(path_sim=dh.get_filedirectory() + '/resolution.h5', 
                    size_events=2000, 
                    reuse_bl=True,
                    ev_discrete_phs=[[1],[1]], 
                    t0_interval=[-1, 1])

    # Create DataHandler for this HDF5 file
    dh_res = ai.DataHandler(channels=[0,1])
    dh_res.set_filepath(dh.get_filedirectory(), 'resolution', appendix=False)

    # Show contents of DataHandler
    dh_res.content()

    # Calculate MP and apply OF
    dh_res.calc_mp()
    dh_res.calc_additional_mp()
    dh_res.apply_of(first_channel_dominant=True)

    dh_res.calc_calibration(starts_saturation=[5, 5], 
                         cpe_factor=[cpe_factor_phonon, cpe_factor_light], 
                         only_stable=True,
                         method='of', 
                         pulser_models=pm, 
                         plot=False)
    
    recoil_energies = dh_res.get("events", "recoil_energy")

    pars_phonon = sp.stats.norm.fit(recoil_energies[0])
    pars_light = sp.stats.norm.fit(recoil_energies[1])

    print(f"Baseline Resolution (Phonon): {1000*pars_phonon[1]:.2f} eVee")
    print(f"Baseline Resolution (Light): {1000*pars_light[1]:.2f} eVee")

    # EFFICIENCY
    n_events = 1000
    ph_max = 0.05
    # Create new HDF5 file with simulated events
    dh.simulate_pulses(path_sim=dh.get_filedirectory() + '/efficiency.h5', 
                    size_events=n_events, 
                    reuse_bl=True,
                    ev_discrete_phs=[np.linspace(0, ph_max, n_events), np.linspace(0, ph_max, n_events)], 
                    t0_interval=[-1, 1]
                    )

    # Create DataHandler for this HDF5 file
    dh_eff = ai.DataHandler(channels=[0,1])
    dh_eff.set_filepath(dh.get_filedirectory(), 'efficiency', appendix=False)

    # Calculate all MP and apply OF (otherwise, we could not perform the same cuts if those quantities are not calculated)
    dh_eff.calc_mp()
    dh_eff.calc_additional_mp()
    dh_eff.apply_of(first_channel_dominant=True)

    # Calculate recoil energies like before (only difference here is method="true_ph" because for this simulation we actually
    # know the correct pulse height)
    dh_eff.calc_calibration(starts_saturation=[5,5], 
                            cpe_factor=[cpe_factor_phonon, cpe_factor_light], 
                            method='true_ph', 
                            pulser_models=pm,  
                            return_pulser_models=False, 
                            plot=False,
                        )
    
    # Create cuts for trigger efficiency
    cuts_triggereff = ai.cuts.LogicalCut()
    # Optimum-filtered pulse heights of the phonon channel are above the trigger threshold
    cuts_triggereff.add_condition(dh_eff.get('events', 'of_ph', 0) > thresholds[0])

    print('Survived Trigger: {}/{}, {} %'.format(cuts_triggereff.counts(), cuts_triggereff.total(), 100*cuts_triggereff.counts()/cuts_triggereff.total()))

    # Create cuts for cut efficiency
    cuts_eff = ai.cuts.LogicalCut()
    # Apply exactly the same cuts as to the real data (above)
    cuts_eff.add_condition(dh_eff.get('events', 'rise_time', 0) > 1.3)
    cuts_eff.add_condition(dh_eff.get('events', 'pulse_height', 0) < 4)
    # Also include the trigger efficiency (this is a logical AND)
    cuts_eff.add_condition(cuts_triggereff.get_flag())

    print('Survived Cuts: {}/{}, {} %'.format(cuts_eff.counts(), cuts_eff.total(), 100*cuts_eff.counts()/cuts_eff.total()))

    bins = np.linspace(0, ph_max, 30)

    hist_triggereff, _ = np.histogram(dh_eff.get('events', 'recoil_energy', 0, cuts_triggereff.get_flag()), bins=bins)
    hist, _ = np.histogram(dh_eff.get('events', 'recoil_energy', 0, cuts_eff.get_flag()), bins=bins)
    hist_all, _ = np.histogram(dh_eff.get('events', 'recoil_energy', 0), bins=bins)

    threshold_pars = ai.fit.fit_trigger_efficiency(binned_energies=bins,
                                               survived_fraction=hist_triggereff/hist_all,
                                               a0_0=0.9,
                                               a1_0=0.01,
                                               a2_0=0.0005,
                                               plot=True,
                                               title='Trigger Efficiency',
                                               xlim=(0., ph_max))
    
    # WRITE XY FILES
    time = dh_stream.record_window()

    for i in range(dh_stream.nmbr_channels):

        ai.data.write_xy_file(filepath=dh.get_filedirectory() + '/Channel_{}_SEV_Particle.xy'.format(i),
                            data=[time, dh.get('stdevent', 'event')[i]],
                            title='Channel {} SEV Particle'.format(i),
                            axis=['Time (ms)', 'Amplitude (V)'])  

        ai.data.write_xy_file(filepath=dh.get_filedirectory() + '/Channel_{}_SEV_TP.xy'.format(i),
                            data=[time, dh.get('stdevent_tp', 'event')[i]],
                            title='Channel {} SEV TP'.format(i),
                            axis=['Time (ms)', 'Amplitude (V)'])
                                
        ai.data.write_xy_file(filepath=dh.get_filedirectory() + '/Channel_{}_NPS.xy'.format(i),
                            data=[dh.get('noise', 'freq'), dh.get('noise', 'nps')[i]],
                            title='Channel {} NPS'.format(i),
                            axis=['Frequency (Hz)', 'Amplitude (a.u.)'])
