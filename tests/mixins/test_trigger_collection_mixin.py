import pytest

import numpy as np
import cait as ai
import cait.versatile as vai

from ..fixtures import tempdir

LENGTH = 100

@pytest.fixture(scope="module")
def stream_csmpl(tempdir):
    data = ai.data.TestData(filepath=tempdir.name+'/test_trigger',
                            duration=LENGTH,
                            channels=[0, 1],
                            sample_frequency=25000,
                            start_s=13,
                            tpas=[20, -1.0, 20, 0.3, 0, 20, 0.5, 1, 20, 3, -1, 20, 0, 10])
    data.generate()

    yield vai.Stream('csmpl', [tempdir.name+'/test_trigger_Ch0.csmpl',
                                   tempdir.name+'/test_trigger_Ch1.csmpl',
                                   tempdir.name+'/test_trigger.test_stamps',
                                   tempdir.name+'/test_trigger.dig_stamps',
                                   tempdir.name+'/test_trigger.par'])

def test_trigger_zscore_errors(tempdir, stream_csmpl):
    dh = ai.DataHandler(nmbr_channels=2)
    dh.set_filepath(tempdir.name, "test_trigger_zscore_errors", appendix=False)
    dh.init_empty()

    # trigger_channel unavailable
    with pytest.raises(KeyError): 
        dh.trigger_zscore(stream_csmpl, 
                          trigger_channels="Ch2",
                          passive_channels="Ch1",
                          testpulse_channels=["0", "0"])
        
    # passive_channel unavailable
    with pytest.raises(KeyError): 
        dh.trigger_zscore(stream_csmpl, 
                          trigger_channels="Ch1",
                          passive_channels="Ch2",
                          testpulse_channels=["0", "0"])
        
    # tp_channel unavailable
    with pytest.raises(KeyError): 
        dh.trigger_zscore(stream_csmpl, 
                          trigger_channels="Ch1",
                          passive_channels="Ch0",
                          testpulse_channels=["0", "2"])
        
    # tp_channel don't add up
    with pytest.raises(ValueError): 
        dh.trigger_zscore(stream_csmpl, 
                          trigger_channels="Ch1",
                          passive_channels="Ch0",
                          testpulse_channels=["0"])

    # controlpulses_above don't add up
    with pytest.raises(ValueError): 
        dh.trigger_zscore(stream_csmpl,
                      trigger_channels="Ch1",
                      passive_channels="Ch0",
                      controlpulses_above=[1,2,3],
                      testpulse_channels=["0", "0"])
    with pytest.raises(ValueError): 
        dh.trigger_zscore(stream_csmpl,
                      trigger_channels="Ch1",
                      controlpulses_above=[1,2],
                      testpulse_channels="0")
        

def test_trigger_zscore(tempdir, stream_csmpl):
    dh1 = ai.DataHandler(nmbr_channels=2)
    dh1.set_filepath(tempdir.name, "test_trigger_zscore_1", appendix=False)
    dh1.init_empty()

    dh1.trigger_zscore(stream_csmpl, 
                      trigger_channels="Ch0",
                      passive_channels="Ch1",
                      testpulse_channels=["0", "0"],
                      copy_events=True,
                      reuse_triggers=False,
                      f_noise=100)

    # Have to create a separate datahandler because otherwise events cannot be included (gives warning)
    dh2 = ai.DataHandler(nmbr_channels=2)
    dh2.set_filepath(tempdir.name, "test_trigger_zscore_2", appendix=False)
    dh2.init_empty()
    
    dh2.trigger_zscore(stream_csmpl, 
                      trigger_channels="Ch0",
                      passive_channels="Ch1",
                      testpulse_channels=["0", "0"],
                      controlpulses_above=[0.5, 0.5],
                      copy_events=True,
                      reuse_triggers=False,
                      f_noise=100)
    
    # Test for controlpulses_above with tuple
    # Have to create a separate datahandler because otherwise events cannot be included (gives warning)
    dh3 = ai.DataHandler(nmbr_channels=2)
    dh3.set_filepath(tempdir.name, "test_trigger_zscore_3", appendix=False)
    dh3.init_empty()
    
    dh3.trigger_zscore(stream_csmpl, 
                      trigger_channels="Ch0",
                      passive_channels="Ch1",
                      testpulse_channels=["0", "0"],
                      controlpulses_above=[(15, 25), 0.5],
                      copy_events=True,
                      reuse_triggers=False,
                      f_noise=100)
    
def test_trigger_of_errors(tempdir, stream_csmpl):
    dh = ai.DataHandler(nmbr_channels=2)
    dh.set_filepath(tempdir.name, "test_trigger_of_errors", appendix=False)
    dh.init_empty()
        
    of = vai.OF(np.ones(int(dh.record_length/2+1), dtype=complex), dt_us=dh.dt_us)

    # thresholds and channels don't match
    with pytest.raises(ValueError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch0",
                      of=of,
                      passive_channels="Ch1",
                      thresholds=[1, 2, 3],
                      testpulse_channels=["0", "0"])
        
    # trigger_channel unavailable
    with pytest.raises(KeyError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch2",
                      of=of,
                      passive_channels="Ch0",
                      thresholds=[1],
                      testpulse_channels=["0", "0"])
        
    # passive_channel unavailable
    with pytest.raises(KeyError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch1",
                      of=of,
                      passive_channels="Ch2",
                      thresholds=[1],
                      testpulse_channels=["0", "0"])
        
    # tp_channel unavailable
    with pytest.raises(KeyError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch1",
                      of=of,
                      passive_channels="Ch0",
                      thresholds=[1],
                      testpulse_channels=["0", "2"])
        
    # tp_channel don't add up
    with pytest.raises(ValueError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch1",
                      of=of,
                      passive_channels="Ch0",
                      thresholds=[1],
                      testpulse_channels=["0"])

    # controlpulses_above don't add up
    with pytest.raises(ValueError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch1",
                      of=of,
                      passive_channels="Ch0",
                      thresholds=[1],
                      controlpulses_above=[1,2,3],
                      testpulse_channels=["0", "0"])
    with pytest.raises(ValueError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch1",
                      of=of,
                      thresholds=1,
                      controlpulses_above=[1,2],
                      testpulse_channels="0")
        
    of2 = vai.OF(np.ones((2, int(dh.record_length/2+1)), dtype=complex), dt_us=dh.dt_us)

    # wrong OF shape
    with pytest.raises(ValueError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch0",
                      of=of2,
                      passive_channels="Ch1",
                      thresholds=[1],
                      testpulse_channels=["0", "0"])
        

def test_trigger_of(tempdir, stream_csmpl):
    dh1 = ai.DataHandler(nmbr_channels=2)
    dh1.set_filepath(tempdir.name, "test_trigger_of_1", appendix=False)
    dh1.init_empty()

    of = vai.OF(np.ones(int(dh1.record_length/2+1), dtype=complex), dt_us=dh1.dt_us)

    dh1.trigger_of(stream_csmpl,
                  trigger_channels="Ch0",
                  of=of,
                  passive_channels="Ch1",
                  testpulse_channels=["0", "0"],
                  thresholds=[0.1],
                  copy_events=True,
                  reuse_triggers=False,
                  f_noise=100)
    
    # Have to create a separate datahandler because otherwise events cannot be included (gives warning)
    dh2 = ai.DataHandler(nmbr_channels=2)
    dh2.set_filepath(tempdir.name, "test_trigger_of_2", appendix=False)
    dh2.init_empty()

    of2 = vai.OF(np.ones(int(dh2.record_length/2+1), dtype=complex), dt_us=dh2.dt_us)

    dh2.trigger_of(stream_csmpl,
                  trigger_channels="Ch0",
                  of=of2,
                  passive_channels="Ch1",
                  testpulse_channels=["0", "0"],
                  controlpulses_above=[0.5, 0.5],
                  thresholds=[0.1],
                  copy_events=True,
                  reuse_triggers=False,
                  f_noise=100)