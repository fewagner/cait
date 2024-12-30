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
                          slave_channels="Ch1",
                          testpulse_channels=["0", "0"])
        
    # slave_channel unavailable
    with pytest.raises(KeyError): 
        dh.trigger_zscore(stream_csmpl, 
                          trigger_channels="Ch1",
                          slave_channels="Ch2",
                          testpulse_channels=["0", "0"])
        
    # tp_channel unavailable
    with pytest.raises(KeyError): 
        dh.trigger_zscore(stream_csmpl, 
                          trigger_channels="Ch1",
                          slave_channels="Ch0",
                          testpulse_channels=["0", "2"])
        
    # tp_channel don't add up
    with pytest.raises(ValueError): 
        dh.trigger_zscore(stream_csmpl, 
                          trigger_channels="Ch1",
                          slave_channels="Ch0",
                          testpulse_channels=["0"])
        

def test_trigger_zscore(tempdir, stream_csmpl):
    dh = ai.DataHandler(nmbr_channels=2)
    dh.set_filepath(tempdir.name, "test_trigger_zscore", appendix=False)
    dh.init_empty()

    dh.trigger_zscore(stream_csmpl, 
                      trigger_channels="Ch0",
                      slave_channels="Ch1",
                      testpulse_channels=["0", "0"],
                      copy_events=True,
                      reuse_triggers=False,
                      n_noise=100)
    
def test_trigger_of_errors(tempdir, stream_csmpl):
    dh = ai.DataHandler(nmbr_channels=2)
    dh.set_filepath(tempdir.name, "test_trigger_of_errors", appendix=False)
    dh.init_empty()

    # no OF available
    with pytest.raises(ValueError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch0",
                      slave_channels="Ch1",
                      thresholds=[1],
                      testpulse_channels=["0", "0"])
        
    vai.OF(np.ones(int(dh.record_length/2+1), dtype=complex), dt_us=dh.dt_us).to_dh(dh, overwrite_existing=True)

    # thresholds and channels don't match
    with pytest.raises(ValueError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch0",
                      slave_channels="Ch1",
                      thresholds=[1, 2, 3],
                      testpulse_channels=["0", "0"])
        
    # trigger_channel unavailable
    with pytest.raises(KeyError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch2",
                      slave_channels="Ch0",
                      thresholds=[1],
                      testpulse_channels=["0", "0"])
        
    # slave_channel unavailable
    with pytest.raises(KeyError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch1",
                      slave_channels="Ch2",
                      thresholds=[1],
                      testpulse_channels=["0", "0"])
        
    # tp_channel unavailable
    with pytest.raises(KeyError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch1",
                      slave_channels="Ch0",
                      thresholds=[1],
                      testpulse_channels=["0", "2"])
        
    # tp_channel don't add up
    with pytest.raises(ValueError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch1",
                      slave_channels="Ch0",
                      thresholds=[1],
                      testpulse_channels=["0"])
        
    vai.OF(np.ones((2, int(dh.record_length/2+1)), dtype=complex), dt_us=dh.dt_us).to_dh(dh, overwrite_existing=True)

    # wrong OF shape
    with pytest.raises(ValueError): 
        dh.trigger_of(stream_csmpl,
                      trigger_channels="Ch0",
                      slave_channels="Ch1",
                      thresholds=[1],
                      testpulse_channels=["0", "0"])
        

def test_trigger_of(tempdir, stream_csmpl):
    dh = ai.DataHandler(nmbr_channels=2)
    dh.set_filepath(tempdir.name, "test_trigger_of", appendix=False)
    dh.init_empty()

    vai.OF(np.ones(int(dh.record_length/2+1), dtype=complex), dt_us=dh.dt_us).to_dh(dh, overwrite_existing=True)

    dh.trigger_of(stream_csmpl,
                  trigger_channels="Ch0",
                  slave_channels="Ch1",
                  testpulse_channels=["0", "0"],
                  thresholds=[0.1],
                  copy_events=True,
                  reuse_triggers=False,
                  n_noise=100)