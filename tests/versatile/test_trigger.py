import os
import pytest

import numpy as np
import cait as ai
import cait.versatile as vai

from ..fixtures import tempdir

@pytest.fixture(scope="module")
def stream_csmpl(tempdir):
    data = ai.data.TestData(filepath=tempdir.name+'/mock_001',
                            duration=100,
                            channels=[0, 1],
                            sample_frequency=25000,
                            start_s=13,
                            tpas=[20, -1.0, 20, 0.3, 0, 20, 0.5, 1, 20, 3, -1, 20, 0, 10])
    data.generate()

    yield vai.Stream('csmpl', [tempdir.name+'/mock_001_Ch0.csmpl',
                                   tempdir.name+'/mock_001_Ch1.csmpl',
                                   tempdir.name+'/mock_001.test_stamps',
                                   tempdir.name+'/mock_001.dig_stamps',
                                   tempdir.name+'/mock_001.par'])

def test_trigger_basic(stream_csmpl):
    i, v = vai.trigger_zscore(stream_csmpl["Ch0"], 2**14)
    i, v = vai.trigger_zscore(stream_csmpl["Ch0"], 2**14, n_triggers=10)
    i, v = vai.trigger_zscore(stream_csmpl["Ch1"], 2**14, apply_first=lambda x: -x)

@pytest.mark.parametrize("trigger", [
    lambda d: vai.trigger_zscore(d, record_length=2**15), 
    lambda d: vai.trigger_of(d, threshold=0.5, of=np.ones(2**15//2+1))
    ]
)
def test_trigger_single_samples(trigger):
    record_length = 2**15
    
    # should not find the peak (outside of search area)
    data = np.zeros(record_length*13)
    data[record_length-1] = 1
    i, _ = trigger(data)
    assert len(i)==0

    # should find the peak on first searched sample
    data = np.zeros(record_length*13)
    peakpos = record_length
    data[peakpos-50:peakpos] = np.linspace(0,1,50)
    data[peakpos:peakpos+50] = np.linspace(1,0,50)
    i, _ = trigger(data)
    assert len(i)==1 and i[0]==peakpos 

    # should not find the peak (outside of search area)
    data = np.zeros(record_length*13)
    data[-record_length] = 1
    i, _ = trigger(data)
    assert len(i)==0 

    # should find the peak on first sample (from back) that is searched
    data = np.zeros(record_length*13)
    peakpos = -2*record_length
    data[peakpos-49:peakpos+1] = np.linspace(0,1,50)
    data[peakpos:peakpos+50] = np.linspace(1,0,50)
    i, _ = trigger(data)
    assert len(i)==1 and i[0]==(len(data) + peakpos)

    # should find the peak on first searched sample but not the second one (within record_length/2 and smaller)
    data = np.zeros(record_length*13)
    peakpos = record_length
    peakpos2 = record_length + 1000
    data[peakpos-50:peakpos] = np.linspace(0,1,50)
    data[peakpos:peakpos+50] = np.linspace(1,0,50)
    data[peakpos2-50:peakpos2] = 0.9*np.linspace(0,1,50)
    data[peakpos2:peakpos2+50] = 0.9*np.linspace(1,0,50)

    i, _ = trigger(data)
    assert len(i)==1 and i[0]==peakpos

    # should find both peaks (outside record_length/2 and smaller)
    data = np.zeros(record_length*13)
    peakpos = record_length
    peakpos2 = record_length + record_length//2 + 1
    data[peakpos-50:peakpos] = np.linspace(0,1,50)
    data[peakpos:peakpos+50] = np.linspace(1,0,50)
    data[peakpos2-50:peakpos2] = 0.9*np.linspace(0,1,50)
    data[peakpos2:peakpos2+50] = 0.9*np.linspace(1,0,50)

    i, _ = trigger(data)
    assert len(i)==2 and i[1]==(record_length+record_length//2+1)
    
def test_trigger_on_data():
    record_length = 2**13
    filepath = os.path.dirname(os.path.realpath(__file__))
    
    # Saved optimum filter mimicing one of a mediocrely performing detector.
    of = vai.OF.from_file(os.path.join(filepath, "testdata", "OF_test"))
    
    # Saved test data mimicing a mediocrely performing detector.
    # If tests work on this one, we should be good ;)
    # I manually checked the data and the trigger for it and the
    # trigger indices below should be regarded as correct.
    data1 = np.loadtxt(os.path.join(filepath, "testdata", "stream_test_19_11.txt"))
    data2 = np.loadtxt(os.path.join(filepath, "testdata", "stream_test_20_12.txt"))
    
    inds, _ = vai.trigger_of(data1, 0.001, of, chunk_size=100)
    assert np.array_equal(np.array(inds),
                          np.array([
                              8192, 51364, 84780, 112808, 148103, 201451, 207502, 240854, 
                              250365, 256341, 321179, 358152, 364750, 371700, 380828, 416178, 
                              451448, 457506, 483314
                          ])
                         )
    
    inds, _ = vai.trigger_zscore(data1, chunk_size=100, threshold=3, record_length=record_length)
    assert np.array_equal(np.array(inds),
                          np.array([
                              51356, 84796, 112783, 148110, 201443, 240906, 256345, 
                              321167, 358153, 451442, 483320
                          ])
                         )
    
    inds, _ = vai.trigger_of(data2, 0.001, of, chunk_size=100)
    assert np.array_equal(np.array(inds),
                          np.array([
                              16942, 60214, 93630, 121658, 156953, 210301, 216352, 249704, 
                              259215, 265191, 330029, 367002, 373600, 380550, 389678, 425028, 
                              460298, 466356, 492164, 536326
                          ])
                         )
    
    inds, _ = vai.trigger_zscore(data2, chunk_size=100, threshold=3, record_length=record_length)
    assert np.array_equal(np.array(inds),
                          np.array([
                              16949, 60206, 93646, 121633, 156960, 210293, 249756, 265195, 
                              330017, 367003, 460292, 492170
                          ])
                         )
    
    
def test_trigger_different_chunk_sizes():
    # make sure the number of triggers is the same regardless of the chunk size
    
    record_length = 2**13
    filepath = os.path.dirname(os.path.realpath(__file__))
    
    # Saved optimum filter mimicing one of a mediocrely performing detector.
    of = vai.OF.from_file(os.path.join(filepath, "testdata", "OF_test"))
    
    # Saved test data mimicing a mediocrely performing detector.
    # If tests work on this one, we should be good ;)
    # I manually checked the data and the trigger for it and the
    # trigger indices below should be regarded as correct.
    data = np.loadtxt(os.path.join(filepath, "testdata", "stream_test_19_11.txt"))

    # Test all chunk sizes from 1 to 100 (data has length ~66 record lengths) for OF trigger 
    # (trigger inds should be the same)
    assert all([
                np.array_equal(
                        np.array(vai.trigger_of(data, 0.001, of, chunk_size=cs)[0]),
                        np.array([ 8192, 51364, 84780, 112808, 148103, 201451, 207502, 240854, 
                                   250365, 256341, 321179, 358152, 364750, 371700, 380828, 416178, 
                                   451448, 457506, 483314 ])
                )
                for cs in range(1, 100)
            ])
    
    # Test all chunk sizes from 1 to 100 (data has length ~66 record lengths) for OF trigger 
    # (trigger inds should be the same)
    assert all([
                np.array_equal(
                        np.array(vai.trigger_zscore(data, chunk_size=cs, threshold=3, record_length=record_length)[0]),
                        np.array([ 51356, 84796, 112783, 148110, 201443, 240906, 256345, 
                                   321167, 358153, 451442, 483320])
                )
                for cs in range(1, 100)
            ])