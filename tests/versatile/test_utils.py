import numpy as np
import cait.versatile as vai

def test_timestamp_coincidence_standard():
    a = np.array([10, 20, 30, 40])
    b = np.array([1, 11, 35, 42, 45])

    inside, coincidence_ind, outside = vai.timestamp_coincidence(a, b, (-1,2))
    assert np.array_equal(inside, np.array([1]))
    assert np.array_equal(outside, np.array([0,2,3,4]))
    assert np.array_equal(coincidence_ind, np.array([0]))
    assert np.array_equal(b[inside], np.array([11]))
    assert np.array_equal(b[outside], np.array([1,35,42,45]))
    assert np.array_equal(a[coincidence_ind], np.array([10]))

    inside, coincidence_ind, outside = vai.timestamp_coincidence(a, b, (-1,3))
    assert np.array_equal(inside, np.array([1,3]))
    assert np.array_equal(outside, np.array([0,2,4]))
    assert np.array_equal(coincidence_ind, np.array([0,3]))
    assert np.array_equal(b[inside], np.array([11,42]))
    assert np.array_equal(b[outside], np.array([1,35,45]))
    assert np.array_equal(a[coincidence_ind], np.array([10,40]))

def test_timestamp_coincidence_all_negative():
    a = np.array([10, 20, 30, 40])
    b = np.array([7, 8, 11, 35, 42, 45])

    inside, coincidence_ind, outside = vai.timestamp_coincidence(a, b, (-3,-1))
    assert np.array_equal(b[inside], np.array([7,8]))
    assert np.array_equal(b[outside], np.array([11,35,42,45]))
    assert np.array_equal(a[coincidence_ind], np.array([10,10]))

def test_sample_noise():
    record_length = 10
    onset = int(record_length/4)

    trigger_inds = [100, 110, 200, 250, 500, 1000, 1150]
    noise_inds = vai.sample_noise(trigger_inds=trigger_inds, record_length=record_length)

    # Check if nothing overlaps
    x = trigger_inds + noise_inds
    x.sort()
    intervals = np.array([np.array(x)-onset, np.array(x)+record_length-onset-1]).flatten(order="F")
    
    assert np.any(np.diff(intervals) < record_length)