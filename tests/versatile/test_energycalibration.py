import numpy as np
import pytest
import scipy as sp

from cait.serialize.serialize import all_subclasses
from cait.versatile import EnergyCalibration, MockData
from cait.versatile.analysisobjects.testpulseresponse import TestpulseResponse
from cait.versatile.analysisobjects.transferfunction import TransferFunction

SEED = 137

TEST_TS = MockData(n_events=137).get_event_iterator().timestamps # shape (137,)
TEST_TPAS = np.array([1.2, 3.4, 5.8, 9.4]) # shape (4,)
TEST_TPH = sp.stats.norm.rvs(loc=np.array([TEST_TPAS]).T, 
                             scale=0.2, 
                             size=(4,len(TEST_TS)),
                             random_state=SEED,
                             ) # shape (4, 137)
TEST_EVAL_TS = MockData(n_events=279).get_event_iterator().timestamps # shape (279,)
TEST_EVAL_TPES = np.linspace(np.linspace(0, 8, 31), 
                             np.linspace(1, 9, 31), 
                             len(TEST_TS)
                             ) # shape (137, 31)
TEST_EVAL_PHS = np.linspace(np.linspace(0, 8, 31), 
                             np.linspace(1, 9, 31), 
                             len(TEST_TS)
                             ) # shape (137, 31)

# Dynamically load all children classes for TestpulseResponse and TransferFunction
# for general sanity checks. Detailed tests (e.g. for different input arguments)
# have to be checked separately in dedicated test functions.
all_tprs = all_subclasses(TestpulseResponse)
all_tfs = all_subclasses(TransferFunction)

@pytest.mark.parametrize("tpr_obj", all_tprs)
def test_sanity_check_testpulse_response(tpr_obj):
    # Test for default arguments
    tpr = tpr_obj()
    prepared_instance = tpr.prepare(x=TEST_TS, tp_phs=TEST_TPH[0])
    
    assert isinstance(prepared_instance, tpr.__class__), f"The 'prepare' method of {tpr.__class__.__name__} must return the class instance, i.e. its last line has to be 'return self'."

    for eval_ts in [
        TEST_EVAL_TS, # test 1d input
        TEST_EVAL_TS[0], # test scalar input
        ]:
        tpe_values = prepared_instance(eval_ts)
        assert isinstance(tpe_values, np.ndarray), f"Calling the {tpr.__class__.__name__} instance must return a numpy array. Not {type(tpe_values)}."
        assert tpe_values.shape == np.shape(eval_ts), f"Calling the {tpr.__class__.__name__} instance must return a numpy array with the same shape as the input array 'x'. Got shapes {tpe_values.shape} and {np.shape(eval_ts)}."

    tpr_obj.preview(x=TEST_TS, tp_phs=TEST_TPH[0], backend="plotly")

    # Test if errors are raised in case of shape mismatch:
    with pytest.raises(ValueError):
        # Unequal lengths of input
        tpr_obj().prepare(x=TEST_TS[1:], tp_phs=TEST_TPH[0])

    with pytest.raises(ValueError):
        # Input not 1d
        tpr_obj().prepare(x=[[1,2,3], [3,4,5]], tp_phs=[[6,7,8], [9,10,11]])

    with pytest.raises(ValueError):
        # Prepare fine but wrong shape for __call__
        tpr_obj().prepare(x=TEST_TS, tp_phs=TEST_TPH[0])([TEST_EVAL_TS[:10], TEST_EVAL_TS[:10]])

@pytest.mark.parametrize("tf_obj", all_tfs)
def test_sanity_check_transfer_function(tf_obj):
    # Test for default arguments
    tf = tf_obj()

    # Test different pairs of input shapes
    for input_tp_phs, input_tpes, input_phs in [
        (
            TEST_TPH[:,0], 
            TEST_EVAL_TPES[0][0], 
            TEST_EVAL_PHS[0][0]
         ), # 1d fit, scalar evaluation
        (
            TEST_TPH[:,0], 
            TEST_EVAL_TPES[0], 
            TEST_EVAL_PHS[0]
        ), # 1d fit, 1d evaluation
        (
            TEST_TPH.T, 
            np.atleast_2d(TEST_EVAL_TPES[:,0]).T, 
            np.atleast_2d(TEST_EVAL_PHS[:,0]).T
        ), # 2d fit, 1d evaluation
        (
            TEST_TPH.T, 
            TEST_EVAL_TPES, 
            TEST_EVAL_PHS
        ), # 2d fit, 2d evaluation
    ]:
        # FORWARD
        ph_output = tf(TEST_TPAS, input_tp_phs, input_tpes)
        assert isinstance(ph_output, np.ndarray), f"Calling the {tf.__class__.__name__} instance must return a numpy array. Not {type(ph_output)}."
        assert ph_output.shape == input_tpes.shape, f"Calling the {tf.__class__.__name__} instance must return a numpy array with the same shape as the 'tpes' input shape. Expected {input_tpes.shape}, got {ph_output.shape}."

        # INVERSE
        tpe_output = tf.inverse(TEST_TPAS, input_tp_phs, input_phs)
        assert isinstance(tpe_output, np.ndarray), f"Calling .inverse() on the {tf.__class__.__name__} instance must return a numpy array. Not {type(tpe_output)}."
        assert tpe_output.shape == input_phs.shape, f"Calling .inverse() on the {tf.__class__.__name__} instance must return a numpy array with the same shape as the 'phs' input shape. Expected {input_phs.shape}, got {tpe_output.shape}."

        # Check consistency (allow 2*eps because we do it backwards and forwards, so we
        # could pick up slightly more inaccuracy than the root finder allows)
        assert np.allclose(
           tf.inverse(TEST_TPAS, input_tp_phs, tf(TEST_TPAS, input_tp_phs, input_tpes)), 
           input_tpes,
           rtol=0,
           atol=2*tf._DEFAULT_SECANT_ROOT_FIND_ARGS["eps"],
           ), f"Calling f.inverse(f(x)) on {tf.__class__.__name__} must return x."
        assert np.allclose(
            tf(TEST_TPAS, input_tp_phs, tf.inverse(TEST_TPAS, input_tp_phs, input_phs)), 
            input_phs,
            rtol=0,
            atol=2*tf._DEFAULT_SECANT_ROOT_FIND_ARGS["eps"],
            ), f"Calling f(f.inverse(x)) on {tf.__class__.__name__} must return x."

    # Test if errors are correctly raised in case of shape mismatch.
    # I.e. we check if input sanitization/validation is performed.
    # Repeat for both forward and inverse.
    for f, test_input in zip([tf, tf.inverse],[TEST_EVAL_TPES, TEST_EVAL_PHS]):
        with pytest.raises(ValueError):
            # Unequal lengths of inputs (tpas, tp_phs)
            f(TEST_TPAS, TEST_TPH[1:,0], test_input[0])
        with pytest.raises(ValueError):
            # Non-1d input for tpas
            f([TEST_TPAS, TEST_TPAS], TEST_TPH[:,0], test_input[0])
        with pytest.raises(ValueError):
            # Too high dimensional input for tpas
            f([TEST_TPAS, TEST_TPAS], TEST_TPH[:,0], test_input[0])
        with pytest.raises(ValueError):
            # Too high dimensional input for tpes
            f(TEST_TPAS, TEST_TPH[:,0], [[test_input[0]]])
        with pytest.raises(ValueError):
            # tpas dimension doesn't match tp_ph dimension
            f(TEST_TPAS, TEST_TPH.T[:,1:], test_input[1:, :])
        with pytest.raises(ValueError):
            # tp_ph dimension doesn't match tpe dimension
            f(TEST_TPAS, TEST_TPH.T, test_input[1:, :])

@pytest.mark.parametrize("tpr_obj", all_tprs)
@pytest.mark.parametrize("tf_obj", all_tfs)
def test_energy_calibration(tpr_obj, tf_obj):
    rng_seed = 137

    N = 5000
    start_us = 1426321613000000
    ts = np.hstack([sp.stats.randint.rvs(low=start_us,
                                        high=start_us+3*1e6*3600, 
                                        size=N, 
                                        random_state=rng_seed), 
                    sp.stats.randint.rvs(low=start_us+4*1e6*3600,
                                        high=start_us+7*1e6*3600, 
                                        size=N,
                                        random_state=rng_seed)])
    tpas = sp.stats.randint.rvs(low=1, high=6, size=2*N, random_state=rng_seed)
    tp_phs = sp.stats.norm.rvs(loc=np.sqrt(0.3*tpas), scale=0.05, size=2*N, random_state=rng_seed)

    my_ecal = EnergyCalibration(
        tp_x=ts,
        tp_phs=tp_phs,
        tpas=tpas,
        testpulse_response=tpr_obj(),
        transfer_function=tf_obj(),
        max_x_gap=0.5, 
    )

    in_arr1 = np.linspace(2, 5, 10)
    in_arr2 = np.array([np.linspace(2, 5, 10), 
                       np.linspace(2, 5, 10), 
                       np.linspace(2, 5, 10)]).T
    out_arr1 = my_ecal(ts[10:20], in_arr1)
    out_arr2 = my_ecal(ts[10:20], in_arr2)
    
    inv_out_arr1 = my_ecal.inverse(ts[10:20], out_arr1)
    inv_out_arr2 = my_ecal.inverse(ts[10:20], out_arr2)

    for in_arr, inv_out_arr in zip([in_arr1, in_arr2], [inv_out_arr1, inv_out_arr2]):
        assert np.allclose(
                in_arr, 
                inv_out_arr,
                rtol=0,
                atol=10*tf_obj._DEFAULT_SECANT_ROOT_FIND_ARGS["eps"],
                ), f"Calling f(f.inverse(x)) on {my_ecal.__class__.__name__} must return x."
        
    # Try basic plotting
    my_ecal.plot_testpulse_response(backend="plotly")
    my_ecal.plot_transfer_function(ts[10], backend="plotly")

    # Trying to plot for multiple times
    with pytest.raises(ValueError):
        my_ecal.plot_transfer_function(ts[10:20])

    # Try preview plotting
    my_ecal.preview(backend="plotly")