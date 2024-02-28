# FUNCTIONS THAT ARE MEANT TO BE APPLIED TO SINGLE EVENTS AND RETURN VALUES FOR THOSE EVENTS (NOT VOLTAGE TRACES AGAIN LIKE EVENT_FUNCTIONS)

import numpy as np

from ...models import CNNModule
from ...resources import get_resource_path

from .abstract_functions import FncBaseClass
from .event_functions import BoxCarSmoothing, RemoveBaseline

# Helper function
def _check_CNNModule_availability():
    # pytorch is not a required dependency of cait
    # therefore, we check if CNNModule is available by instantiating it
    # (The check for missing modules is performed once classes are instantiated)
    CNNModule(512,2,'k',0,0.1)

# Needs explanation of the models
class AIClassifyBool(FncBaseClass):
    """
    Use a pre-trained neural network to classify a voltage trace into "particle pulse" and "not particle pulse" and return True if the output is "particle pulse".
    Also works for multiple channels simultaneously.

    :param model: The model that we want to use. See below:
    :type model: str, optional

    :return: True or False, depending on whether the voltage trace is classified as event pulse or artefact.
    :rtype: boolean
    """
    def __init__(self, model: str = "cnn-clf-binary-v2.ckpt"):
        # First check if module is available
        _check_CNNModule_availability()
        # Then load from checkpoint
        self._model = CNNModule.load_from_checkpoint(get_resource_path(model))

    def __call__(self, event):
        self._prediction = self._model.predict({"event_ch0": event}).numpy().astype(bool)
        return self._prediction[0] if self._prediction.size == 1 else self._prediction
    
    def preview(self, event):
        self(event)
        x = np.arange(event.shape[-1])
        
        if np.ndim(event) > 1:
            d = dict()
            for i in range(np.ndim(event)):
                d[f'channel {i}'] = [x, event[i]]
        else:
            d = {'event': [x, event]}
            
        return dict(line = d, axes=dict(xaxis={"label": f"Is event pulse: {self._prediction}"}))
    
# Needs explanation of the models
class AIClassifyProb(FncBaseClass):
    """
    Use a pre-trained neural network to classify a voltage trace into "particle pulse" and "not particle pulse" and return the percentage of it being a "particle pulse".
    Also works for multiple channels simultaneously.

    :param model: The model that we want to use. See below:
    :type model: str, optional

    :return: Probability (0 to 1), depending on how probable it is that the voltage trace is an event pulse.
    :rtype: boolean
    """
    def __init__(self, model: str = "cnn-clf-binary-v2.ckpt"):
        # First check if module is available
        _check_CNNModule_availability()
        # Then load from checkpoint
        self._model = CNNModule.load_from_checkpoint(get_resource_path(model))

    def __call__(self, event):
        self._prediction = np.exp(self._model.get_prob({"event_ch0": event}).numpy()[:,1])
        return self._prediction[0] if self._prediction.size == 1 else self._prediction
    
    def preview(self, event):
        self(event)
        x = np.arange(event.shape[-1])
        
        if np.ndim(event) > 1:
            d = dict()
            for i in range(np.ndim(event)):
                d[f'channel {i}'] = [x, event[i]]
        else:
            d = {'event': [x, event]}
            
        return dict(line = d, axes=dict(xaxis={"label": f"Event pulse with probability: {self._prediction}"}))

#### HAS NO TESTCASE YET ####
class CalcMP(FncBaseClass):
    """
    Calculates main parameters for an event. 
    If the argument ``dT`` is set to ``None``, the output is an array of shape ``(n_channels, 9)``, where the nine entries are ``ph, t_0, t_rise, t_max, t_decaystart, t_half, t_end, offset, lin_drift``, and quantities starting with ``t_`` are given as sample indices.
    If the argument ``dT`` is set (to the microsecond time base of the recording), the (human readable) quantities ``pulse_height (V), onset (ms), rise_time (ms), decay_time (ms), slope (V)`` as a tuple.
    Also works for multiple channels simultaneously.

    :param dt_us: The microsecond time base of the recording. If not provided, physical time constants cannot be computed and the function outputs indices. See above.
    :type dt_us: int, optional
    :param peak_bounds: The region in the record window which is considered for peak search. A tuple ``(0,1)`` searches the entire window, ``(0,1/2)`` only searches the first half, etc. Defaults to ``(1/5, 2/5)``.
    :type peak_bounds: tuple, optional
    :param edge_size: The (relative) size of the record window that is used to compute the linear drift. Defaults to 1/8, meaning that the first and last 1/8th of the record window is used.
    :type edge_size: float, optional
    :param box_car_smoothing: Arguments for class:`BoxCarSmoothing`, which are used for the application of the moving average. Defaults to ``{'length': 50}``.
    :type box_car_smoothing: dict, optional
    :param fit_baseline: Arguments for class:`FitBaseline`, which are used for the baseline subtractions. Defaults to ``{'model': 0, 'where': 1/8, 'xdata': None}``.
    :type fit_baseline: dict, optional

    :return: Either an array or tuple. See above.
    :rtype: np.ndarray, tuple

    **Example:**
    ::
        import cait.versatile as vai

        # Get events from mock data (and remove baseline)
        it = vai.MockData().get_event_iterator().with_processing(vai.RemoveBaseline())

        # Without the 'dt_us' keyword (output is numpy array)
        mp_array = vai.apply(vai.CalcMP(), it)

        # WITH the 'dt_us' keyword (output is tuple with physical values)
        pulse_height, onset, rise_time, decay_time, slope = vai.apply(vai.CalcMP(dt_us=10), it)

        vai.Histogram({'ch0': pulse_height[:,0], 'ch1': pulse_height[:,1]})
    """
    def __init__(self,
                 dt_us: int = None, 
                 peak_bounds: tuple = (1/5, 2/5),
                 edge_size: float = 1/8,
                 box_car_smoothing: dict = {'length': 50}, 
                 fit_baseline: dict = {'model': 0, 'where': 1/8, 'xdata': None},
                 ):
        
        if not (isinstance(peak_bounds, tuple) and len(peak_bounds) == 2):
            raise ValueError(f"'{peak_bounds}' is not a valid interval.")
        if peak_bounds[0] > peak_bounds[1] or any([peak_bounds[0]<0, peak_bounds[1]<0, peak_bounds[0]>1, peak_bounds[1]>1]):
            raise ValueError("'peak_bounds' must be a tuple of increasing values between 0 and 1")
        
        self._dt_us = dt_us
        self._peak_bounds = peak_bounds
        self._edge_size = edge_size
        self._box_car_smoothing = BoxCarSmoothing(**box_car_smoothing)
        self._remove_baseline = RemoveBaseline(fit_baseline)
    
    def __call__(self, event):
        length = event.shape[-1]
        sl = slice(int(self._peak_bounds[0]*length), int(self._peak_bounds[1]*length))
        self._smooth_event = self._box_car_smoothing(event)

        self._lin_drift = (np.mean(self._smooth_event[..., -int(self._edge_size*length):], axis=-1, keepdims=True) - 
                           np.mean(self._smooth_event[..., :int(self._edge_size*length)], axis=-1, keepdims=True)
                           ) / length

        self._offset = np.mean(event[..., :int(self._edge_size*length)], axis=-1, keepdims=True)

        self._bl_removed_event = self._remove_baseline(self._smooth_event)
        self._t_max = (np.argmax(self._bl_removed_event[..., sl], axis=-1, keepdims=True) 
                       + sl.start).astype(np.int32)

        self._ph = np.take_along_axis(self._bl_removed_event, self._t_max, axis=-1)

        # Helper array to slice multiple channels simultaneously
        inds = np.arange(length)

        self._t0 = length - 1 - np.argmax(
            np.flip(
                np.logical_and(
                self._bl_removed_event < 0.2*self._ph,
                inds < self._t_max),
                axis=-1),
            axis=-1, keepdims=True)
        
        self._t_rise = np.argmax(
            np.logical_and(
                self._bl_removed_event > 0.8*self._ph, 
                np.logical_and(inds >= self._t0, inds < self._t_max)),
            axis=-1, keepdims=True)
        
        self._t_end = np.argmax(
            np.logical_and(
                self._bl_removed_event < 0.368*self._ph, 
                inds >= self._t_max),
            axis=-1, keepdims=True)
        self._t_end[self._t_end == 0] = length - 1
        
        self._t_decaystart = np.argmax(
            np.logical_and(
                self._bl_removed_event < 0.9*self._ph, 
                inds >= self._t_max),
            axis=-1, keepdims=True)
        self._t_decaystart[self._t_decaystart == 0] = length - 1
        
        self._t_half = np.argmax(
            np.logical_and(
                self._bl_removed_event < 0.736*self._ph, 
                inds >= self._t_max),
            axis=-1, keepdims=True)
        self._t_half[self._t_half == 0] = length - 1

        # If no time base is specified, the indices are returned (all in one numpy array)
        if self._dt_us is None:
            return np.concatenate([self._ph, self._t0, self._t_rise, self._t_max, self._t_decaystart, self._t_half, self._t_end, self._offset, self._lin_drift], axis=-1)
        
        else:
            return (self._ph.flatten(),                                              # pulse height
                    ((self._t0 - length/4)*self._dt_us/1000).flatten(),              # onset (ms)
                    ((self._t_rise - self._t0)*self._dt_us/1000).flatten(),          # rise time (ms)
                    ((self._t_end - self._t_decaystart)*self._dt_us/1000).flatten(), # decay time (ms)
                    (self._lin_drift*length).flatten()                               # slope (V)
                    )
    
    def preview(self, event) -> dict:
        self(event)
        mp = np.concatenate([self._t0, self._t_rise, self._t_max, self._t_decaystart, self._t_half, self._t_end], axis=-1)
        x = np.arange(event.shape[-1])
        if self._dt_us is not None:
            x = (x - event.shape[-1]/4)*self._dt_us/1000
        
        if np.ndim(event) > 1:
            d1 = dict()
            d2 = dict()
            for i in range(np.ndim(event)):
                d1[f'channel {i}'] = [x, event[i]]
                d1[f'channel {i} bcs'] = [x, self._smooth_event[i]]
                d2[f'MP channel {i}'] = [x[mp[i]], self._smooth_event[i][mp[i]]]
        else:
            d1 = {'event': [x, event], 'event bcs': [x, self._smooth_event]}
            d2 = {'MP': [x[mp], self._smooth_event[mp]]}
            
        return dict(line = d1, scatter=d2, axes=dict(xaxis={"label": "time (ms)" if self._dt_us is not None else "index"}))
    
#### NOT YET FINISHED ####
class CalcAddMP(FncBaseClass):
    def __init__(self):
        ...

    def __call__(self, event):
        ...