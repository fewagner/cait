# FUNCTIONS THAT ARE MEANT TO BE APPLIED TO SINGLE EVENTS AND RETURN VALUES FOR THOSE EVENTS (NOT VOLTAGE TRACES AGAIN LIKE EVENT_FUNCTIONS)

import numpy as np

from ...models import CNNModule
from ...resources import get_resource_path

from .abstract_functions import FncBaseClass
from .event_functions import BoxCarSmoothing, RemoveBaseline
from .fit_functions import FitBaseline

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

#### NOT YET FINISHED ####
class CalcMP(FncBaseClass):
    def __init__(self,
                 dT: int = None, 
                 peak_bounds: tuple = (0, 1),
                 edge_size: float = 1/8,
                 box_car_smoothing: dict = {'length': 50}, 
                 fit_baseline: dict = {'model': 0, 'where': 1/8, 'xdata': None},
                 ):
        
        if not (isinstance(peak_bounds, tuple) and len(peak_bounds) == 2):
            raise ValueError(f"'{peak_bounds}' is not a valid interval.")
        if peak_bounds[0] > peak_bounds[1] or any([peak_bounds[0]<0, peak_bounds[1]<0, peak_bounds[0]>1, peak_bounds[1]>1]):
            raise ValueError("'peak_bounds' must be a tuple of increasing values between 0 and 1")
        
        self._dT = dT
        self._peak_bounds = peak_bounds
        self._edge_size = edge_size
        self._box_car_smoothing = BoxCarSmoothing(**box_car_smoothing)
        self._remove_baseline = RemoveBaseline(fit_baseline)
    
    def __call__(self, event):
        length = event.shape[-1]
        sl = slice(int(self._peak_bounds[0]*length), int(self._peak_bounds[1]*length))
        self._smooth_event = self._box_car_smoothing(event)

        self._lin_drift = (np.mean(self._smooth_event[-int(self._edge_size*length):]) - np.mean(self._smooth_event[:int(self._edge_size*length)])) / length

        self._offset = np.mean(event[:int(self._edge_size*length)])

        self._bl_removed_event = self._remove_baseline(self._smooth_event)
        self._t_max = int(np.argmax(self._bl_removed_event[sl], axis=-1) + sl.start)
        self._ph = self._bl_removed_event[self._t_max]
        
        self._t0 = np.where(self._bl_removed_event[:self._t_max] < 0.2*self._ph)[0][-1]
        self._t_rise = self._t0 + np.where(self._bl_removed_event[self._t0:self._t_max] > 0.8*self._ph)[0][0]
        self._t_decaystart = self._t_max + np.where(self._bl_removed_event[self._t_max:] < 0.9*self._ph)[0][0]
        self._t_half = self._t_decaystart + np.where(self._bl_removed_event[self._t_decaystart:] < 0.736*self._ph)[0][0]
        self._t_end = self._t_half + np.where(self._bl_removed_event[self._t_half:] < 0.368*self._ph)[0][0]
        
        return self._ph, self._t0, self._t_rise, self._t_max, self._t_decaystart, self._t_half, self._t_end, self._offset, self._lin_drift
    
    def preview(self, event) -> dict:
        mp = self(event)
        return dict(line = {'event': [None, event], 'MP': [list(mp[1:7]), event[list(mp[1:7])]]})
    
#### NOT YET FINISHED ####
class CalcAddMP(FncBaseClass):
    def __init__(self):
        ...

    def __call__(self, event):
        ...