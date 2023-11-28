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
                 peak_bounds: tuple = None,
                 box_car_smoothing: dict = {'length': 50}, 
                 fit_baseline: dict = {'model': 0, 'where': 8, 'xdata': None}):
        
        if peak_bounds is None:
            self._peak_bounds = slice(None, None, None)
        elif type(peak_bounds) is tuple and len(peak_bounds) == 2:
            self._peak_bounds = slice(peak_bounds[0], peak_bounds[1])
        else:
            raise ValueError(f"'{peak_bounds}' is not a valid interval.")
        
        self._box_car_smoothing = BoxCarSmoothing(**box_car_smoothing)
        self._remove_baseline = RemoveBaseline(**fit_baseline)
        self._get_offset = FitBaseline(model=0, where=8)
    
    def __call__(self, event):
        self._offset_MP = self._get_offset(event)
    
#### NOT YET FINISHED ####
class CalcAddMP(FncBaseClass):
    def __init__(self):
        ...

    def __call__(self, event):
        ...