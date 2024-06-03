import numpy as np

from ..functionbase import FncBaseClass
from ....models import CNNModule
from ....resources import get_resource_path

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
    
    @property
    def batch_support(self):
        return 'trivial'
    
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
    
    @property
    def batch_support(self):
        return 'trivial'
    
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