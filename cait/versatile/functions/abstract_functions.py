from abc import ABC, abstractmethod

class FncBaseClass(ABC):   
    """
    Abstract class that represents a function which is meant to be applied to event voltage traces. The configuration of the function (i.e. setting its non-event input values) is done upon initialization. As a convention, function inputs will be saved in corresponding class attributes prepended with an underscore (e.g. for a function receiving a parameter `param`, a corresponding `self._param` will be defined in the `__init__`.)

    This class defines two necessary methods: `__call__` which takes one argument (the event voltage trace; can be multiple channels, depending on the function implementation) and returns the result. Multiple results are returned as tuples (e.g. `return result1, result2`) and if one result is a list/vector, it is encouraged to return a `numpy.ndarray`.
    The `preview` method is called when the function is used in combination with the :class:`Preview` class and (if implemented) is expected to return a dictionary of the form specified in :class:`Preview`. 
    """
    @abstractmethod
    def __call__(self, event):
        ...
    
    def preview(self, event) -> dict:
        raise NotImplementedError(f"{self.__class__.__name__} does not support preview.")
        
class FitFncBaseClass(FncBaseClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def model(self, x, *pars):
        ...
