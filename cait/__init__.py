import os

from . import serialize, styles
from ._version import __version__
from .data_handler import DataHandler
from .evaluation_tools import EvaluationTools
from .event_interface import EventInterface
from .limit import *
from .models._model_handler import ModelHandler
from .resources import *
from .viztool import VizTool

# The total number of workers available for multiprocess.Pool
_available_workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()

__all__ = ['EventInterface',
           'DataHandler',
           'ModelHandler',
           'EvaluationTools',
           'VizTool',
           'data',
           'datasets',
           'evaluation',
           'features',
           'filter',
           'fit',
           'models',
           'readers',
           'simulate',
           'serialize',
           'trigger',
           'styles',
           'cuts',
           'calibration',
           'mixins',
           'limit',
           'augment',
           'resources',
           ]