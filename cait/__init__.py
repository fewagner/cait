from .event_interface import EventInterface
from .data_handler import DataHandler
from .models._model_handler import ModelHandler
from .evaluation_tools import EvaluationTools
from .viztool import VizTool
from .limit import *
from .resources import *

from ._version import __version__

import os
# The total number of workers available for multiprocessing.Pool
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
           'trigger',
           'styles',
           'cuts',
           'calibration',
           'mixins',
           'limit',
           'augment',
           'resources',
           ]