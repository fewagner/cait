from .processing.align import Align
from .processing.boxcarsmoothing import BoxCarSmoothing
from .processing.downsample import Downsample
from .processing.optimumfiltering import OptimumFiltering
from .processing.removebaseline import RemoveBaseline
from .processing.tukey import TukeyWindow

from .scalarfunctions.calcmp import CalcMP
from .scalarfunctions.fitbaseline import FitBaseline
from .scalarfunctions.templatefit import TemplateFit
from .scalarfunctions.npeaks import NPeaks
from .scalarfunctions.triggersurvival import TriggerSurvival

from .functionbase import FncBaseClass