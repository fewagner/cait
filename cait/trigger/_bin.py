# imports

import numpy as np
import numba as nb
from ..data._raw import convert_to_V
from ..filter._of import filter_event
from ..styles import use_cait_style, make_grid
from scipy import signal
import matplotlib.pyplot as plt
import sqlite3
from time import time, strptime, mktime
from tqdm.auto import tqdm


# functions

def get_record_window():
    # TODO
    pass
