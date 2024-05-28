from typing import Callable
from inspect import signature, _empty
from multiprocessing import Pool
import itertools

import numpy as np
from tqdm.auto import tqdm

from ..iterators.iteratorbase import IteratorBaseClass
from ..iterators.batchresolver import BatchResolver

def apply(f: Callable, ev_iter: IteratorBaseClass, n_processes: int = 1):
    """
    Apply a function to events provided by an EventIterator. 

    Multiprocessing and resolving batches as returned by the iterator is done automatically. The function returns a numpy array where the first dimension corresponds to the events returned by the iterator. Higher dimensions are as returned by the function that is applied. Batches are resolved, i.e. calls with an `EventIterator(..., batch_size=1)` and `EventIterator(..., batch_size=100)` yield identical results. 

    *Important*: Since `apply` uses multiprocessing, it is best not to use functions that are defined locally within jupyter lab, but rather to define them in a separate `.py` file and load them from the notebook. This is only relevant if you are trying to define your own function and not if you are just using already existing `cait` functions.

    :param f: Function to be applied to events. Note the restriction above.
    :type f: Callable
    :param ev_iter: Events for which the function should be applied.
    :type ev_iter: `~class:cait.versatile.file.EventIterator`
    :param n_processes: Number of processes to use for multiprocessing.
    :type n_processes: int

    :return: Results of `f` for all events in `ev_iter`. Has same structure as output of `f` (just with an additional event dimension).
    :rtype: Any

    **Example:**
    ::
        import cait.versatile as vai
        import numpy as np

        def func1(event): return np.max(event)
        def func2(event): return np.min(event), np.max(event)

        # Example when func has one output
        it = vai.MockData().get_event_iterator(batch_size=42)
        out = vai.apply(func1, it)

        # Example when func has two outputs
        it = vai.MockData().get_event_iterator(batch_size=42)
        out1, out2 = vai.apply(func2, it)
    """
    # Check if 'ev_iter' is a cait.versatile iterator object
    if not isinstance(ev_iter, IteratorBaseClass):
        raise TypeError(f"Input argument 'ev_iter' must be an instance of {IteratorBaseClass} not '{type(ev_iter)}'.")
    
    # Check if 'ev_iter' is not empty
    if len(ev_iter)==0:
        raise IndexError(f"Input argument 'ev_iter' must contain at least 1 event (iterator is empty).")
    
    # Check if 'f' is indeed a function
    if not callable(f):
        raise TypeError(f"Input argument 'f' must be callable.")
    
    # Check if 'f' takes exactly one required argument (the event)
    n_req_args = np.sum([x.default == _empty for x in signature(f).parameters.values()])
    if n_req_args != 1:
        raise TypeError(f"Input function {f} has too many required arguments ({n_req_args}). Only functions which take one (non-default) argument (the event) are supported.")
    
    if ev_iter.uses_batches: f = BatchResolver(f, ev_iter.n_channels)

    with ev_iter as ev_it:
        if n_processes > 1:
            with Pool(n_processes) as pool:
                out = list(tqdm(pool.imap(f, ev_it), total=ev_iter.n_batches))
        else:
            out = [f(ev) for ev in tqdm(ev_it, total=ev_iter.n_batches)]
    
    # Chain batches such that the list is indistinguishable from a list using no batches
    # (If uses_batches, 'out' is a list of lists)
    if ev_iter.uses_batches: out = list(itertools.chain.from_iterable(out))

    # If elements in 'out' are tuples, this means that the function had multiple outputs.
    # In this case, we transpose the list so that we have a tuple of outputs where each element in the tuple is also converted to a numpy.array of len(ev_iter)
    if isinstance(out[0], tuple): 
        out = tuple(np.array(x) for x in zip(*out))
    else:
        out = np.array(out)

    return out