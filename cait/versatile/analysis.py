from typing import Type, Callable
from multiprocessing import Pool
from inspect import signature
import itertools

import numpy as np
from tqdm.auto import tqdm

from .iterators import BatchResolver, IteratorBaseClass

def apply(f: Callable, ev_iter: Type[IteratorBaseClass], n_processes: int = 1):
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

    >>> # Example when func has one output
    >>> it = dh.get_event_iterator("events", batch_size=42)
    >>> out = apply(func, it)
    >>> # Example when func has two outputs
    >>> it = dh.get_event_iterator("events", batch_size=42)
    >>> out1, out2 = apply(func, it)
    """
    # Check if 'ev_iter' is a cait.versatile iterator object
    if not isinstance(ev_iter, IteratorBaseClass):
        raise TypeError(f"Input argument 'ev_iter' must be an instance of {IteratorBaseClass} not '{type(ev_iter)}'.")
    
    # Check if 'f' is indeed a function
    if not callable(f):
        raise TypeError(f"Input argument 'f' must be callable.")
    
    # Check if 'f' takes exactly one argument (the event)
    if len(signature(f).parameters) != 1:
        raise TypeError(f"Input function {f} has too many arguments ({len(signature(f).parameters)}). Only functions which take one argument (the event) are supported.")
    
    if ev_iter.uses_batches: f = BatchResolver(f)

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