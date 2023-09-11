from multiprocessing import Pool
from tqdm.auto import tqdm
import numpy as np

class _BatchResolver:
    def __init__(self, f):
        self.f = f

    def __call__(self, batch):
        return [self.f(ev) for ev in batch]

def apply(f, ev_iter, n_processes=1, unpack=False):
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
    :param unpack: Set to True if you want the result to be unpacked into separate arrays. If False, a single array will be returned (containing all values). Defaults to False.
    :type unpack: bool

    :return: Results of `f` for all events in `ev_iter`
    :rtype: `~class:numpy.ndarray`

    >>> it = dh.get_event_iterator("events", batch_size=42)
    >>> arr = apply(func, it)
    
    >>> out1, out2, = apply(func, it, unpack=True)
    """
    if ev_iter.uses_batches: f = _BatchResolver(f)

    with ev_iter as ev_it:
        if n_processes > 1:
            with Pool(n_processes) as pool:
                out = list(tqdm(pool.imap(f, ev_it), total=ev_iter.n_batches))
        else:
            out = [f(ev) for ev in tqdm(ev_it, total=ev_iter.n_batches)]
    
    # Convert to numpy array and concatenate along the first dimension (event dimension)
    # in case batches were used
    array = np.concatenate(out, axis=0) if ev_iter.uses_batches else np.array(out)
        
    return (*array.T,) if unpack else array