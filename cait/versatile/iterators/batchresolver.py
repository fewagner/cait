from typing import Union, Callable

import numpy as np

from ..eventfunctions.functionbase import FncBaseClass

class BatchResolver:
    """
    Helper Class to resolve batched iterators.
    """
    def __init__(self, f: Union[FncBaseClass, Callable], n_channels: int):
        self._f = f
        self._bs = "none" if not hasattr(f, "batch_support") else f.batch_support
        self._n_channels = n_channels

        if self._bs not in ["none", "trivial", "full"]:
            raise NotImplementedError(f"{self._bs} is not a valid batch_support string.")
        
    def __call__(self, batch):
        # If function can process batches (possibly of multiple channels)
        # by itself, it is applied to the entire batch
        if self._bs == "full":
            result = self._f(batch)
            if isinstance(result, tuple): return list(zip(*result))
            else: return result
        
        # Function is applied separately to each event 
        # (with possibly multiple channels)
        if self._bs == "none":
            return [self._f(ev) for ev in batch]
        
        # Function is applied to a flattened version of the batch
        if self._bs == "trivial":
            # If batch has only one channel, 
            # the function is applied to the entire batch at once
            batch = np.array(batch)
            if self._n_channels == 1:
                result = self._f(batch)

                # If the result is a tuple, it is repackaged so that it
                # has the same shape as for "none". Otherwise, the 
                # numpy array is returned directly
                if isinstance(result, tuple): return list(zip(*result))
                else: return result
            
            # If batch has multiple channels, we reshape the array to mimic
            # a batch of only one channel. Before returning, the result is 
            # reshaped again
            n_events = batch.shape[0]
            total_traces = n_events*self._n_channels
            record_length = batch.shape[-1]

            result = self._f(batch.reshape(total_traces, record_length))

            # If function returns tuple (e.g. fit result and rms),
            # we have to repackage it again
            if isinstance(result, tuple):
                temp = [np.squeeze(
                            np.reshape(
                                x, (n_events, 
                                    self._n_channels, 
                                    np.array(x).size//n_events//self._n_channels
                                    )
                            )
                        )[()] for x in result]
                return list(zip(*temp))

            # If the result is a numpy array, we can just reshape and return it
            missing_dim = result.size//n_events//self._n_channels

            return np.squeeze(
                result.reshape(n_events, self._n_channels, missing_dim)
                )[()]