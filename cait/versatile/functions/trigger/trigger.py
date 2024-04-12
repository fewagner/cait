from typing import Union, List, Callable

import numpy as np
from tqdm.auto import tqdm

from ...eventfunctions.processing.removebaseline import RemoveBaseline
from ...datasources.stream.streambase import StreamBaseClass

####################################################
### FUNCTIONS IN THIS FILE HAVE NO TESTCASES YET ###
####################################################

def trigger(stream, 
            key: str, 
            threshold: float,
            record_length: int,
            overlap_fraction: float = 1/8,
            trigger_block: int = None,
            n_triggers: int = None,
            preprocessing: Union[Callable, List[Callable]] = None):
    """
    Trigger a single channel of a stream object with options for adding preprocessing like optimum filtering, applying window functions, or inverting the stream.
    If no preprocessing is specified, only the baseline of the voltage trace is removed (using a constant baseline model and the first `int(overlap_fraction/2)` samples of the record window). If a function, e.g. `lambda x: -x` is given, it is added *after* the removal of the baseline. Only if `RemoveBaseline` is explicitly given, the user can choose when it is applied, e.g. `[lambda, x: x**2, RemoveBaseline()]` first squares the voltage trace and removes the baseline afterwards. 

    :param stream: The stream object with the channel to trigger.
    :type stream: StreamBaseClass
    :param key: The name of the channel in `stream` to trigger.
    :type key: str
    :param threshold: The threshold above which events should be triggered.
    :type threshold: float
    :param record_length: The length of the record window to use for triggering. Typically, this is a power of 2, e.g. 16384.
    :type record_length: int
    :param overlap_fraction: The fraction of the record window that should overlap between subsequent maximum searches. Number in the interval (0, 1/4], defaults to 1/8.
    :type overlap_fraction: float
    :param trigger_block: The number of samples for which the trigger should be blocked after a successful trigger. Has to be larger than `record_length`. If `None`, `record_length` is used. Defaults to None.
    :type trigger_block: int
    :param n_triggers: The maximum number of events to trigger. E.g. useful to look at the first 100 triggered events. Defaults to None, i.e. all events in the stream are triggered
    :type n_triggers: int
    :param preprocessing: Functions to apply to the voltage trace before determining the maxima. E.g. optimum filtering. Note that if `preprocessing` does not include `RemoveBaseline`, it is added as a first function in the preprocessing chain since baseline removal is mandatory. If the user wants to remove the baseline at another point in this chain, it has to be included in the `preprocessing` list. 
    :type preprocessing: Union[Callable, List[Callable]]

    :return: Tuple of trigger indices and trigger heights.
    :rtype: Tuple[List[int], List[float]]
    """
    
    if not isinstance(stream, StreamBaseClass):
        raise TypeError(f"Input argument 'stream' has to be of type 'StreamBaseClass', not {type(stream)}.")
    if key not in stream.keys:
        raise KeyError(f"Stream has no key '{key}'.")
    if overlap_fraction <= 0 or overlap_fraction > 0.25:
        raise ValueError("Input argument 'overlap_fraction' is out of range (0, 0.25].")
    if int(record_length*overlap_fraction/2)<1:
        raise ValueError("An overlap of at least 1 sample is required for baseline subtraction. The value you provided for 'overlap_fraction' resulted in an overlap of 0.")
    if trigger_block is not None and trigger_block < record_length:
        raise ValueError("Input argument 'trigger_block' has to be larger than 'record_length'.")
    
    if preprocessing is not None:
        if type(preprocessing) in [list, tuple]:
            if not all([callable(f) for f in preprocessing]):
                raise TypeError("All entries of list 'preprocessing' must be callable.")
        elif callable(preprocessing):
            preprocessing = [preprocessing]
        else:
            raise TypeError(f"Unsupported type '{type(preprocessing)}' for input argument 'preprocessing'.")
    else:
        preprocessing = []

    # If RemoveBaseline is not part of 'preprocessing', it is added (because removal of baseline is mandatory)
    # It is added to the front by default, i.e. baseline removal happens first.
    # If user wants RemoveBaseline at specific point in preprocessing chain, they just have to include it in the 'preprocessing' list
    if not any([isinstance(f, RemoveBaseline) for f in preprocessing]):
        preprocessing.insert(0, 
                             RemoveBaseline({"model":0,
                                             "where": overlap_fraction/2}))

    # Block trigger for one record window if not specified otherwise
    if trigger_block is None: trigger_block = record_length

    current_ind = int(0)
    max_ind = int(len(stream)-record_length)
    overlap = int(np.floor(record_length*overlap_fraction))
    # Slice to use for maximum search
    s_search = slice(overlap, -overlap)
    
    trigger_inds = []
    trigger_vals = []
    resampled = False

    print(f"Start triggering '{key}' of {stream} with threshold {threshold}, record length {record_length} and preprocessing:")
    print('\n'.join(['- '+str(i) for i in preprocessing]))
    
    with tqdm(total=max_ind) as progress:
        while current_ind < max_ind:
            # Slice to read from stream
            s = slice(current_ind, current_ind + record_length)

            # Read voltage trace from stream and apply preprocessing
            trace = stream[key, s, 'as_voltage']
            for p in preprocessing: trace = p(trace)

            # Read maximum index and value
            trigger_ind = np.argmax(trace[s_search])
            trigger_val = trace[s_search][trigger_ind]

            # Only executed when trigger candidate was found in previous iteration
            if resampled:
                trigger_inds.append(current_ind + overlap + trigger_ind)
                trigger_vals.append(trigger_val)
                resampled = False

                # We found a trigger: The absolute index of the trigger is (current_ind + overlap + trigger_ind). 
                # The next possible sample to trigger is (current_ind + overlap + trigger_ind + trigger_block).
                # By moving current_ind to (current_ind + overlap + trigger_ind + trigger_block - overlap), i.e. to (current_ind + trigger_ind + trigger_block) we implement the trigger block.
                current_ind += trigger_ind + trigger_block
                progress.update(trigger_ind + trigger_block)

                if (n_triggers is not None) and (len(trigger_inds) >= n_triggers):
                    break

            # Standard case. Finds trigger candidates
            elif trigger_val > threshold:
                # Move search window such that triggered sample is first sample to be searched ('trigger_ind' is index *within* search window)
                current_ind += trigger_ind
                progress.update(trigger_ind)
                # Search for larger values to the right of the trigger by re-running the step
                resampled = True

            # If not trigger candidate was found, we just move the record window to the right
            else:
                # Move window forward by record length and compensate overlap
                current_ind += record_length - 2*overlap 
                progress.update(record_length - 2*overlap)
            
    print(f"Found {len(trigger_inds)} triggers.")
    
    return trigger_inds, trigger_vals














def add_and_discard(input_array, new_element, max_size=5):
    """
    Adds a new element to the input array and discards the oldest element if the array exceeds a maximum size.

    :param input_array: The input array to which the new element will be added.
    :type input_array: list
    :param new_element: The new element to be added to the array.
    :type new_element: Any
    :param max_size: The maximum size of the array. Defaults to 5.
    :type max_size: int, optional
    :return: None
    """

    input_array.append(new_element)
    if len(input_array) > max_size:
        input_array.pop(0)


def Fourie_Trigger(stream, sigma=5.4, windowsize=710,windowsize_mean=34,stepsize=200,sampling_frequency=50000,blindingtime=2**16):

    """
    Detects events in a data stream  using Fourier analysis.

    :param stream: The input streaming signal.
    :type stream: array_like
    :param sigma: The threshold  for trigger for the mean trigger. Defaults to 5.4.
    :type sigma: float, optional
    :param windowsize: The size of the moving  window for FFT. Defaults to 710.
    :type windowsize: int, optional
    :param windowsize_mean: The size of the window for calculating mean and standard deviation in delta stream. Defaults to 34.
    :type windowsize_mean: int, optional
    :param stepsize: The step size defines jumps on stream .How many elements are pushed into the moving  window. Defaults to 200.
    :type stepsize: int, optional
    :param sampling_frequency: The sampling frequency of the signal. Defaults to 50000.
    :type sampling_frequency: int, optional
    :param blindingtime: The duration to skip after detecting a trigger to avoid multiple detections. Defaults to 65536.
    :type blindingtime: int, optional

    :return: Timestamps of detected triggers and their corresponding amplitudes.
    :rtype: tuple
    """
    timestamps=[]
    deltastream=[]
    diff=[0,0]                                                                                                            

    j=0
    dyn_factor=int(sampling_frequency/50000 )                                                                              #Facotr to adapt for different samplingrates
    upper_limit =(len(stream)-windowsize)/(stepsize*dyn_factor)
    amplitude=[]

    pbar = tqdm(total=upper_limit)                                                                                         #Progressbar
    while j<((len(stream)-windowsize)/(stepsize*dyn_factor)):    
        
        i=j*stepsize                                                                                                        # jump trough data
        window = stream[i:i+windowsize]                                                                                     # load data for window
        fs = sampling_frequency                                                                                             # Sampling frequency
        fft_result = np.fft.fft(window)                                                                                     #Caluclate fft and frequencies
        frequencies = np.fft.fftfreq(len(window), 1/fs)
        fft_result = fft_result[frequencies >= 0]
        frequencies = frequencies[frequencies >= 0]                                                     
        positive_frequencies_mask = frequencies <=25                                                                        #mask for frequencies  
        result=np.sum(fft_result[positive_frequencies_mask])                                                                #sum fft result for frequencies below 25
        if j==0:
            diff[1]=result
        else:                                                                                                              #
            diff[0]=diff[1]
            diff[1]=result
            add_and_discard(deltastream,diff[1]-diff[0],windowsize_mean)                                                   #add delta point to deltadatastream

        if j>windowsize_mean:
            mean=np.mean(deltastream[0:-1])
            std=np.std(deltastream[0:-1])
            if deltastream[-1]>sigma*std+mean:
                idx=i+windowsize-stepsize/4
                timestamps.append(idx)
                amplitude.append(np.max(stream[int(idx-200):int(idx)+200] )-np.mean(stream[int(idx-2000):int(idx)-200]))
                j+=int(blindingtime/(stepsize*dyn_factor))
                pbar.update(int(blindingtime/(stepsize*dyn_factor)))
        pbar.update(1)
        j+=1
    pbar.close()
    return timestamps,amplitude
