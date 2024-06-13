from typing import List, Any, Tuple
from functools import partial
from multiprocessing import Pool

import numpy as np
import numba as nb
from numpy.typing import ArrayLike
from scipy.signal import find_peaks
from tqdm.auto import tqdm
import random

from ..trigger.trigger_zscore import zscore_chunk

# STILL NEEDS DOCSTRING
@nb.jit
def search_zscores(zscores: ArrayLike, begin: int, threshold: float, window_size: int):
    data_len = len(zscores)
    ends, beginnings = [], []

    i = 0
    while i < (data_len - window_size - 1):
        # Check if next value exceeds threshold
        if zscores[i] > threshold or zscores[i] < -threshold:
            j = i 
            if j - 8000 + begin > 0:
                ends.append(j - window_size - 8000 + begin)
            else:
                ends.append(0)
            beginnings.append(j + 2**15 + begin)
            i = j + int(32768)
        i += 1

    return beginnings, ends

# STILL NEEDS DOCSTRING
@nb.jit
def search_deltastream(delta_stream: ArrayLike, 
                       begin: int, 
                       sigma: float, 
                       window_size: int,
                       window_size_mean: int,
                       stepsize: int):
    ends, beginnings = [], []
    k = 0
    #Apply mean trigger for delta stream
    while k < (len(delta_stream) - window_size_mean - 1):
        window = delta_stream[k:k+window_size_mean]
        window_mean = np.mean(window)
        window_std = np.std(window)
        next_value = delta_stream[k + window_size_mean + 1]

        if next_value > (window_mean + sigma*window_std):
            k = k + window_size_mean + 1
            j = k*stepsize + window_size
            #add begin and end of stream without event
            if (j - 8000 + begin)>0:
                ends.append(j - 8000 + begin)
            else:
                ends.append(begin)
            beginnings.append(j + 32000 + begin)
            k = k + 40000//stepsize
            
        k += 1

    return beginnings, ends

def add_and_discard(input_array: List, new_element: Any, max_size: int = 5):
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

def divide_array(tuples: list, record_length: int):
    """
    This function divides intervals into smaller segments of specified length.

    :param tuples: A list of tuples representing intervals.
    :type tuples: list(tuple)
    :param record_length: Length of the segments. Defaults to 2**15.
    :type record_length: int

    :return: A list of tuples representing the divided segments.
    :rtype: list(tuple)
    """
    out = []
    for a,b in tqdm(tuples, desc="Dividing array"):
        # max Nr of baselines
        max_nr_array = int((b-a)/record_length)
        # Specify the number of remaining samples if the maximum number of arrays is accommodated.
        leftover = (b-a) - record_length*max_nr_array
        # While the end of the current segment (a + record length) is less than or equal to the end of the interval (b)
        while (a + record_length) <= b:
            if leftover>0 :
                #Generate random offset
                random_offset = random.randint(0, leftover)
                leftover = leftover - random_offset     
            else:
                random_offset = 0
            start = a + random_offset
            # Append the start representing the segment to the Output list
            out.append((start,start+record_length))
            # Update the starting index of the next segment by shifting it forward by record length + 1
            a = start + record_length + 1

    return out

def level_shift_detector(stream: ArrayLike, record_length: int):
    """
    Detects level shifts in a given stream of data.

    :param stream: The stream of data.
    :type stream: ArrayLike
    :param record_length: Desired length of arrays
    :type record_length: int
    :return: A list of tuples representing the intervals where no level shifts were detected.
    :rtype: list
    """
    # define begin of stream without level-shift
    int_starts = [0]
    # define ending of stream without level-shift
    int_ends = []

    stream_length = len(stream)

    # First rough Levelshift Search
    five_windows = []
    std_widows = []
    idx = []

    # jump trough data with step size 300000
    for i in tqdm(range(0, stream_length, 300000), desc="Rough Level-Shift-Search"):
        # load data from stream
        window = stream[i:i+32000]
        # save index number
        idx.append(i)

        # add new window to the five windows array
        add_and_discard(five_windows, np.median(window), 5)
        # calculate the std
        std_widows.append(np.std(five_windows))

    # search for large difference in  std
    peaks, _ = find_peaks(std_widows, height=0.3*np.std(std_widows))

    # second Levelshift Search
    # tag idx where level-shift was detected
    anomalies = np.array(idx)[peaks]
    
    #search with smaller step size close to the tagged levelshifts
    for anomaly in tqdm(anomalies, desc="Fine Level-Shift-Search"):
        five_windows = []         
        std_widows = []     
        idx = []    
        for i in (range(anomaly - int(0.8e6), anomaly, 8000)):
            window = stream[i:i + 16500]
            add_and_discard(five_windows, np.median(window))
            std_widows.append(np.std(five_windows))
        #Search for maximum change in std
        peak = np.argmax(std_widows)
        peak = peak + (anomaly - int(0.8e6))
        #add begin and end of stream without level-shift
        int_starts.append(peak + 32000)
        int_ends.append(peak - 32000)
    int_ends.append(stream_length)
    # generate tuples of (begin,end) of level-shift free stream
    intervals = zip(int_starts, int_ends)
    #check if  substreams is longer than desired length
    return [(a, b) for (a, b) in intervals if b - a >= record_length]

def process_intervals_fourier(chunk: Tuple[int, ArrayLike],
                              sigma: float,
                              dt_us: int,
                              window_size: int,
                              window_size_mean: int,
                              stepsize: int):
    """
    This function processes intervals based on Fourier frequency trigger.

    :param chunk: The beginning index of the interval in the original stream and the interval of data.
    :type chunk: Tuple[int, ArrayLike]
    :param sigma: The sigma value for trigger detection.
    :type sigma: float
    :param dt_us: Microsecond timebase of the recording.
    :type dt_us: int
    :param window_size: The size of the window for FFT.
    :type window_size: int
    :param window_size_mean: The size of the window for calculating mean and standard deviation for mean trigger.
    :type window_size_mean: int
    :param stepsize: The step size for moving the window.
    :type stepsize: int

    :return: A list of tuples representing the intervals where no pulse were triggered
    :rtype: list(tuple)
    """
    begin, interval = chunk
    tem = []
    end = begin + len(interval)
    starts = [begin]
    ends = []
    fs = int(1e6/dt_us)
    #factor to adjust window_length
    dyn_factor = fs/50000
    window_size = int(window_size*dyn_factor)
    stepsize = int(stepsize*dyn_factor)

    #jump trough data with stepsize
    for i in range(0, len(interval)-window_size, stepsize):
        window = interval[i:i+window_size]
        #Calculate fft and frequencies
        fft_result = np.fft.rfft(window)
        frequencies = np.fft.rfftfreq(len(window), 1/fs)
        # mask for frequencies
        frequency_mask = frequencies <= 25
        # sum fft result for frequencies below 25
        tem.append(np.sum(np.abs(fft_result[frequency_mask])))

        #To Do: Make diff before building fft

    # create delta stream by making the dif between values
    delta_stream = np.diff(tem)       
     
    #Apply mean trigger for delta stream
    a, b = search_deltastream(delta_stream, begin, sigma, window_size, window_size_mean, stepsize)
    starts.extend(a)
    ends.extend(b)

    ends.append(end)
    #generate tuples of (begin,end) of level-shift free stream
    intervals = list(zip(starts, ends))
    
    return intervals.copy()

def process_intervals_mean(chunk: Tuple[int, ArrayLike],
                           sigma: float = 6, 
                           window_size: int = 1500):
    """
    Process intervals for mean trigger detection.

    :param interval: The beginning index of the interval in the original stream and the interval of data.
    :type interval: Tuple[int, ArrayLike]
    :param sigma: Threshold factor for trigger detection (default is 6).
    :type sigma: float, optional
    :param window_size: Size of the window for moving average and standard deviation calculation (default is 1500).
    :type window_size: int, optional
    :return: List of tuples representing intervals where triggers were detected.
    :rtype: list
    """
    begin, interval = chunk
    beginnings, ends = [begin], []

    stream_len = len(interval)
    end = begin + stream_len

    zscores = zscore_chunk(interval, window_size)

    a, b = search_zscores(zscores, begin, sigma, window_size)
    beginnings.extend(a)
    ends.extend(b)

    ends.append(end)
    interval_tuples = list(zip(beginnings, ends))

    return interval_tuples.copy()

def apply_fourier_trigger(stream: ArrayLike,
                          tuples: list,
                          dt_us: int,
                          n_cores: int = None,
                          sigma: float = 8,
                          window_size: int = 300,
                          window_size_mean: int = 5,
                          stepsize: int = 300,
                          record_length: int = 2**15):
    """
    This function applies Fourier frequency trigger to specified intervals in a stream of data.

    :param stream: The input stream of data.
    :type stream: ArrayLike
    :param tuples: A list of tuples representing intervals.
    :type tuples: list(tuple)
    :param dt_us: The microsecond timebase of the recording.
    :type dt_us: int
    :param n_cores: Number of CPU cores to utilize for parallel processing. Defaults to None, i.e. as many as possible.
    :type n_cores: int, optional
    :param sigma: The sigma value for trigger detection. Defaults to 8.
    :type sigma: float, optional
    :param window_size: The size of the window for FFT. Defaults to 300.
    :type window_size: int, optional
    :param window_size_mean: The size of the window for calculating mean and standard deviation. Defaults to 5.
    :type window_size_mean: int, optional
    :param stepsize: The step size for moving the window. Defaults to 300.
    :type stepsize: int, optional
    :param record_length: Minimum length of record. Defaults to 32768 (2**15).
    :type record_length: int, optional

    :return: A list of tuples representing the intervals where no pulse were triggered
    :rtype: list(tuple)
    """
    out = []
    f = partial(process_intervals_fourier,
                sigma=sigma,
                dt_us=dt_us,
                window_size=window_size, 
                window_size_mean=window_size_mean, 
                stepsize=stepsize)

    chunks = ((start, stream[start:end]) for start, end in tuples)

    with Pool(n_cores) as pool:
        tem_out = list(tqdm(pool.imap(f, chunks), desc="Applying Frequency Trigger", total=len(tuples)))
      
    # add clean arrays to Output
    for inner_array in tem_out: out.extend(inner_array)  
    
    # check if sub stream is longer than desired length
    return [(a, b) for (a, b) in out if b - a >= record_length and a > 0]                           

def apply_mean_trigger(stream: ArrayLike,
                       tuples: list, 
                       n_cores: int = 1, 
                       sigma: float = 6, 
                       window_size: int = 1500, 
                       record_length: int = 2**15):
    """
    Apply Fourier mean trigger detection on multiple segments of a stream.

    :param stream: The input stream of data.
    :type stream: ArrayLike
    :param tuples: List of tuples representing segments of the stream to process.
    :type tuples: list of tuples
    :param n_cores: Number of CPU cores to use for parallel processing (default is -1, using all available cores).
    :type n_cores: int, optional
    :param sigma: Threshold factor for trigger detection (default is 6).
    :type sigma: float, optional
    :param window_size: Size of the window for moving average and standard deviation calculation (default is 1500).
    :type window_size: int, optional
    :param record_length: Length of each record (default is 2**15).
    :type record_length: int, optional
    :return: List of tuples representing intervals where triggers were detected.
    :rtype: list
    """
    out = []
    f = partial(process_intervals_mean,
                sigma=sigma, 
                window_size=window_size)
    
    chunks = ((start, stream[start:end]) for start, end in tuples)
    
    with Pool(n_cores) as pool:
        tem_out = list(tqdm(pool.imap(f, chunks), desc="Applying Mean Trigger", total=len(tuples)))

    for inner_array in tem_out: out.extend(inner_array)
    
    return [(a, b) for (a, b) in out if b - a >= record_length and a > 0]

def decaying_baseline_remover(stream: ArrayLike, tuples: list):
    """
    Remove decaying baselines from segments of a stream.

    :param stream: The input stream of data.
    :type stream: ArrayLike
    :param tuples: List of tuples representing segments of the stream to process.
    :type tuples: list of tuples
    :return: List of tuples representing segments where decaying baselines were removed.
    :rtype: list
    """
    out = []
    for (a,b) in tqdm(tuples, desc="Removing decaying baselines"):  
        temp = stream[a:b]
        # calculate std of beginning and and of stream and build mean
        sdt_dev = (np.std(temp[0:100]) + np.std(temp[-100:-1]))/2

        #check if mean(begin)-mean(end) differ
        if abs(np.mean(temp[0:100]) - np.mean(temp[-100:-1])) < sdt_dev:
            out.append(a)

    return out

def get_clean_bs_idx(stream: ArrayLike,
                     record_length: int,
                     dt_us: int,
                     remove_decaying_baseline: bool = True,
                     **kwargs):
    """
    Process stream data to extract good intervals and generate Noise Power Spectrum (NPS).

    :param VDAQ: Hardware identifier for VDAQ (default is "vdaq2").
    :type VDAQ: str, optional
    :param channel:  Identifier for the desired channel (default is "ADC1").
    :type channel: str, optional
    :param src: Filepath to the source data file (default is "../Data/test_bck_004v2.bin").
    :type src: str, optional
    :param Fit_order: Order of the polynomial fit for baseline removal (default is 0).
    :type Fit_order: int, optional
    :param record_length: Length of each record (default is 2**15).
    :type record_length: int, optional
    :param remove_decaying_baseline: Flag to indicate whether to remove decaying baselines (default is True).
    :type remove_decaying_baseline: bool, optional
    :param kwargs: Additional keyword arguments which are passed to 'apply_fourier_trigger' and/or 'apply_mean_trigger'.
    :type kwargs: any, optional

    :return: Noise Power Spectrum (NPS), frequency array, and good intervals.
    :rtype: tuple
    """
    # separate kwargs
    fourier_kwargs = {k:kwargs[k] for k in kwargs if k in ["n_cores", "sigma", "window_size", 
                                                           "window_size_mean", "stepsize"]}
    mean_kwargs = {k:kwargs[k] for k in kwargs if k in ["n_cores", "sigma", "window_size"]}

    # Detect level shifts and apply Fourier trigger
    good_intervals = level_shift_detector(stream=stream, 
                                          record_length=record_length)
    
    good_intervals = apply_fourier_trigger(stream=stream, 
                                           tuples=good_intervals, 
                                           dt_us=dt_us, 
                                           record_length=record_length,
                                           **fourier_kwargs)
    
    # Apply Fourier mean and divide array
    good_intervals = apply_mean_trigger(stream=stream, 
                                        tuples=good_intervals, 
                                        record_length=record_length,
                                        **mean_kwargs)
    
    good_intervals = divide_array(tuples=good_intervals, 
                                  record_length=record_length)

    # Optionally remove decaying baselines
    if remove_decaying_baseline:
        good_intervals = decaying_baseline_remover(stream, tuples=good_intervals)
    else:
        good_intervals = [x[0] for x in good_intervals]
    return good_intervals

def get_clean_bs_idx_draft(stream: ArrayLike,
                           record_length: int,
                           remove_decaying_baseline: bool = True,
                           n_bslines: int = 300,
                           **kwargs):
    """
    Selects random positions in the stream to extract good intervals and generate a fast Noise Power Spectrum (NPS), improve.

    :param VDAQ: Hardware identifier for VDAQ (default is "vdaq2").
    :type VDAQ: str, optional
    :param channel:  Identifier for the desired channel (default is "ADC1").
    :type channel: str, optional
    :param src: Filepath to the source data file (default is "../Data/test_bck_004v2.bin").
    :type src: str, optional
    :param Fit_order: Order of the polynomial fit for baseline removal (default is 0).
    :type Fit_order: int, optional
    :param record_length: Length of each record (default is 2**15).
    :type record_length: int, optional
    :param remove_decaying_baseline: Flag to indicate whether to remove decaying baselines (default is True).
    :type remove_decaying_baseline: bool, optional
    :param n_bslines: Nr of random chosen baselines (default is 300).
    :type n_bslines: int, optional
    :param kwargs: Additional keyword arguments which are passed to 'apply_mean_trigger'.
    :type kwargs: any, optional

    :return: Noise Power Spectrum (NPS), frequency array, and good intervals.
    :rtype: tuple
    """
    mean_kwargs = {k:kwargs[k] for k in kwargs if k in ["n_cores", "sigma", "window_size"]}

    stream_length = len(stream)

    # length of the array to search in
    subarray_length = 100000

    # Make sure there are enough samples in the stream to select from
    if (stream_length - 10*subarray_length*n_bslines) <= 0:
        print("Stream length is too short to select indices with the given subarray length.")
        return 0, 0
    else:
        random_indices = []
        save_counter = 0
        
        # Generate random indices with a minimum distance of subarray_length
        while len(random_indices) < n_bslines:
            idx = random.randint(0, stream_length - subarray_length * n_bslines)
            if all(abs(idx - existing_idx) >= subarray_length for existing_idx in random_indices):
                random_indices.append(idx)
            save_counter += 1
            if save_counter>1000:
                break

        good_intervals=[]
        
        #Define good intervals
        for idx in random_indices: 
            good_intervals.append((idx, idx+100000))

        #good_intervals = apply_fourier_trigger(stream, key, good_intervals, record_length=record_length)
        
        good_intervals = apply_mean_trigger(stream=stream, 
                                            tuples=good_intervals, 
                                            record_length=record_length, 
                                            **mean_kwargs)

        good_intervals = divide_array(tuples=good_intervals, record_length=record_length)

        # Optionally remove decaying baselines
        if remove_decaying_baseline:
            good_intervals = decaying_baseline_remover(stream, tuples=good_intervals)
        else:
            good_intervals = [x[0] for x in good_intervals]
        
        return good_intervals