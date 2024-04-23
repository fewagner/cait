from typing import List, Any
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.signal import find_peaks
from tqdm.auto import tqdm
import random

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

def level_shift_detector(stream, key: str, record_length: int):
    """
    Detects level shifts in a given stream of data.

    :param stream: The stream of data.
    :type stream: numpy.ndarray
    :param record_lenght: Desired length of arrays
    :type record_lenght: int
    :return: A list of tuples representing the intervals where no level shifts were detected.
    :rtype: list
    """
    # define begin of stream without level-shift
    Intervall_begins = [0]
    # define ending of stream without level-shift
    Intervall_ends = []

    Streamlen = len(stream)

    # First rough Levelshift Search
    five_windows = []
    std_widows = []
    idx = []

    # jump trough data with step size 300000
    for i in tqdm(range(0, Streamlen, 300000), desc="Rough Level-Shift-Search"):
        # load data from stream
        window = stream[key, i:i+32000]
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
    for anomalie in tqdm(anomalies, desc="Fine Level-Shift-Search"):
        five_windows = []         
        std_widows = []     
        idx = []    
        for i in (range(anomalie - int(0.8e6), anomalie, 8000)):
            window = stream[key, i:i + 16500]
            add_and_discard(five_windows, np.median(window))
            std_widows.append(np.std(five_windows))
        #Search for maximum change in std
        peak = np.argmax(std_widows)
        peak = peak + (anomalie - int(0.8e6))
        #add begin and end of stream without level-shift
        Intervall_begins.append(peak + 32000)
        Intervall_ends.append(peak - 32000)
    Intervall_ends.append(Streamlen)
    # generate tuples of (begin,end) of level-shift free stream
    Interval_tuples = list(zip(Intervall_begins, Intervall_ends))
    #check if  substreams is longer than desired length
    Output = [(a, b) for (a, b) in Interval_tuples if b - a >= record_length]

    return Output

def fourier_trigger(stream, 
                    key: str,
                    begin: int, 
                    end: int, 
                    sigma: float, 
                    window_size: int,
                    window_size_mean: int,
                    stepsize: int):
    """
    This function applies a Fourier frequency trigger to a given stream of data within specified intervals.

    :param stream: The input stream of data.
    :type stream: mmap
    :param begin: The beginning index of the stream.
    :type begin: int
    :param end: The ending index of the stream.
    :type end: int
    :param sigma: The sigma value for trigger detection.
    :type sigma: float
    :param window_size: The size of the window for FFT.
    :type window_size: int
    :param window_size_mean: The size of the window for calculating mean and standard deviation for mean trigger.
    :type window_size_mean: int
    :param stepsize: The step size for moving the window.
    :type stepsize: int

    :return: A list of tuples representing the intervals where no pulse were triggred
    :rtype: list(tuple)
    """
    tem = []               
    Beginning = [begin]
    Ends = []
    s = stream[key, begin:end]
    fs = int(1e6/stream.dt_us)
    #factor to ajust windowlength
    dyn_factor = int(fs/50000)

    #jump trough data with stepsize
    for i in range(0, len(s)-window_size*dyn_factor, stepsize*dyn_factor):
        window = s[i:i+window_size]
        #Caluclate fft and frequencies
        fft_result = np.fft.rfft(window)
        frequencies = np.fft.rfftfreq(len(window), 1/fs)
        # mask for frequencies
        frequency_mask = frequencies <= 25
        # sum fft result for frequencies below 25
        tem.append(np.sum(fft_result[frequency_mask]))

        #To Do: Make diff before buildinng fft

    # create delta stream by makinn the dif between values
    delta_stream = np.diff(tem)       
    
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
                Ends.append(j - 8000 + begin)
            else:
                Ends.append(begin)
            Beginning.append(j + 32000 + begin)
            k = k + 40000//stepsize
            
        k += 1  

    Ends.append(end)
    #generate tuples of (begin,end) of level-shift free stream
    Interval_tuples = list(zip(Beginning, Ends))
    
    return Interval_tuples

def process_intervals_fourier(begin: int,
                              end: int,
                              stream,
                              key: str,
                              sigma: float,
                              window_size: int,
                              window_size_mean: int,
                              stepsize: int):
    """
    This function processes intervals based on Fourier frequency trigger.

    :param stream: The input stream of data.
    :type stream: mmap
    :param begin: The beginning index of the stream.
    :type begin: int
    :param end: The ending index of the stream.
    :type end: int
    :param sigma: The sigma value for trigger detection.
    :type sigma: float
    :param window_size: The size of the window for FFT.
    :type window_size: int
    :param window_size_mean: The size of the window for calculating mean and standard deviation for mean trigger.
    :type window_size_mean: int
    :param stepsize: The step size for moving the window.
    :type stepsize: int

    :return: A list of tuples representing the intervals where no pulse were triggered
    :rtype: list(tuple)
    """
    temp_tuple = fourier_trigger(stream, key, begin, end, sigma, window_size, window_size_mean, stepsize)

    return temp_tuple.copy()

def apply_fourier_trigger(stream,
                          key: str,
                          tuples: list,
                          n_cores: int = None,
                          sigma: float = 8, 
                          window_size: int = 300,
                          window_size_mean: int = 5,
                          stepsize: int = 300,
                          record_length: int = 2**15):
    """
    This function applies Fourier frequency trigger to specified intervals in a stream of data.

    :param stream: The input stream of data.
    :type stream: mmap
    :param tuples: A list of tuples representing intervals.
    :type tuples: list(tuple)
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

    :return: A list of tuples representing the intervals where no pulse were triggred
    :rtype: list(tuple)
    """
    Output = []
    f = partial(process_intervals_fourier, 
                stream=stream,
                key=key,
                sigma=sigma, 
                window_size=window_size, 
                window_size_mean=window_size_mean, 
                stepsize=stepsize)
    
    with Pool(n_cores) as pool:
        Tem_Output = list(pool.starmap(f, tqdm(tuples, desc="Applying Frequency Trigger")))
      
    # add clean arrays to Output
    for inner_array in Tem_Output: Output.extend(inner_array)  
    
    # check if sub stream is longer than desired length
    Output = [(a, b) for (a, b) in Output if b - a >= record_length and a > 0]                                             
    
    return Output

def moving_average_and_std_trigger(stream, 
                                   key: int, 
                                   begin: int, 
                                   end: int, 
                                   sigma: float, 
                                   window_size: int):
    """
    Detect intervals where the stream values deviate from the moving average.

    :param stream: The input stream of data.
    :type stream: array_like
    :param begin: Start index of the segment to process.
    :type begin: int
    :param end: End index of the segment to process.
    :type end: int
    :param sigma: Threshold factor for trigger detection.
    :type sigma: float
    :param window_size: Size of the window for moving average and standard deviation calculation.
    :type window_size: int
    :return: List of tuples representing intervals where triggers were detected.
    :rtype: list
    """
    beginning = [begin]
    ends = []

    # Extract segment of the stream
    stream_segment =stream[key, begin:end]
    stream_len = len(stream_segment)
    stream_segment=np.array(stream_segment)/min(stream_segment)
    #stream_segment=np.array(stream_segment)
    window = np.ones(window_size) / window_size
    moving_avg = np.convolve(stream_segment, window, mode='valid')
    # Compute the rolling standard deviation using the formula std = sqrt(E[X^2] - (E[X])^2)
    rolling_squared_mean = np.convolve(stream_segment**2, window, mode='valid')
    moving_std = np.sqrt(rolling_squared_mean - moving_avg**2)



    # Calculate threshold values
    threshold_high = moving_avg + sigma * moving_std
    threshold_low = moving_avg - sigma * moving_std

    i = 0
    while i < (stream_len - window_size - 1):
        next_value = stream_segment[i + window_size]

        # Check if next value exceeds threshold
        if next_value > threshold_high[i] or next_value < threshold_low[i]:
            j = i 
            if j  - 8000 + begin > 0:
                ends.append(j - window_size - 8000 + begin)
            else:
                ends.append(0)
            beginning.append(j + 2**15 + begin)
            i = j + int(32768)
        i += 1

    ends.append(end)
    interval_tuples = list(zip(beginning, ends))

    return interval_tuples

def process_intervals_mean(begin: int, 
                           end: int, 
                           stream, 
                           key: str, 
                           sigma: float = 6, 
                           window_size: int = 1500):
    """
    Process intervals for mean trigger detection.

    :param stream: The input stream of data.
    :type stream: array_like
    :param begin: Start index of the segment to process.
    :type begin: int
    :param end: End index of the segment to process.
    :type end: int
    :param sigma: Threshold factor for trigger detection (default is 6).
    :type sigma: float, optional
    :param window_size: Size of the window for moving average and standard deviation calculation (default is 1500).
    :type window_size: int, optional
    :return: List of tuples representing intervals where triggers were detected.
    :rtype: list
    """
    temp_tuple = moving_average_and_std_trigger(stream, key, begin, end, sigma, window_size)

    return temp_tuple.copy()

def apply_mean_trigger(stream,
                       key: str,
                       tuples: list, 
                       n_cores: int = None, 
                       sigma: float = 6, 
                       window_size: int = 1500, 
                       record_length: int = 2**15):
    """
    Apply Fourier mean trigger detection on multiple segments of a stream.

    :param stream: The input stream of data.
    :type stream: array_like
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
    Output = []
    f = partial(process_intervals_mean, 
                stream=stream,
                key=key,
                sigma=sigma, 
                window_size=window_size)
    
    with Pool(n_cores) as pool:
        Tem_Output = list(pool.starmap(f, tqdm(tuples, desc="Applying Mean Trigger")))

    for inner_array in Tem_Output: Output.extend(inner_array)
    
    Output = [(a, b) for (a, b) in Output if b - a >= record_length and a > 0]
    
    return Output

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
    Output = []
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
            Output.append((start,start+record_length))
            # Update the starting index of the next segment by shifting it forward by record length + 1
            a = start + record_length + 1

    return Output

def decaying_baseline_remover(stream, key: str, tuples: list):
    """
    Remove decaying baselines from segments of a stream.

    :param stream: The input stream of data.
    :type stream: array_like
    :param tuples: List of tuples representing segments of the stream to process.
    :type tuples: list of tuples
    :return: List of tuples representing segments where decaying baselines were removed.
    :rtype: list
    """
    Output = []
    for (a,b) in tqdm(tuples, desc="Removing decaying baselines"):  
        temp = stream[key, a:b]
        # calculate std of beginning and and of stream and build mean
        sdt_dev = (np.std(temp[0:100]) + np.std(temp[-100:-1]))/2

        #check if mean(begin)-mean(end) differ
        if abs(np.mean(temp[0:100]) - np.mean(temp[-100:-1])) < sdt_dev:
            Output.append(a)

    return Output

def get_clean_bs_idx(stream,
                     key: str,
                     record_length: int, 
                     remove_decaying_baseline: bool = True):
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
    :return: Noise Power Spectrum (NPS), frequency array, and good intervals.
    :rtype: tuple
    """
    # Detect level shifts and apply Fourier trigger
    Good_intervals = level_shift_detector(stream=stream, key=key, record_length=record_length)
    Good_intervals = apply_fourier_trigger(stream, key, Good_intervals, record_length=record_length)
    
    # Apply Fourier mean and divide array
    Good_intervals = apply_mean_trigger(stream, key, Good_intervals, record_length=record_length)
    Good_intervals = divide_array(Good_intervals, record_length=record_length)

    # Optionally remove decaying baselines
    if remove_decaying_baseline:
        Good_intervals = decaying_baseline_remover(stream, key, tuples=Good_intervals)
    else:
        Good_intervals = [x[0] for x in Good_intervals]
    return Good_intervals

def get_clean_bs_idx_draft(stream, 
                           key: str, 
                           record_length: int,
                           remove_decaying_baseline: bool = True,
                           n_bslines: int = 300):
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
    :param n_bslines: Nr of random choosen baselines (default is 300).
    :type n_bslines: int, optional
    :return: Noise Power Spectrum (NPS), frequency array, and good intervals.
    :rtype: tuple
    """
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

        Good_intervals=[]
        
        #Define good intervals
        for idx in random_indices: 
            Good_intervals.append((idx, idx+100000))

        #Good_intervals = apply_fourier_trigger(stream, key, Good_intervals, record_length=record_length)
        
        Good_intervals = apply_mean_trigger(stream, key, Good_intervals, record_length=record_length)

        Good_intervals = divide_array(Good_intervals, record_length=record_length)

        # Optionally remove decaying baselines
        if remove_decaying_baseline:
            Good_intervals = decaying_baseline_remover(stream, key, tuples=Good_intervals)
        else:
            Good_intervals = [x[0] for x in Good_intervals]
        
        return Good_intervals