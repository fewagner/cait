import numpy as np
from scipy.signal import find_peaks
from joblib import Parallel,delayed
import bottleneck as bn
from tqdm.auto import tqdm
import random


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


def divide_array(input_array, subarray_length):
    """
    Divides the input array into subarrays of a specified length.

    :param input_array: The input array to be divided into subarrays.
    :type input_array: list
    :param subarray_length: The length of each subarray.
    :type subarray_length: int
    :return: A list containing subarrays of the input array.
    :rtype: list
    """

    Subarrays = []
    for i in range(0, len(input_array), subarray_length):
        subarray = input_array[i:i + subarray_length]
        if len(subarray) == subarray_length:
            Subarrays.append(subarray)
    return Subarrays




def LevelShift_detector(stream,record_length):
    """
    Detects level shifts in a given stream of data.

    :param stream: The stream of data.
    :type stream: numpy.ndarray
    :param record_lenght: Desired length of arrays
    :type record_lenght: int
    :return: A list of tuples representing the intervals where no level shifts were detected.
    :rtype: list
    """
    
    Intervall_begins = [0]                                                                                     #define begin of stream without level-shift
    Intervall_ends = []                                                                                        #define ending of stream without level-shift
    Streamlen = len(stream)


    # First rough Levelshift Search
    
    five_windows = []
    std_widows = []
    idx = []
    for i in tqdm(range(0, Streamlen, 300000),desc="Rough Level-Shift-Search"):                                 # jump trough data with stepsize 300000
        window = stream[i:i + 32000]                                                                            #load data from strea
        idx.append(i)                                                                                           # save indexnumber
        add_and_discard(five_windows, np.median(window), 5)                                                     # add new window to the five windows array
        std_widows.append(np.std(five_windows))                                                                 # calculate the std
    peaks, _ = find_peaks(std_widows, height=0.3*np.std(std_widows))                                            # search for large difference in  std


    # Second Levelshift Search
    anomalies = np.array(idx)[peaks]                                                                            # tag idx where level-shift was detecded
    for anomalie in tqdm(anomalies,desc="Fine Level-Shift-Search"):                                               #search with smaller step size close to the tagged levelshifts
        five_windows = []                                                                                         
        std_widows = []     
        idx = []    
        for i in (range(anomalie - int(0.8e6), anomalie, 8000)):
            window = stream[i:i + 16500]
            add_and_discard(five_windows, np.median(window))
            std_widows.append(np.std(five_windows))
        peak = np.argmax(std_widows)                                                                            #Search for maximum change in std
        peak = peak + (anomalie - int(0.8e6))
        Intervall_begins.append(peak + 32000)                                                                   #add begin and end of stream without level-shift
        Intervall_ends.append(peak - 32000)
    Intervall_ends.append(Streamlen)
    Interval_tuples = list(zip(Intervall_begins, Intervall_ends))                                               #generate tuples of (begin,end) of level-shift free stream
    Output = [(a, b) for (a, b) in Interval_tuples if b - a >= record_length]                                   #check if  substreams is longer than desired length
    return Output


def Fourie_Trigger(stream,Begin,End, sigma, windowsize,windowsize_mean,stepsize,sampling_frequency):

    """
    This function applies a Fourier frequency trigger to a given stream of data within specified intervals.

    :param stream: The input stream of data.
    :type stream: mmap
    :param Begin: The beginning index of the stream.
    :type Begin: int
    :param End: The ending index of the stream.
    :type End: int
    :param sigma: The sigma value for trigger detection.
    :type sigma: float
    :param windowsize: The size of the window for FFT.
    :type windowsize: int
    :param windowsize_mean: The size of the window for calculating mean and standard deviation for mean trigger.
    :type windowsize_mean: int
    :param stepsize: The step size for moving the window.
    :type stepsize: int
    :param sampling_frequency: The sampling frequency of the data.
    :type sampling_frequency: int

    :return: A list of tuples representing the intervals where no pulse were triggred
    :rtype: list(tuple)
    """

    tem = []                                                                                                    
    Beginning=[Begin]
    Ends=[]
    stream=stream[Begin:End]
    dyn_factor=int(sampling_frequency/50000  )                                                                                    #factor to ajust windowlength
    for i in range(0, len(stream)-windowsize*dyn_factor,stepsize*dyn_factor):                                               #jump trough data with stepsize 
        window = stream[i:i+windowsize]                                                                                     # load data
        fs = sampling_frequency                                                                                             # Sampling frequency
        fft_result = np.fft.fft(window)                                                                                     #Caluclate fft and frequencies
        frequencies = np.fft.fftfreq(len(window), 1/fs)
        fft_result = fft_result[frequencies >= 0]
        frequencies = frequencies[frequencies >= 0]                                                     
        positive_frequencies_mask = frequencies <=25                                                                        #mask for freequencies  
        tem.append(np.sum(fft_result[positive_frequencies_mask]))                                                           #sum fft result for frequencies below 25

                                                                                                                            #To Do: Make diff before buildinng fft
    delta_stream = np.diff(tem)                                                                                             # create delta stream by makinn the dif between values
    k=0
    

    while k <(len(delta_stream) - windowsize_mean-1):                                                                       #Apply mean trigger for deelta stream
        window = delta_stream[k:k + windowsize_mean]
        window_mean = np.mean(window)
        window_std = np.std(window)
        next_value = delta_stream[k + windowsize_mean+1]

        if next_value > (window_mean +sigma * window_std):
            k=k+windowsize_mean+1
            j=k*stepsize+windowsize
            Ends.append(j-8000+Begin)                                                                                       #add begin and end of stream without event
            Beginning.append(j+32000+Begin)
            k=k+int(40000/stepsize)
            
        k+=1    
    Ends.append(End)
    Interval_tuples = list(zip(Beginning, Ends))                                                                            #generate tuples of (begin,end) of level-shift free stream
    return Interval_tuples



def process_Intervals_Fourie(stream,Begin,End ,sigma, windowsize,windowsize_mean,stepsize,sampling_frequency):

    """
    This function processes intervals based on Fourier frequency trigger.

    :param stream: The input stream of data.
    :type stream: mmap
    :param Begin: The beginning index of the stream.
    :type Begin: int
    :param End: The ending index of the stream.
    :type End: int
    :param sigma: The sigma value for trigger detection.
    :type sigma: float
    :param windowsize: The size of the window for FFT.
    :type windowsize: int
    :param windowsize_mean: The size of the window for calculating mean and standard deviation for mean trigger.
    :type windowsize_mean: int
    :param stepsize: The step size for moving the window.
    :type stepsize: int
    :param sampling_frequency: The sampling frequency of the data.
    :type sampling_frequency: int

    :return: A list of tuples representing the intervals where no pulse were triggred
    :rtype: list(tuple)
    """


    temp_tuple=Fourie_Trigger(stream,Begin,End, sigma, windowsize,windowsize_mean,stepsize,sampling_frequency)
    Output_tuple=[]
    for tpl in temp_tuple:
        Output_tuple.append(tpl)
    return Output_tuple




def Apply_Fourie_Trigger(stream,Tuples,sampling_frequency,Nr_cores=-1,sigma=8, windowsize=300,windowsize_mean=5,stepsize=300,record_length=2**15):


    """

    This function applies Fourier frequency trigger to specified intervals in a stream of data.

    :param stream: The input stream of data.
    :type stream: mmap
    :param Tuples: A list of tuples representing intervals.
    :type Tuples: list(tuple)
    :param sampling_frequency: The sampling frequency of the data.
    :type sampling_frequency: int
    :param Nr_cores: Number of CPU cores to utilize for parallel processing. Defaults to -1.
    :type Nr_cores: int, optional
    :param sigma: The sigma value for trigger detection. Defaults to 8.
    :type sigma: float, optional
    :param windowsize: The size of the window for FFT. Defaults to 300.
    :type windowsize: int, optional
    :param windowsize_mean: The size of the window for calculating mean and standard deviation. Defaults to 5.
    :type windowsize_mean: int, optional
    :param stepsize: The step size for moving the window. Defaults to 300.
    :type stepsize: int, optional
    :param record_length: Minimum length of record. Defaults to 32768 (2**15).
    :type record_length: int, optional

    :return: A list of tuples representing the intervals where no pulse were triggred
    :rtype: list(tuple)
    """
    Output=[]
    Tem_Output=Parallel(n_jobs=Nr_cores)(delayed(process_Intervals_Fourie)(stream,Begin,End,sigma, windowsize,windowsize_mean,stepsize,sampling_frequency) for (Begin,End) in tqdm(Tuples,desc="Applying Frequency Trigger"))
    [Output.extend(inner_array) for inner_array in Tem_Output]                                                             #add clean arrays to Output
    Output = [(a, b) for (a, b) in Output if b - a >= record_length and a > 0]                                             #check if  substreams is longer than desired length
    return Output



def moving_average_and_std_trigger(stream, begin, end, sigma, window_size):
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
    stream_segment = stream[begin:end]
    stream_len = len(stream_segment)

    # Calculate moving average and standard deviation
    moving_avg = bn.move_mean(stream_segment, window=window_size, min_count=1)
    moving_std = bn.move_std(stream_segment, window=window_size, min_count=1)

    # Calculate threshold values
    threshold_high = moving_avg + sigma * moving_std
    threshold_low = moving_avg - sigma * moving_std

    i = 0
    while i < (stream_len - window_size - 1):
        if i > window_size:
            next_value = stream_segment[i + window_size]

            # Check if next value exceeds threshold
            if next_value > threshold_high[i] or next_value < threshold_low[i]:
                j = i + window_size
                if j - window_size - 8000 + begin > 0:
                    ends.append(j - window_size - 8000 + begin)
                else:
                    ends.append(0)
                beginning.append(j + 2**15 + begin)
                i = j + int(32768)
        i += 1

    ends.append(end)
    interval_tuples = list(zip(beginning, ends))
    return interval_tuples


def process_Intervals_Mean(stream, Begin, End, sigma=6, window_size=1500):
    """
    Process intervals for mean trigger detection.

    :param stream: The input stream of data.
    :type stream: array_like
    :param Begin: Start index of the segment to process.
    :type Begin: int
    :param End: End index of the segment to process.
    :type End: int
    :param sigma: Threshold factor for trigger detection (default is 6).
    :type sigma: float, optional
    :param window_size: Size of the window for moving average and standard deviation calculation (default is 1500).
    :type window_size: int, optional
    :return: List of tuples representing intervals where triggers were detected.
    :rtype: list
    """
    temp_tuple = moving_average_and_std_trigger(stream, Begin, End, sigma, window_size)
    Intervals = []
    for tpl in temp_tuple:
        Intervals.append(tpl)
    return Intervals


def Apply_Mean_Trigger(stream, Tuples, Nr_cores=-1, sigma=6, window_size=1500, record_length=2**15):
    """
    Apply Fourier mean trigger detection on multiple segments of a stream.

    :param stream: The input stream of data.
    :type stream: array_like
    :param Tuples: List of tuples representing segments of the stream to process.
    :type Tuples: list of tuples
    :param Nr_cores: Number of CPU cores to use for parallel processing (default is -1, using all available cores).
    :type Nr_cores: int, optional
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
    Tem_Output = Parallel(n_jobs=Nr_cores)(delayed(process_Intervals_Mean)(stream, Begin, End, sigma, window_size) for (Begin, End) in tqdm(Tuples,desc="Applying Mean Trigger"))

    [Output.extend(inner_array) for inner_array in Tem_Output]
    Output = [(a, b) for (a, b) in Output if b - a >= record_length and a > 0]
    return Output



def divide_array(Tuples,recordlenght=int(2**15)):

    """ 

    This function divides intervals into smaller segments of specified length.

    :param Tuples: A list of tuples representing intervals.
    :type Tuples: list(tuple)
    :param recordlenght: Length of the segments. Defaults to 2**15.
    :type recordlenght: int, optional

    :return: A list of tuples representing the divided segments.
    :rtype: list(tuple)
    """

    Output=[]
    for a,b in tqdm(Tuples,desc="Dividing array"):
        max_nr_array=int((b-a)/recordlenght)                                         # max Nr of baselines
        leftover=(b-a)-recordlenght*max_nr_array                                     #Specify the number of remaining samples if the maximum number of arrays is accommodated.
        while a+recordlenght<=b:                                                     # While the end of the current segment (a + record length) is less than or equal to the end of the interval (b)
            if leftover>0 :                                                            
                random_offset=random.randint(0, leftover)                              #Generate random offset
                leftover=leftover-random_offset     
            else:
                random_offset=0
            start=a+random_offset
            Output.append((start,start+recordlenght))                                   # Append the start representing the segment to the Output list
            a=start+recordlenght+1                                                      # Update the starting index of the next segment by shifting it forward by record length + 1
    return Output






def decaying_Baseline_remover(stream, Tuples):

    """
    Remove decaying baselines from segments of a stream.

    :param stream: The input stream of data.
    :type stream: array_like
    :param Tuples: List of tuples representing segments of the stream to process.
    :type Tuples: list of tuples
    :return: List of tuples representing segments where decaying baselines were removed.
    :rtype: list
    """

    Output=[]
    for (a,b) in tqdm(Tuples,desc="Removing decaying baselines"):  
        temp=stream[a:b]                                                                        #Load data
        sdt_dev=(np.std(temp[0:100])+np.std(temp[-100:-1]))/2                                   # calculate std of beegining and and of stream and build mean
        if abs(np.mean(temp[0:100])-np.mean(temp[-100:-1]))<sdt_dev:                            #check if mean(begin)-mean(end) differ
            Output.append(a)
    return Output




def get_clean_bs_idx(self,keys:str,record_length, **kwargs):
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

    remove_decaying_baseline = kwargs.get('remove_decaying_baseline', True)
    # Determine sampling frequency and adjust stream length
    sampling_frequency = int(1e6 / self.dt_us)
    # Detect level shifts and apply Fourier trigger
    Good_intervals = LevelShift_detector(stream=self[keys], record_length=record_length)
    Good_intervals = Apply_Fourie_Trigger(self[keys], Good_intervals, sampling_frequency=sampling_frequency, record_length=record_length)
    
    # Apply Fourier mean and divide array
    Good_intervals = Apply_Mean_Trigger(self[keys], Good_intervals, record_length=record_length)
    Good_intervals = divide_array(Good_intervals, recordlenght=record_length)

    # Optionally remove decaying baselines
    if remove_decaying_baseline:
        Good_intervals = decaying_Baseline_remover(self[keys], Tuples=Good_intervals)
    else:
        Good_intervals=[x[0] for x in Good_intervals]
    return Good_intervals





def get_clean_bs_idx_draft(self,keys:str,record_length, **kwargs):
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
    :param Nr_bslines: Nr of random choosen baselines (default is 300).
    :type Nr_bslines: int, optional
    :return: Noise Power Spectrum (NPS), frequency array, and good intervals.
    :rtype: tuple
    """
    # Determine factors

    remove_decaying_baseline = kwargs.get('remove_decaying_baseline', True)
    Nr_bslines= kwargs.get('Nr_bslines', 300)                                                              #Nr of desired baselines


    sampling_frequency = int(1e6 / self.dt_us)
    stream_length=len(self)

    
    subarray_length = 100000                                                                            #Legth of the array to search in
                                                                                             
    if stream_length - 10*subarray_length * Nr_bslines <= 0:                                            # Make sure there are enough samples in the stream to select from
        print("Stream length is too short to select indices with the given subarray length.")
        return 0,0
    else:
        random_indices = []
        save_counter=0
        
        while len(random_indices) < Nr_bslines:                                                         # Generate random indices with a minimum distance of subarray_length
            idx = random.randint(0, stream_length - subarray_length * Nr_bslines)
            if all(abs(idx - existing_idx) >= subarray_length for existing_idx in random_indices):
                random_indices.append(idx)
            save_counter+=1
            if save_counter>1000:
                break
        Good_intervals=[]
       
        for idx in random_indices:                                                                       #Define good intervalls
            Good_intervals.append((idx,idx+100000))


        Good_intervals = Apply_Fourie_Trigger(self[keys], Good_intervals, sampling_frequency=sampling_frequency, record_length=record_length)
        
        Good_intervals = Apply_Mean_Trigger(self[keys], Good_intervals, record_length=record_length)
        Good_intervals = divide_array(Good_intervals, recordlenght=record_length)

        # Optionally remove decaying baselines
        if remove_decaying_baseline:
            Good_intervals = decaying_Baseline_remover(self[keys], Tuples=Good_intervals)
        else:
            Good_intervals=[x[0] for x in Good_intervals]
        return Good_intervals
        