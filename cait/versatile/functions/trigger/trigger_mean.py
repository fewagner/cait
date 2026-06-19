# STILL NEEDS DOCSTRING
@nb.jit
def _search_zscores(zscores: ArrayLike, begin: int, threshold: float, window_size: int):
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


def _process_intervals_mean(chunk: Tuple[int, ArrayLike],
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

    a, b = _search_zscores(zscores, begin, sigma, window_size)
    beginnings.extend(a)
    ends.extend(b)

    ends.append(end)
    interval_tuples = list(zip(beginnings, ends))

    return interval_tuples.copy()

def trigger_mean(stream: ArrayLike,
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

    # out = []
    # f = partial(_process_intervals_mean,
    #             sigma=sigma, 
    #             window_size=window_size)

    filter_fnc = partial(_process_intervals_mean, 
                         sigma=sigma, 
                         window_size=window_size)

    # Trigger to get the trigger indices
    inds, _ =  trigger_base(stream=stream,
                            threshold=sigma,
                            filter_fnc=filter_fnc,
                            record_length=record_length,
                            n_triggers=n_triggers,
                            chunk_size=chunk_size,
                            apply_first=apply_first,
                            n_processes=n_processes)

    if not inds: return [], []
    
    # chunks = ((start, stream[start:end]) for start, end in tuples)
    
    # with Pool(n_cores) as pool:
    #     tem_out = list(tqdm(pool.imap(f, chunks), desc="Applying Mean Trigger", total=len(tuples)))

    # for inner_array in tem_out: out.extend(inner_array)
    
    # return [(a, b) for (a, b) in out if b - a >= record_length and a > 0]

    return [(a, b) for (a, b) in inds if b - a >= record_length and a > 0]

