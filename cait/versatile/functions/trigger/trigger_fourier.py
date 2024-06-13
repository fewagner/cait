from contextlib import nullcontext

import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from ..nps_auto.get_clean_bs_idx import add_and_discard

def trigger_fourier(stream: ArrayLike,
                    record_length: int,
                    dt_us: int,
                    sigma: float = 5.4, 
                    windowsize: int = 710,
                    windowsize_mean: int = 34,
                    stepsize: int = 200):
    """
    Use a Fourier based trigger algorithm to detect pulses in a raw data stream.

    :param stream: The stream to trigger
    :type stream: ArrayLike
    :param record_length: The desired record length of the events (determines the blinding time after a found trigger).
    :type record_length: int
    :param dt_us: The timebase used for the recorded data in microseconds.
    :type dt_us: int
    :param sigma: The threshold for the mean trigger. Defaults to 5.4.
    :type sigma: float, optional
    :param windowsize: The size of the moving FFT window. Defaults to 710.
    :type windowsize: int, optional
    :param windowsize_mean: The size of the window for calculating mean and standard deviation in delta stream. Defaults to 34.
    :type windowsize_mean: int, optional
    :param stepsize: The step size defines jumps on the stream. How many elements are pushed into the moving window. Defaults to 200.
    :type stepsize: int, optional

    :return: Indices of detected triggers and their corresponding amplitudes (ADC values).
    :rtype: tuple

    **Example:**
    ::
        import cait.versatile as vai

        # Construct stream object
        stream = vai.Stream(hardware="vdaq2", src="path/to/stream_file.bin")
        # Perform triggering
        trigger_inds, amplitudes = vai.trigger_fourier(stream["ADC1"], 2**16, stream.dt_us)
        # Get trigger timestamps from trigger indices
        timestamps = stream.time[trigger_inds]
        # Plot trigger amplitude spectrum
        vai.Histogram(amplitudes)
    """

    trigger_inds, amplitudes, triggers_found = [], [], 0

    deltastream = []
    diff = [0, 0]

    fs = int(1e6/dt_us)

    #Factor to adapt for different sampling frequencies (algorithm optimized for 50 kHz)
    dyn_factor = fs/50000
    stepsize = int(stepsize*dyn_factor)
    step_length = int(record_length//stepsize)

    n_steps = int(len(stream) - windowsize)//stepsize

    j = 0
    # StreamBaseClass can be kept open in a context. To make it work also with regular
    # arrays, we differentiate here via the existence of an '__enter__' method
    with stream if hasattr(stream, "__enter__") else nullcontext(stream) as s:
        with tqdm(total=n_steps) as pbar:
            while j < n_steps:    
                # jump trough data
                i = j*stepsize
                # load data for window                        
                window = s[i:i+windowsize]
                # calculate FFT and frequencies
                fft_result = np.fft.rfft(window)      
                frequencies = np.fft.rfftfreq(len(window), 1/fs)
                # mask for frequencies  
                frequency_mask = frequencies <= 25
                # sum fft result for frequencies below 25
                result = np.sum(fft_result[frequency_mask])

                if j==0:
                    diff[1] = result
                else:
                    diff[0] = diff[1]
                    diff[1] = result
                    # add delta point to delta data stream
                    add_and_discard(deltastream, diff[1]-diff[0], windowsize_mean)

                if j > windowsize_mean:
                    mean = np.mean(deltastream)
                    std = np.std(deltastream)
                    if deltastream[-1] > (mean + sigma*std):
                        idx = int(i + windowsize - stepsize//4)

                        trigger_inds.append(idx)
                        amplitudes.append(
                            np.max(s[int(idx-200):int(idx)+200]) - np.mean(s[int(idx-2000):int(idx)-200])
                        )
                        triggers_found += 1

                        j += step_length

                        pbar.update(step_length)
                        pbar.set_postfix({"triggers found": triggers_found})

                pbar.update(1)

                j += 1

    return trigger_inds, amplitudes