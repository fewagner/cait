import numpy as np
from tqdm.auto import tqdm

from ..nps_auto.get_clean_bs_idx import add_and_discard

def trigger_fourier(stream, 
                    key: str,
                    record_length: int,
                    sigma: float = 5.4, 
                    windowsize: int = 710,
                    windowsize_mean: int = 34,
                    stepsize: int = 200):
    """
    Use a Fourier based trigger algorithm to detect pulses in a raw data stream.

    :param stream: The stream to trigger
    :type stream: StreamBaseClass
    :param key: The key of the channel in the stream that you want to trigger.
    :type key: string
    :param record_length: The desired record length of the events (determines the blinding time after a found trigger).
    :type record_length: int
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
        trigger_inds, amplitudes = vai.trigger_fourier(stream, "ADC1", 2**16)
        # Get trigger timestamps from trigger indices
        timestamps = stream.time[trigger_inds]
        # Plot trigger amplitude spectrum
        vai.Histogram(amplitudes)
    """

    trigger_inds = []
    amplitudes = []

    deltastream = []
    diff = [0, 0]

    fs = int(1e6/stream.dt_us)

    #Factor to adapt for different sampling frequencies
    dyn_factor = int(fs/50000)
    upper_limit = (len(stream) - windowsize)//(stepsize*dyn_factor)

    j = 0
    with tqdm(total=upper_limit) as pbar:
        while j < (len(stream) - windowsize)/(stepsize*dyn_factor):    
            # jump trough data
            i = j*stepsize 
            # load data for window                                   
            window = stream[key, i:i+windowsize]
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
                    idx = i + windowsize - stepsize//4

                    trigger_inds.append(idx)
                    amplitudes.append(
                        np.max(stream[key, int(idx-200):int(idx)+200]) - np.mean(stream[key, int(idx-2000):int(idx)-200])
                    )
                    j += record_length//(stepsize*dyn_factor)

                    pbar.update(record_length//(stepsize*dyn_factor))

            pbar.update(1)

            j += 1

    return trigger_inds, amplitudes