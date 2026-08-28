from re import I
from typing import List, Tuple, Union
import warnings

import numba
import numpy as np

from ...analysisobjects import OF, SEV
from ..functionbase import ScalarFncBaseclass
from ..processing.optimumfiltering import OptimumFiltering, OptimumFiltering2D
from ..processing.removebaseline import RemoveBaseline


@numba.guvectorize(
    [(
        # Inputs ---------------------------
        numba.float64[:, :], # event
        numba.int32[:, :],   # where
        numba.int32[:],      # relative_to
        # Outputs --------------------------
        numba.int32[:],      # out_max_inds
        numba.int32[:],      # out_eval_inds
        numba.float64[:],    # out_max_vals
        numba.float64[:],    # out_vals
        numba.int32[:, :],   # where_actual
    )], 
    '(n,m),(n,k),(n)->(n),(n),(n),(n),(n,k)',
    nopython=True,
)
def _max_search(
    # Inputs ----
    event,
    where, 
    relative_to, 
    # Outputs ---
    out_max_inds, 
    out_eval_inds, 
    out_max_vals, 
    out_vals, 
    where_actual,
):
    """
    Perform the evaluation/maximum search on the filtered traces. If applicable, it is done considering relative search/evaluation intervals between channels.

    Note that this is a ``numba`` pre-compiled function which automatically dispatches additional dimensions. Also note that the function signature includes the outputs as well (as this is the mechanism required by ``numpy``/``numba``).

    :param event: The event. Has to have shape (N, M), where N is the number of channels (needs to be 1 if single-channel) and M is the number of samples. Can also have shape (L, N, M) in which case it is treated as a batch of L (multi-channel) events. The outputs will then be of shape (L, ...) as well.
    :type event: np.ndarray
    :param where: Integer array specifying where to search for the maxima (one entry per channel). Has shape (N, 2), where, again, N is the number of channels. The first column then corresponds to the search interval starts, and the second one to the search interval ends. If they are identical, the trace is just evaluated at the specified sample.
    :type where: np.ndarray
    :param relative_to: Integer array specifying relative to which channel the ``where`` argument applies. If -1, it applies globally, e.g. if the corresponding entry in ``where`` says [100, 200], the maximum in the sample interval [100, 200] is searched. If it is anything BUT -1, the interval has to be understood relative to the (previously found) maximum of the respective channel. E.g. if 0 and the maximum of channel 0 is at sample 137, a ``where`` entry of [100, 200] would cause a search in the interval [237, 337].
    :type relative_to: np.ndarray

    :return: Tuple of the arrays' maximum indices (regardless of whether they were evaluated at those indices), the arrays' evaluation indices, the values of the arrays at the maximum indices, the values of the arrays at the evaluation indices, and the index interval that was actually searched (in global index coordinates).
    :rtype: Tuple[np.ndarray]
    """
    # Initialize all output arrays with zeros.
    for j in range(event.shape[-2]):
        if event.ndim > 2:
            for k in range(event.shape[-3]):
                out_max_inds[k, j] = 0
                out_eval_inds[k, j] = 0
                out_max_vals[k, j] = 0.0
                out_vals[k, j] = 0.0
                where_actual[k, j, :] = np.zeros(where.shape[-1])
        else:
            out_max_inds[j] = 0
            out_eval_inds[j] = 0
            out_max_vals[j] = 0.0
            out_vals[j] = 0.0
            where_actual[j, :] = np.zeros(where.shape[-1])
                
    n_ch = event.shape[-2]

    # Array to keep track of which channels were already evaluated.
    done = np.zeros((n_ch, 1), dtype=numba.boolean)

    # Find global triggers first (those which are not relative to
    # any other channel, i.e. have relative_to=-1)
    for i in range(n_ch):
        if relative_to[i] < 0:
            # Evaluation at fixed index.
            if where[i, 0] == where[i, 1]:
                out_eval_inds[i] = where[i, 0]
            # Max search in defined interval.
            else:
                a, b = where[i]
                out_eval_inds[i] = np.argmax(event[i][a:b]) + a
           
            done[i] = True
        
        # Save the actual evaluation interval.
        where_actual[i, 0] = where[i, 0]
        where_actual[i, 1] = where[i, 1]
        
    # The calling function made sure that all other entries in 'relative_to'
    # specify one of the already calculated channels, so we can loop until
    # we disentangled the dependencies.
    # We cannot go through the list once from the beginning because if we 
    # do so, references to channels later in the list are impossible.
    while not np.all(done):
        for i in range(n_ch):
            if done[i]:
                continue

            # If the channel that channel i references has already been marked
            # as ready, we can calculate the pulse height for channel i as well.
            if done[relative_to[i]]:
                # Evaluation at fixed index
                if where[i, 0] == where[i, 1]:
                    out_eval_inds[i] = where[i, 0] + out_eval_inds[relative_to[i]]
                # Max search in defined interval relative to specified channel.
                else:
                    a, b = where[i]
                    a = out_eval_inds[relative_to[i]] + a
                    b = out_eval_inds[relative_to[i]] + b
                    out_eval_inds[i] = np.argmax(event[i][a:b]) + a

                # Save the actual evaluation interval.
                where_actual[i, 0] = where[i, 0] + out_eval_inds[relative_to[i]]
                where_actual[i, 1] = where[i, 1] + out_eval_inds[relative_to[i]]
                # Now channel i is ready as well.
                done[i] = True
    
    # Use the evaluation indices that we constructed above to just evaluate
    # the arrays. Also calculate the array maxima (positions).
    for i in range(n_ch):
        out_max_inds[i] = np.argmax(event[i])
        out_vals[i] = event[i][out_eval_inds[i]]
        out_max_vals[i] = event[i][out_max_inds[i]]

    # Note that the function doesn't return here but through its input arguments.
    # See https://numba.pydata.org/numba-doc/dev/user/vectorize.html#the-guvectorize-decorator


@numba.guvectorize(
    [(
        # Inputs ---------------------------
        numba.float64[:, :], # event
        numba.int32[:],      # eval_pos
        numba.float64[:, :], # sev
        numba.float64[:],    # phs
        numba.int32[:],      # sev_eval_pos
        numba.int32,         # peak_rms_width
        # Outputs --------------------------
        numba.float64[:],    # rms
        numba.float64[:],    # rms_peak
    )], 
    '(n,m),(n),(n,m),(n),(n),()->(n),(n)',
    nopython=True,
) 
def _rms(event, eval_pos, sev, phs, sev_eval_pos, peak_rms_width, rms, rms_peak):
    """
    Calculate the filter (peak) RMS between filtered events and a scaled reference SEV at a given position.

    Note that this is a ``numba`` pre-compiled function which automatically dispatches additional dimensions. Also note that the function signature includes the outputs as well (as this is the mechanism required by ``numpy``/``numba``).

    :param event: The filtered event. Has to have shape (N, M), where N is the number of channels (needs to be 1 if single-channel) and M is the number of samples. Can also have shape (L, N, M) in which case it is treated as a batch of L (multi-channel) events. The outputs will then be of shape (L, ...) as well.
    :type event: np.ndarray
    :param eval_pos: The evaluation positions obtained in '_max_search'. Has to have shape (L, N).
    :type eval_pos: np.ndarray
    :param sev: The filtered SEV to compare to. It is scaled by 'phs' and shifted to align with 'eval_pos'. Has to have shape (N, M)
    :type sev: np.ndarray
    :param phs: The pulse heights obtained in '_max_search', used to scale 'sev'. Has to have shape (L, N, M).
    :type phs: np.ndarray
    :param sev_eval_pos: The evaluation positions of the SEV (positions where the SEV would get evaluated according to the maximum search requirements of '_max_search'). It is used to align it with 'eval_pos'. Has to have shape (N,).
    :type sev_eval_pos: np.ndarray
    :param peak_rms_width: The number of samples (before and after 'eval_pos') that are used for calculating the peak RMS. Has to be a scalar.
    :type peak_rms_width: np.ndarray

    :return: Tuple of RMS and peak RMS values for all events.
    :rtype: Tuple[np.ndarray]
    """
    # Initialize all output arrays.
    for j in range(event.shape[-2]):
        if event.ndim > 2:
            for k in range(event.shape[-3]):
                rms[k, j] = 0.0
                rms_peak[k, j] = 0.0
        else:
            rms[j] = 0.0
            rms_peak[j] = 0.0

    n_ch = event.shape[-2]
    
    for i in range(n_ch):
        shift = eval_pos[i] - sev_eval_pos[i]
        
        lpb = eval_pos[i] - peak_rms_width
        upb = eval_pos[i] + peak_rms_width + 1
        
        # These indices are a nightmare. Do not touch unless you have good
        # reason to. They were validated to be correct (by filtering a SEV
        # and requiring that the RMS and fit RMS are ~0).
        if shift == 0:
            reference = phs[i]*sev[i]
            cropped_event = event[i]
            reference_peak = reference[lpb:upb]
            cropped_event_peak = event[i][lpb:upb]
        elif shift > 0:
            # 'Move' reference to the right.
            reference = phs[i]*sev[i][:-shift]
            # Crop event according to reference's shift.
            cropped_event = event[i][shift:]
            reference_peak = reference[lpb-shift:upb-shift]
            cropped_event_peak = event[i][lpb:upb]
        else:
            # 'Move' reference to the left (note that 'shift' here is negative).
            reference = phs[i]*sev[i][-shift:]
            # Crop event according to reference's shift.
            cropped_event = event[i][:shift]
            reference_peak = reference[lpb:upb]
            cropped_event_peak = event[i][lpb:upb]
        
        # Only calculate RMS if the same number of samples survive (and any at all).
        samples_left_ev = cropped_event.size
        samples_left_ref = reference.size
        if samples_left_ev == samples_left_ref and samples_left_ref > 0:
            rms[i] = np.sqrt(np.mean((reference - cropped_event)**2))
        
        # Only calculate peak RMS if the same number of samples survive (and any at all).
        peak_samples_left_ev = cropped_event_peak.size
        peak_samples_left_ref = reference_peak.size
        if peak_samples_left_ev == peak_samples_left_ref and peak_samples_left_ref > 0:
            rms_peak[i] = np.sqrt(np.mean((reference_peak - cropped_event_peak)**2))

@numba.guvectorize(
    [(
        # Inputs ---------------------------
        numba.float64[:, :, :],     # event
        numba.float64[:, :],        # sev
        numba.float64[:],           # phs
        numba.int16[:, :],          # filter_groups
        numba.float64[:, :, :, :],  # inverted_nps
        numba.int64,                # dt_us
        # Outputs --------------------------
        numba.float64[:],           # chi2
        numba.float64[:],           # chi2_red
    )], 
    '(l,n,m),(n,m),(n),(j,k),(j,m,2,2),()->(n),(n)',
    nopython=True,
)
def _chi2(event, sev, phs, filter_groups, inverted_nps, dt_us, chi2, chi2_red) -> None:
    """
    Calculate the filter (reduced) Chi-squared between events and a scaled reference SEV.

    Note that this is a ``numba`` pre-compiled function which automatically dispatches additional dimensions. Also note that the function signature includes the outputs as well (as this is the mechanism required by ``numpy``/``numba``).

    :param event: The unfiltered event, in frequency space. Has to have shape (N, M), where N is the number of channels (needs to be 1 if single-channel) and M is the number of samples. Can also have shape (L, N, M) in which case it is treated as a batch of L (multi-channel) events. The outputs will then be of shape (L, ...) as well.
    :type event: np.ndarray
    :param sev: The unfiltered SEV, in frequency space, to compare to. It is scaled by 'phs'. Has to have shape (N, M)
    :type sev: np.ndarray
    :param phs: The pulse heights obtained in '_max_search', used to scale 'sev'. Has to have shape (N).
    :type phs: np.ndarray
    :param filter_groups: Specify which filters/channels should be filtered together (producing one output trace). Has to be an array of array of integers (specifying channels that should be filtered with the regular 1D filter)
    :type filter_groups: np.ndarray
    :param inverted_nps: The inverted NPS (or NCM) of the channels. List of elements with length 'len(filter_groups)' and shape (M//2+1) - for 1D-filtered channels - or (M//2+1, 2, 2) - for 2D-filtered channels.
    :type inverted_nps: np.ndarray
    :param dt_us: The time interval (in us) between two consecutive samples. Has to be a scalar.
    :type dt_us: np.ndarray

    :return: Tuple of Chi-squared and reduced Chi-squared values for all events.
    :rtype: Tuple[np.ndarray]
    """
    # Initialize all output arrays.
    for j in range(event.shape[-2]):
        if event.ndim > 2:
            for k in range(event.shape[-3]):
                chi2[k, j] = 0.0
                chi2_red[k, j] = 0.0
        else:
            chi2[j] = 0.0
            chi2_red[j] = 0.0

    res = event.reshape(-1, event.shape[-1]) - phs.reshape(-1, 1) * sev
    delta_f = 1.0 / (event.shape[-1] * dt_us * 1e-6)
    dof = event.shape[-1] + 1

    # Keep track of the number of channels treated
    i = 0

    for k, (fg, inv_N) in enumerate(zip(filter_groups, inverted_nps)):
        # Count channels in the group, after removing the -1 placeholders
        n_ch_fg = np.count_nonzero(fg != -1)

        if n_ch_fg == 1:
            #If the channel is alone in the group, then the NPS is in [:, 0, 0]
            chisq = np.sum(np.real(res[i] * np.conj(res[i]) * inv_N[0, 0])) * delta_f

        else:
            # Grab the right residuals pair, then format it and compute chi-squared
            # res_transp = res[i:i+2].T                                                 # (M//2+1, 2)
            # res_col = res_transp[:, :, None]                                          # (M//2+1, 2, 1)
            # res_row = np.conj(res_transp)[:, None, :]                                 # (M//2+1, 1, 2)
            # chisq = np.sum(np.real(res_row @ inv_N @ res_col).squeeze()) * delta_f    # (M//2+1, 1, 1), then squeezed and summed

            chisq = np.sum(np.real(np.conj(res[i:i+2].T)[:, None, :] @ inv_N @ res[i:i+2].T[:, :, None]).squeeze()) * delta_f

        if chi2.ndim == 1:
            chi2[k] = chisq
            chi2_red[k] = chisq/dof
        else:
            for j in range(n_ch_fg):
                chi2[k, j] = chisq
                chi2_red[k, j] = chisq/dof

        i += n_ch_fg

def _mixed_of_helper(event, of_list, filter_groups):
    # Helper function that resolves a mix of 1D and 2D OFs.
    # Feel free to come up with something more elegant. This is terrible.
    all_evaluations = []
    for of, fg in zip(of_list, filter_groups):
        if isinstance(fg, int):
            all_evaluations.append(of(event[..., [fg,], :]))
        else:
            all_evaluations.append(of(event[..., list(fg), :]))

    n_channels_effective = len(of_list)
    n_out_samples = all_evaluations[-1].shape[-1]

    if event.ndim == 3:
        if n_channels_effective > 1:
            out = np.zeros((event.shape[0], n_channels_effective, n_out_samples))
            for i, res in enumerate(all_evaluations):
                out[:, i, :] = res.squeeze()
        else:
            out = np.zeros((event.shape[0], 1, n_out_samples))
            out[:, 0, :] = all_evaluations[0].squeeze()
    # Has to have ndim=2. In that case, it cannot be batched anymore.
    elif n_channels_effective > 1:
        out = np.zeros((n_channels_effective, n_out_samples))
        for i, res in enumerate(all_evaluations):
            out[i, :] = res.squeeze()
    # Has to have ndim=2, no batches, and only one output trace.
    else:
        out = np.zeros((1, n_out_samples))
        out[0, :] = all_evaluations[0].squeeze()

    return out

def _pad_filter_groups(arr):
    # convert ints to length-1 tuples
    rows = [(x,) if isinstance(x, int) else tuple(x) for x in arr]
    n = len(arr)
    m = max(len(r) for r in rows)
    # Pad arrays with -1 placeholders to get the right dimension
    padded_arr = np.full((n, m), -1, dtype=int)
    for i, r in enumerate(rows):
        padded_arr[i, :len(r)] = r
    return padded_arr

class OFPulseHeight(ScalarFncBaseclass):
    """
    Calculate optimum filter pulse heights.

    For single channels, this merely filters event traces and performs a maximum search in a specified range of the record window (or evaluates the trace at a specific index). For multiple channels, however, the maximum search (or index evaluation) can be performed relative to other channels. 

    2D optimum filtering is also supported.
     
    :param of: The optimum filter(s) to use. One for each channel, i.e. has shape ``(N, M//2+1)`` where ``N`` is the number of channels and ``M`` is the record length. Using 2D filters is no exception here: The filters have to be supplied as shape ``(N, M//2+1)`` arrays.
    :type of: np.ndarray
    :param sev: The standard events corresponding to the filters. They are used as reference to calculate the filter (peak) RMS. One standard event per entry in ``of`` is needed. I.e. has to be of shape ``(N, M)``.
    :type sev: np.ndarray
    :param filter_groups: Only relevant if you intend to use 2D filtering. Here, you may specify which filters/channels should be filtered together (producing one output trace). Has to be a list of integers (specifying channels that should be filtered with the regular 1D filter) and tuples of integers (specifying channels that should be filtered together using the 2D filter). E.g. the argument ``[(0, 1), 2]`` would apply the 2D filter to channels 0 and 1 together, and the 1D filter to channel 2. You may only specify each channel exactly once. Defaults to None, i.e. all channels are treated independently and no 2D optimum filtering is applied.
    :type filter_groups: List[Union[int, Tuple[int]]], optional
    :param nps: The Noise Power Spectra (or Noise Covariance Matrices) of the channels, for the filter Chi-squared calculation. List of elements either with shape ``(N, M//2+1)`` (for channels to be filtered with 1D filter)  or ``(N, 2, 2, M//2+1)`` (for pairs of channels to be filtered with 2D filter). The channels must be provided in the same order as they appear in the filter_groups parameter! Defaults to None, i.e. no Chi-squared is computed.
    :type nps: List[np.ndarray], optional
    :param max_search: A list (one entry for each channel, or channel group in case you use a 2D filter) that specifies where to evaluate the filtered trace or where to look for maxima (potentially relative to another channel). If the entry is an integer ``k``, the filtered trace is evaluated at that integer ``k``. If the entry is a tuple of integers ``(a, b)``, the maximum is searched in the interval ``(a, b)``. If the entry is a tuple of floats ``(x, y)``, the maximum is searched in the interval ``(record_length*x, record_length*y)``, i.e. a float tuple specifies the search range relative to the record window. If the entry in ``relative_to`` of the respective channel is ``None``, the evaluation at ``k``, or search in ``(a, b)``/``(x, y)`` are absolute, i.e. relative to the full record window. If the entry in ``relative_to`` specifies another channel, the evaluation is done relative to the evaluation position of that channel. E.g. if channel 0 is evaluated at sample ``r`` (either because the user specified evaluation at ``r`` or because the maximum was found at ``r``) and channel 1's ``relative_to`` is channel 0 and its ``max_search`` is -10, then the filtered trace of channel 1 is evaluated 10 samples before the evaluation position of channel 0. Defaults to None, i.e. all evaluations are global. Defaults to ``(0.2, 0.4)``, i.e. searching between 20% and 40% of the record window for all channels.
    :type max_search: List[Union[int, float, Tuple[Union[int, float]]]], optional
    :param relative_to: Together with ``max_search``, you can specify relations between the filter evaluations of different channels. Has to be a list (one entry for each channel, or channel group in case you use a 2D filter). For ``None`` entries, the respective channel will have its maxima searched (or evaluated) **globally** (relative to the full record window) according to the respective entry in ``max_search``. If it is an integer ``c``, the evaluation is performed **relative to the evaluation index** of channel ``c``. This evaluation index can either result from a fixed evaluation position (specified in ``max_search``) or an actual maximum search. Defaults to ``None``, i.e. independent evaluation for all channels.
    :type relative_to: List[int], optional
    :param peak_rms_width: The number of samples before and after the filter evaluation sample that are used for calculating the filter RMS value. Defaults to 5 samples
    :type peak_rms_width: int, optional
    :param kwargs: Additional keyword arguments for :class:`cait.versatile.OptimumFiltering` (or :class:`cait.versatile.OptimumFiltering2D`). Most notably, you can set ``method='linear'`` for a linear convolution of event traces larger than the one dictated by ``sev``.
    :type kwargs: Any

    :return: Tuple of calculated values. See ``vai.OFPulseHeight.names()`` and ``vai.OFPulseHeight.dtypes`` for what they are and which data types they have. Their description is given below.
    :rtype: Tuple[Union[float, int]]

    Output parameters are:

    - **of_ph** (V): The 'optimum filter pulse height'. Either evaluated at a fixed sample or through the maximum search in a specified interval (see arguments ``max_search`` and ``relative_to``). Note that this is not necessarily the global maximum of the filtered event.
    - **of_max_val** (V): The global maximum of the filtered event.
    - **of_eval_pos** (samples): The position (index) where the filtered event was evaluated (the value of the filtered event at this index is **of_ph**).
    - **of_max_pos** (samples): The position (index) of the global maximum of the filtered event (the value of the filtered event at this index is **of_max_val**).
    - **of_rms** (V): The root mean square (RMS) value between the filtered event and the filtered reference SEV. The reference is obtained by scaling the SEV to **of_ph** and shifting it relative to the event such that their **of_eval_pos** indices are aligned. This RMS value is global (all samples which were not invalidated because of the shifting).
    - **of_peak_rms** (V): Same as **of_rms** but only the number of samples specified by argument ``peak_rms_width`` before and after the **of_eval_pos** is used for the calculation.
    - **of_chi2**: Filter Chi-squared of the events
    - **of_chi2_red**: Reduced Chi-squared **of_chi2**, but divided by the number of degrees of freedom.

    .. warning::
        If you directly use this function, make sure to retrieve its results by first applying it to an iterator ``of_res = vai.apply(f, it)`` and unpacking it into a dictionary ``of_res_dict = {k: v for k, v in zip(f.names(), of_res)}`` (as explained in the example below). By doing so, you future-proof your code as more outputs might be added to the function in the future.

    **Example:**

    .. code-block:: python

        import cait.versatile as vai

        record_length = 2**14

        # Generate mock data (two channels)
        md = vai.MockData(record_length=record_length)
        it = md.get_event_iterator()
        sev, of = md.sev, md.of

        # Filtering single channel
        f1 = vai.OFPulseHeight(
            of=of[0],
            sev=sev[0],
        )

        # See what the function is doing
        vai.Preview(it[0], f1)

        # Apply function to all events (this returns a tuple of arrays).
        of_res = vai.apply(f1, it[0])

        # Generate a directory where the keys correspond to the names
        # of the function outputs and the values to the calculated values.
        # Always (!) access them like this as the number of output values
        # might increase in the future. By accessing them like this, you make
        # sure that your code doesn't break.
        of_res_dict = {k: v for k, v in zip(f1.names(), of_res)}

    .. image:: media/OFPulseHeight_preview1.png

    .. code-block:: python

        # Filtering two channels and perform a maximum search for
        # the second one in an interval [-300, -100] samples relative
        # to the maximum found for the first channel.
        f2 = vai.OFPulseHeight(
            of=of,
            sev=sev,
            relative_to=[None, 0], 
            max_search=[(0.2, 0.4), (-300, -100)],
        )

        vai.Preview(it, f2)

    .. image:: media/OFPulseHeight_preview2.png

    .. code-block:: python

        # Generate mock stream data with three channels
        s = vai.MockStream(
            rate_Hz=1, 
            seed=137,
            pulse_shape=[[0.5, 0.5, 0.3, 0.1, 10.0], [0.3, 0.5, 0.3, 0.01, 4.0], [0.3, 0.5, 0.3, 0.01, 4.0]],
            tp_shape=[[0.5, 0.5, 0.3, 0.01, 20.0], [0.5, 0.5, 0.3, 0.01, 10.0], [0.5, 0.5, 0.3, 0.01, 10.0]],
            baseline_sig=[0.005, 0.001, 0.002],
        )
        it = s.get_event_iterator(
            ["Ch0", "Ch1", "Ch2"], 
            record_length=record_length, 
            inds=[(4+x)*record_length for x in range(10)],
        )

        # Filter two of the three channels with larger stream window
        # (potentially reduces filter edge effects).
        f3 = vai.OFPulseHeight(
            of=of,
            sev=sev,
            method="linear",
        )

        vai.Preview(it[:2].with_extended_window(), f3)

    .. image:: media/OFPulseHeight_preview3.png

    .. code-block:: python

        # Filter first two channels with 2D filter and read third one
        # relative to the evaluation position of the 2D filtered trace.
        # Note that here, we just use the filter/sev of the second channel 
        # twice. This is just for this demonstration (you would have 
        # three different filters/templates of course).
        f4 = vai.OFPulseHeight(
            of=[of[0], of[1], of[1]],
            sev=[sev[0], sev[1], sev[1]],
            filter_groups=[(0, 1), 2],    # 0 and 1 together, 2 separately
            relative_to=[None, 0],        # (0, 1) absolute, 2 relative to (0, 1)
            max_search=[(0., 1.), -1200], # maximum search in entire window for (0, 1),
                                          # fixed evaluation of 2 (1200 samples before
                                          # maximum of (0, 1)).
        )

        vai.Preview(it, f4)

    .. image:: media/OFPulseHeight_preview4.png
    """
    _outputs = (
        ("of_ph", float),
        ("of_max_val", float),
        ("of_eval_pos", int),
        ("of_max_pos", int),
        ("of_rms", float),
        ("of_peak_rms", float),
        ("of_chi2", float),
        ("of_chi2_red", float),
    )

    def __init__(
        self, 
        of: np.ndarray, 
        sev: np.ndarray,
        *,
        filter_groups: List[Union[int, Tuple[int]]] = None,
        max_search: List[Union[int, float, Tuple[Union[int, float]]]] = (0.2, 0.4),
        relative_to: List[int] = None, 
        peak_rms_width: int = 5,
        nps: Union[List[np.ndarray], np.ndarray] = None,
        dt_us: int = None,
        **kwargs,
    ):
        # Sanitize input arrays
        of = np.atleast_2d(of)
        sev = np.atleast_2d(sev)
        self._record_length = sev.shape[-1]
        self._n_channels = sev.shape[0]
        
        # Check if shapes of OF and SEV are compatible.
        if not of.shape[0] == sev.shape[0]:
            raise ValueError(f"OF and SEV have to have the same number of channels. Got {of.shape[0]} and {sev.shape[0]}.")
        if of.shape[-1] != sev.shape[-1]//2 + 1:
            raise ValueError(f"Input argument 'of' has to have shape (N, M//2+1) for input argument 'sev' of shape (N, M).")
        
        # If filter_groups is None, all channels are treated normally (no 2D OF treatment)
        if filter_groups is None:
            filter_groups = list(range(self._n_channels))

        if not all(isinstance(x, (int, tuple)) for x in filter_groups):
            raise ValueError(f"Input argument 'filter_groups' has to be a list of ints or tuple of ints. Got {[type(x) for x in filter_groups]}")
        for x in filter_groups:
            if isinstance(x, tuple):
                if not all(isinstance(i, int) for i in x):
                    raise ValueError(f"Input argument 'filter_groups' has to be a list of ints or tuple of ints. Got {[type(i) for i in x]}")
        # Flatten list of ints and tuples of ints and check if all ints are valid channel indices.
        fgroups_flattend = [x for y in filter_groups for x in (y if isinstance(y, tuple) else (y,))]
        if len(fgroups_flattend) > self._n_channels:
            raise ValueError(f"You specified more channels in 'filter_groups' ({len(fgroups_flattend)}) than there are available channels in 'of' and 'sev' ({self._n_channels}).")
        if len(fgroups_flattend) < self._n_channels:
            raise ValueError(f"You specified less channels in 'filter_groups' ({len(fgroups_flattend)}) than there are available channels in 'of' and 'sev' ({self._n_channels}).")
        for i in fgroups_flattend:
            if i < 0 or i >= self._n_channels:
                raise ValueError(f"Values in 'filter_groups' have to specify valid channel indices. Got index {i} but number of channels is {self._n_channels}.")
                
        self._n_filters = len(filter_groups)
        
        # If max_search and relative_to aren't given as lists, the input is used for all channels.
        if not isinstance(max_search, list):
            max_search = [max_search]*self._n_filters
        if not isinstance(relative_to, list):
            relative_to = [relative_to]*self._n_filters
        
        # Check if exactly one element in max_search and relative_to is given for each channel
        if not len(max_search) == self._n_filters:
            raise ValueError(f"If 'max_search' is a list, it has to have length==n_channels (counting 2d-OF groups as single channel). Got {len(max_search)} and {self._n_filters}.")
        if not len(relative_to) == self._n_filters:
            raise ValueError(f"If 'relative_to' is a list, it has to have length==n_channels (counting 2d-OF groups as single channel). Got {len(relative_to)} and {self._n_filters}.")
        
        # Make sure that at least one entry in relative_to is None,
        # meaning that at least one of the evaluations is absolute.
        # (If all evaluations are relative to each other, we cannot
        # resolve it).
        if relative_to.count(None) == 0:
            raise ValueError(f"At least one entry in 'relative_to' has to be None. Got {relative_to}.")
        
        # Check if relative channels can be resolved (i.e. second 
        # channel can only be relative to first, third can only be
        # relative to first or second, etc.
        for i, rt in enumerate(relative_to):
            if rt is None:
                continue 

            # Self-reference
            if rt == i:
                raise ValueError(f"Self-reference for channel {rt}.")
                
            if rt >= self._n_filters:
                raise ValueError(f"Relative channel reference '{rt}' is invalid. Only {self._n_filters} channels (or 2d-OF groups of channels) are available.")
                
            if not isinstance(rt, int):
                raise ValueError(f"All entries in 'relative_to' have to be None or integers. Got {type(rt)}.")
            
            s = rt
            for j in range(self._n_filters):
                s = relative_to[s]
                if s is None:
                    break
                    
            if j == len(relative_to):
                raise ValueError(f"Relative channels cannot be resolved for relative_to={relative_to}.")
                
        # Sanitize relative_to for use with numba.
        relative_to_sanitized = np.zeros(len(relative_to), dtype=np.int32)
        for i, rt in enumerate(relative_to):
            if rt is None:
                relative_to_sanitized[i] = -1
            else:
                relative_to_sanitized[i] = rt
        
        # Make sure that the search intervals are all given as (tuples of) integers.
        # Also sanitize for use with numba.
        where = np.zeros((len(max_search), 2), dtype=np.int32)
        for i, ms in enumerate(max_search):
            if isinstance(ms, int):
                where[i, :] = np.array([ms, ms], dtype=np.int32)
            elif isinstance(ms, float):
                where[i, :] = np.array([int(ms*self._record_length), int(ms*self._record_length)], dtype=np.int32)
            elif isinstance(ms, tuple):
                if len(ms) != 2:
                    raise ValueError(f"For tuple entries in 'max_search', the tuple has to have length 2. Got {len(ms)}.")
                temp_list = []
                for val in ms:
                    if isinstance(val, int):
                        temp_list.append(val)
                    elif isinstance(val, float):
                        temp_list.append(int(val*self._record_length))
                    else:
                        raise ValueError(f"For tuple entries in 'max_search', the tuple entries must be integer or floats. Got {type(val)}.")
                
                where[i, :] = np.array(temp_list, dtype=np.int32)
            else:
                raise ValueError(f"All entries in 'max_search' have to be integers, floats, or tuples. Got {type(ms)}.")
        
        # The SEV has fixed length, i.e. we cannot use method="linear" for filtering it.
        # We therefore define a dummy function which does nothing UNLESS the method is 'linear',
        # in which case we pad the SEV before it's filtered.
        if "method" in kwargs and kwargs["method"].lower() == "linear":
            def outer(f):
                def inner(x):
                    k = np.ndim(x) - 1
                    return f(
                        np.pad(x, (*([(0, 0)]*k), (self._record_length, self._record_length)))
                    )
                return inner
        else:
            def outer(f):
                def inner(x): 
                    return f(x)
                return inner

        handle_method = outer

        # Regular case (1D-OF only)
        if self._n_channels == self._n_filters:
            self._filter = OptimumFiltering(of, **kwargs)
            sev_filter = handle_method(OptimumFiltering(of, **kwargs))
        # 2D-OF case
        else:
            of_list = [
                (
                    OptimumFiltering(of[fg], **kwargs) 
                    if isinstance(fg, int) 
                    else OptimumFiltering2D(of[..., list(fg), :], **kwargs)
                )
                for fg in filter_groups
            ]
            of_list_sev = [
                (
                    handle_method(OptimumFiltering(of[fg], **kwargs) )
                    if isinstance(fg, int) 
                    else handle_method(OptimumFiltering2D(of[..., list(fg), :], **kwargs))
                )
                for fg in filter_groups
            ]
            
            self._filter = lambda ev: _mixed_of_helper(ev, of_list, filter_groups)
            sev_filter = lambda ev: _mixed_of_helper(ev, of_list_sev, filter_groups)
                
        self._search = lambda event: _max_search(event, where, relative_to_sanitized)
        
        sev_filtered = np.atleast_2d(sev_filter(np.atleast_2d(sev)))
        _, sev_eval_pos, *_ = self._search(sev_filtered)
        
        self._rms = lambda event, phs, eval_pos: _rms(event, eval_pos, sev_filtered, phs, sev_eval_pos, peak_rms_width)

        # We already checked above if filter_group is None
        if nps is not None:
            nps = np.atleast_2d(nps)
            dt_us = dt_us or next((getattr(n, 'dt_us') for n in nps if hasattr(n, 'dt_us')), None)
            if dt_us is None:
                raise ValueError("If you provide a 'nps' for Chi-squared calculation, you either have to provide a vai.NPS object or set 'dt_us'.")            
            if len(nps) != len(filter_groups):
                raise ValueError(f"If you provide a 'nps', the number of elements in 'filter_groups' and 'nps' has to be the same. Got {len(filter_groups)} and {len(nps)}")
            for n, group in zip(nps, filter_groups):
                # If two channels are grouped together, check that a NCM with the right dimension is provided
                if isinstance(group, tuple):
                    if (n.ndim != 3) or (n.shape[-2] != 2) or (n.shape[-3] != 2):
                        raise ValueError("For each tuple in 'filter_groups', the corresponding 'nps' must be a Noise Covariance Matrix with dimension (2, 2, M//2+1).")
                elif isinstance(group, int):
                    if (n.ndim != 1):
                        raise ValueError("For each int in 'filter_groups', the corresponding 'nps' must be a Noise Power Spectrum with dimension (M//2+1).")

                if n.shape[-1] != (sev.shape[-1]//2 + 1):
                    raise ValueError(f"The elements in 'nps' has to have the last dimension with shape (M//2+1) for input argument 'sev' of shape (N, M). Got ({n.shape[-1]}) and ({sev.shape[-2]}, {sev.shape[-1]//2 + 1}).")

                if not isinstance(nps, np.ndarray):
                    raise TypeError("'nps' must be a list of np.ndarray!")

            # Un-apply normalization (see vai.NPS calculation)
            n_samples = 2 * (sev.shape[-1] - 1)
            W = n_samples * n_samples
            delta_f = 1.0 / (n_samples * dt_us * 1e-6)

            inverted_nps = []

            # Compute inverted NPS (or NCM)
            for n in nps:
                n_norm = n * W * delta_f / 2
                if n.ndim == 1:
                    # 1D inversion
                    n_inv = 1. / n_norm
                    # Convert to dimensions consistent with NCM (M//2+1, 0, 0). The NPS will reside in [:, 0, 0].
                    if len(filter_groups) < self._n_channels:          
                        out = np.full((n_inv.shape[0], 2, 2), np.nan)
                        out[:, 0, 0] = n_inv
                        n_inv = out
                else:
                    # 2D inversion
                    # Transpose to apply correct inversion
                    n_inv = np.linalg.pinv(n_norm.transpose(2, 0, 1))

                inverted_nps.append(n_inv)

            sev_fft = np.fft.rfft(sev, axis=-1)
            padded_fg = _pad_filter_groups(filter_groups)
                
            self._chi2 = lambda event, phs: _chi2(event, sev_fft, phs, padded_fg, inverted_nps, dt_us)

        else:
            warnings.warn("WARNING: You did not provide any 'nps' and/or 'dt_us' as an input, hence the 'chi2' and 'chi2_red' output dataset will be filled with zeros.")
            # If no NPS is provided, define a Chi2 dummy function that returns arrays of the correct shape, filled with zeros
            def _chi2_dummy(event, phs):
                chi2 = np.zeros(self._n_channels, dtype=float)
                chi2_red = np.zeros(self._n_channels, dtype=float)
                return chi2, chi2_red
            
            self._chi2 = lambda event, phs: _chi2_dummy(event, phs)
    
    def __call__(self, event):
        if self._n_channels == 1:
            if event.ndim > 1:
                # (n_events, record_length) -> (n_events, 1, record_length)
                event = np.transpose(np.atleast_3d(event), [0, 2, 1])
            else:
                # (record_length, ) -> (1, record_length)
                event = np.atleast_2d(event)
        
        self._filtered_event = np.atleast_2d(self._filter(event))
        max_pos, eval_pos, max_ph, eval_ph, self._actual_where = self._search(self._filtered_event)
        rms, peak_rms = self._rms(self._filtered_event, eval_ph, eval_pos)
        chi2, chi2_red = self._chi2(np.atleast_2d(np.fft.rfft(event, axis=-1)), eval_ph)
        
        return (
            eval_ph, 
            max_ph, 
            eval_pos, 
            max_pos, 
            rms, 
            peak_rms,
            chi2, 
            chi2_red,
        )

    @property
    def batch_support(self):
        return "full"
    
    def preview(self, event):
        out_dict = {k: v for k, v in zip(self.names(), self(event))}
        ev_filtered = self._filtered_event.squeeze()
        event_x = np.arange(event.shape[-1])
        filtered_x = np.arange(event.shape[-1])

        # If method="linear" was used, the filtered trace and 
        # initial trace have different lengths.
        if ev_filtered.shape[-1] != event.shape[-1]:
            event_x = event_x - self._record_length
            filtered_x = event_x[self._record_length: -self._record_length]

        aw = self._actual_where.squeeze()
        ev = RemoveBaseline()(event)
        lb = min(np.min(ev), np.min(ev_filtered))
        ub = max(np.max(ev), np.max(ev_filtered))
        
        if np.ndim(event) > 1:
            d, s = dict(), dict()
            for i in range(self._n_channels):
                d[f"Ch{i}"] = [event_x, ev[i]]
            if self._n_filters == 1:
                d[f"filtered event"] = [filtered_x, ev_filtered]
                s[f"eval_pos"] = [[out_dict["of_eval_pos"]], [out_dict["of_ph"]]]
                s[f"max_pos"] = [[out_dict["of_max_pos"]], [out_dict["of_max_val"]]]
                d[f"Search"] = [[aw[0], aw[0], aw[1], aw[1], aw[0]], [lb, ub, ub, lb, lb]]
            else:
                for i in range(self._n_filters):
                    d[f"filtered Ch{i}"] = [filtered_x, ev_filtered[i]]
                    s[f"eval_pos Ch{i}"] = [[out_dict["of_eval_pos"][i]], [out_dict["of_ph"][i]]]
                    s[f"max_pos Ch{i}"] = [[out_dict["of_max_pos"][i]], [out_dict["of_max_val"][i]]]
                    d[f"Search Ch{i}"] = [[aw[i][0], aw[i][0], aw[i][1], aw[i][1], aw[i][0]], [lb, ub, ub, lb, lb]]
        else:
            d = {
                "event": [event_x, ev], 
                "filtered event": [filtered_x, ev_filtered],
                "Search": [[aw[0], aw[0], aw[1], aw[1], aw[0]], [lb, ub, ub, lb, lb]]
            }
            s = {
                "eval_pos": [np.atleast_1d(out_dict["of_eval_pos"]), np.atleast_1d(out_dict["of_ph"])],
                "max_pos": [np.atleast_1d(out_dict["of_max_pos"]), np.atleast_1d(out_dict["of_max_val"])],
            }
        
        fmt_arr = lambda l: ", ".join([f"{x:.2g}" for x in np.atleast_1d(l)])
        label = f"ph=[{fmt_arr(out_dict['of_ph'])}] V, rms=[{fmt_arr(out_dict['of_rms'])}] V, peak_rms=[{fmt_arr(out_dict['of_peak_rms'])}] V, chi2=[{fmt_arr(out_dict['of_chi2'])}]"

        return dict(line=d, scatter=s, axes=dict(xaxis=dict(label=label)))