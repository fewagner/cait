from typing import Union, List

import numpy as np

from ..functionbase import FncBaseClass
from ..scalarfunctions.mainparameters import MainParameters

class FluxQuantumLossCorrection(FncBaseClass):
    """
    Correct event for flux quantum loss (FQL). Works only for one channel at a time.

    :param method: One of three methods: "mmd", "slope" or "true_ph". "mmd" (mininmum-minimum difference) calculates the FQL as difference between the minima before and after the pulse (with some fluctuation mitigation). "slope" calculates the FQL as the slope of the event. "true_ph" assumes that the true pulse height (without FQL) is known - and provided in the parameter true_pulseheight - and calculates the FQL as the difference between true and apparent pulse height. Defaults to the recommended "mmd".
    :type method: str
    :param fql_voltage: If the voltage drop of an FQL is known and provided here, the correction shift will be an integer multiple of this value, instead of any value. In this case, the number of FQ lost is determined by subtracting the threshold (param "thresh") from the calculated shift and then rounding up to an integer multiple of fql_voltage. Defaults to None.
    :type fql_voltage: float, optional
    :param thresh: Minimum shift value to accept as FQL and correct for. Smaller or negative values are assumed to not stem from FQLs. Defaults to 0.2 V.
    :type thresh: float
    :param true_pulseheight: The known true pulse height as defined by the maximum possible pulse height for fully saturated pulses. Necessary for method "true_ph". Defaults to None.
    :type true_pulseheight: float, optional
    :param return_shift_value: If True, not the shifted event, but instead the shift value is returned. Defaults to False.
    :type return_shift_value: bool
    :param location: Name(s) of the parameter(s) from :class:`cait.versatile.MainParameters` to use to determine where the correction should be applied. May be a string (in which case this method is applied to all channels), or a list with the same number of entries as the number of channels being processed. Must be one of `"onset_CAT"`, `"onset"`, `"min_deriv_index"`, `"max_deriv_index"`, or `"peak_loc"`. Defaults to `"onset_CAT"`.
    :param location: Union[str, List[str]], optional
    :param reset_thresh: Threshold for the `"min_deriv"` value from :class:`cait.versatile.MainParameters`, above which it is assumed that a SQUID reset has occurred, and a correction is applied.
    :type reset_thresh: float
    :param reset_mask: A number of samples on either side of the `"min_deriv_index"` from from :class:`cait.versatile.MainParameters` used to calculcate the correction.  In addition, **twice** this number of samples will be masked during the correction, which can be used to mask out artifacts caused by the reset.
    :type reset_mask: int

    :return: Event with FQL corrected, or value of shift if return_shift_value is set to True.
    :rtype: Union[numpy.ndarray, float]

    **Example:**

    .. code-block:: python

        import numpy as np
        import cait.versatile as vai

        # Get events and SEV from mock data (and select first channel)
        md = vai.MockData()
        sev = md.sev[0]
        events = md.get_event_iterator()[0].with_processing(vai.RemoveBaseline())

        # Define a function that (roughly) mimics events with FQL.
        # (This is of course only needed to make this example
        # self-contained. You would have actual data for such events.)
        SHIFT = 3
        def mock_fql(event):
            fake_event = event.copy()
            ph = np.max(fake_event)
            flag = fake_event > ph - SHIFT
            fake_event[flag] = ph - SHIFT
            k = np.argmax(flag[::-1])
            fake_event[-k:] += ph*sev[-k:]/np.max(sev[-k:]) - vai.BoxCarSmoothing()(fake_event)[-k:] - SHIFT

            return fake_event

        # Add this function as processing (to fake FQL events).
        # Again, you don't need this because you have FQL data.
        events = events.with_processing(mock_fql)

        # Preview the working of the FQL correction
        vai.Preview(events, vai.FluxQuantumLossCorrection())

        # Or calculate the shift value by setting 'return_shift_value=True'.
        # The result are values close to 3, i.e. the one we 'simulated'
        shift_values = vai.apply(vai.FluxQuantumLossCorrection(return_shift_value=True), events)

        # Check the distribution of shift values
        vai.Histogram(shift_values)

    .. image:: media/FQLC_preview.png
    """
    _locations = [
            "peak_position",
            "onset",
            "min_deriv_index",
            "max_deriv_index",
            "onset_CAT",
            ]
    def __init__(self,
                 method: str = "mmd",
                 fql_voltage: float = None,
                 thresh: float = 0.2,
                 true_pulseheight: float = None,
                 return_shift_value: bool = False,
                 location: Union[str, List[str]] = "onset_CAT",
                 reset_thresh: float = 4,
                 reset_mask: int = 20,
                 ):
        self._mp = MainParameters()#bcs=dict(length=1))
        self._method = method
        self._thresh = thresh
        self._true_pulseheight = true_pulseheight
        self._fql_voltage = fql_voltage
        self._return_shift_value = return_shift_value
        self._loc = location
        self._rthresh = reset_thresh
        self._rmask = int(reset_mask)

    def __call__(self, event):
        event = np.array(event)

        # We will work in batches
        orig_shape = None
        was_1d = False
        if event.ndim == 1:
            was_1d = True
            orig_shape = event.shape
            event = event[None, None, :]
        elif event.ndim == 2:
            # Could either be batched or multiple channels, doesn't really matter
            orig_shape = event.shape
            event = event[None, ...]


        if isinstance(self._thresh, (int, float)):
            self._thresh = np.full(event.shape[1], self._thresh)
        if isinstance(self._true_pulseheight, (int, float)):
            self._thresh = np.full(event.shape[1], self._true_pulseheight)
        if isinstance(self._fql_voltage, (int, float)):
            self._thresh = np.full(event.shape[1], self._fql_voltage)

        if isinstance(self._loc, str):
            # Make sure there is one for each channel
            _loc = [self._loc] * event.shape[1]
        elif isinstance(self._loc, list) and len(self._loc) == 1 and event.shape[1] > 1:
            # Treat a length-1 list the same as a str
            _loc = [self._loc[0]] * event.shape[1]
        else:
            assert len(self._loc) == event.shape[1], f"Incorrect shape to parameter 'location'; must either be a single string, or a list with the same length as the number of channels, got {len(self._loc)} and {event.shape[1]}.\nNOTE: if you have constructed an iterator with multiple channels and an explicit location for each, you will not be able to index specific channels; in this case, loop over the channels instead."
            _loc = self._loc

        assert np.all([x in self._locations for x in _loc]), f"Incorrect value to parameter 'location'; got {_loc}, all values must be in {self._locations}"

        self._event_nobl = np.array(event)
        self._corrected_event = np.array(self._event_nobl)

        self._ph = np.zeros(event.shape[:-1])
        self._t0 = np.zeros(event.shape[:-1])
        self._t_max = np.zeros(event.shape[:-1])
        self._t_end = np.zeros(event.shape[:-1])
        self._lin_drift = np.zeros(event.shape[:-1])
        self._t_min_1 = np.zeros(event.shape[:-1], dtype=int)
        self._t_min_2 = np.zeros(event.shape[:-1], dtype=int)
        self._baseline_mean_1 = np.zeros(event.shape[:-1])
        self._baseline_mean_2 = np.zeros(event.shape[:-1])
        self._flux_loss = np.zeros(event.shape[:-1])
        self._slope = np.zeros(event.shape[:-1])

        mp = np.array(self._mp(event))
        mpd = {k:v for k, v in zip(self._mp.names(), mp)}
        locs = [self._mp.names().index(x) for x in _loc]

        for ic in range(event.shape[1]):
            for ib in range(event.shape[0]):
                # Determine if there was a SQUID reset (a jump from the lowest baseline
                # to the highest)
                if mpd["max_deriv"][ib, ic] > self._rthresh:
                    # SQUID reset occurred; fix the reset before applying the FQL correction
                    idx0 = int(mpd["max_deriv_index"] - self._rmask)
                    idx1 = int(mpd["max_deriv_index"] + self._rmask)
                    event[ib, ic, idx0:idx1] = event[ib, ic, idx0]
                    event[ib, ic, idx1:] -= event[ib, ic, idx1] - event[ib, ic, idx0]

                    self._event_nobl[ib, ic] = event[ib, ic]
                    self._corrected_event[ib, ic] = self._event_nobl[ib, ic]
                    mp[:, ib, ic] = self._mp(event[ib, ic])
                    mpd = {k:v for k, v in zip(self._mp.names(), mp)}


                # Calculate main parameters
                self._t0[ib, ic] = mp[locs[ic], ib, ic]
                self._ph[ib, ic], self._t_max[ib, ic], self._lin_drift[ib, ic] = \
                        mp[np.array([0, 1, 6]), ib, ic]
                # The old t_end was relative to the record window, this is lost in the
                # new MainParameters, but the position relative to the peak position should
                # be close enough
                self._t_end[ib, ic] = mp[4, ib, ic] + mp[1, ib, ic]
                # t0 also needs to be relative to the record window
                #self._t0[ic] += event.shape[-1] // 4

                # Find minima of voltage trace before and after pulse, take difference
                if self._method == "mmd":
                    # Model was developed for a fixed record length. Here, the indices
                    # 600 and 100 were found to work well. Consequently, we scale the
                    # indices now for an arbitrary record length.
                    k1, k2 = int(600/2**14*event.shape[-1]), int(100/2**14*event.shape[-1])

                    if int(self._t_max[ib, ic]) > k1:
                        self._t_min_1[ib, ic] = k1 + np.argmin(self._event_nobl[ib, ic, k1:int(self._t_max[ib, ic])]) # start at 600 so we don't end up out of bounds when averaging later
                    else:
                        self._t_min_1[ib, ic] = k1
                    if int(self._t_max[ib, ic]) < event.shape[-1]:
                        self._t_min_2[ib, ic] = max(k1,int(self._t_max[ib, ic])) + np.argmin(self._event_nobl[ib, ic, max(k1,int(self._t_max[ib, ic])):])
                    else:
                        self._t_min_2[ib, ic] = event.shape[-1]
                    # here, the 600 catches the case of a faulty or non-event with tmax before sample #600
                    # take the values for the average not around the minima as the minima are subject to fluctuations.
                    # also don't take values after the minima as the (/another) pulse (pileup) might come into play there.
                    self._baseline_mean_1[ib, ic] = np.mean(self._event_nobl[ib, ic, self._t_min_1[ib, ic]-k1:self._t_min_1[ib, ic]-k2])
                    self._baseline_mean_2[ib, ic] = np.mean(self._event_nobl[ib, ic, self._t_min_2[ib, ic]-k1:self._t_min_2[ib, ic]-k2])
                    self._flux_loss[ib, ic] = self._baseline_mean_1[ib, ic]-self._baseline_mean_2[ib, ic]

                elif self._method == "slope":
                    self._slope[ib, ic] = self._lin_drift[ib, ic] * event.shape[-1]
                    self._flux_loss[ib, ic] = -self._slope[ib, ic]

                elif self._method == "true_ph":
                    if self._true_pulseheight is None:
                        raise ValueError('True pulse height needs to be provided for the fqlc method "true_ph".')

                    self._flux_loss[ib, ic] = self._true_pulseheight[ic] - self._ph[ib, ic]

                else:
                    raise ValueError('Choose fqlc method from "mmd" (minimum-minimum difference), "slope" or "true_ph".')

                if self._fql_voltage is not None: # FQL voltage is known and provided -> correct only for integer multiples
                    self._flux_loss[ib, ic] = self._fql_voltage[ib, ic] * np.ceil((self._flux_loss[ib, ic] - self._thresh[ic])/self._fql_voltage[ib, ic])

                if self._return_shift_value:
                    return self._flux_loss[ib, ic]

                if self._flux_loss[ib, ic] >= self._thresh[ic]: # only correct actual fql, not baseline drifts or the like
                    _mp = MainParameters(bcs={'length': 1}, fbl=dict(model=0, where=1/8))(event[ib, ic]) # recalculate onset without smoothing to be more precise
                    self._t0[ib, ic] = int(_mp[locs[ic]])
                    self._corrected_event[ib, ic, int(self._t0[ib, ic])+1:] += self._flux_loss[ib, ic]

        if was_1d:
            return self._corrected_event[0, 0]
        elif orig_shape is not None:
            self._corrected_event = self._corrected_event.reshape(orig_shape)
        return self._corrected_event

    def preview(self, event):
        event = np.atleast_2d(event)
        self(event)
        t = (np.arange(event.shape[-1])-event.shape[-1]/4)*0.04 # convert samples to ms

        if event.ndim > 1:
            event.reshape(-1, event.shape[-1])
            d = {}
            ax = {}
            for ic in range(event.shape[0]):
                print(self._event_nobl.shape, self._corrected_event.shape)
                d[f"channel {ic} uncorrected"] = [t, self._event_nobl[0, ic]]
                d[f"channel {ic} FQL-corrected"] = [t, self._corrected_event[ic]]
        else:
            d = {'Event': [t, self._event_nobl[0, 0]],
                 'FQL corrected event': [t, self._corrected_event[0, 0]]}
        ax = {"xaxis": {"label": "Time [ms]"},
              "yaxis": {"label": "Voltage [V]"}}

        return dict(axes=ax, line=d)

    @property
    def batch_support(self):
        return 'full'
