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
    def __init__(self,
                 method: str = "mmd",
                 fql_voltage: float = None,
                 thresh: float = 0.2,
                 true_pulseheight: float = None,
                 return_shift_value: bool = False):
        self._mp = MainParameters()
        self._method = method
        self._thresh = thresh
        self._true_pulseheight = true_pulseheight
        self._fql_voltage = fql_voltage
        self._return_shift_value = return_shift_value

    def __call__(self, event):
        event = np.array(event)

        orig_shape = None
        was_1d = False
        if event.ndim < 2:
            was_1d = True
            event = event[None, ...]
        elif event.ndim > 2:
            orig_shape = event.shape
            event = event.reshape(-1, event.shape[-1])

        if isinstance(self._thresh, (int, float)):
            self._thresh = np.full(event.shape[0], self._thresh)
        if isinstance(self._true_pulseheight, (int, float)):
            self._thresh = np.full(event.shape[0], self._true_pulseheight)
        if isinstance(self._fql_voltage, (int, float)):
            self._thresh = np.full(event.shape[0], self._fql_voltage)

        self._event_nobl = np.array(event)
        self._corrected_event = np.array(self._event_nobl)

        self._ph = np.zeros(event.shape[0])
        self._t0 = np.zeros(event.shape[0])
        self._t_max = np.zeros(event.shape[0])
        self._t_end = np.zeros(event.shape[0])
        self._lin_drift = np.zeros(event.shape[0])
        self._t_min_1 = np.zeros(event.shape[0], dtype=int)
        self._t_min_2 = np.zeros(event.shape[0], dtype=int)
        self._baseline_mean_1 = np.zeros(event.shape[0])
        self._baseline_mean_2 = np.zeros(event.shape[0])
        self._flux_loss = np.zeros(event.shape[0])
        self._slope = np.zeros(event.shape[0])

        for ic in range(event.shape[0]):
            # Calculate main parameters
            mp = np.array(self._mp(event[ic]))
            self._t0[ic] = min(mp[10], mp[12])
            self._ph[ic], self._t_max[ic], self._lin_drift[ic] = \
                    mp[np.array([0, 1, 6])]
            # The old t_end was relative to the record window, this is lost in the
            # new MainParameters, but the position relative to the peak position should
            # be close enough
            self._t_end[ic] = mp[4] + mp[1]
            # t0 also needs to be relative to the record window
            self._t0[ic] += event.shape[-1] // 4

            # Find minima of voltage trace before and after pulse, take difference
            if self._method == "mmd":
                # Model was developed for a fixed record length. Here, the indices
                # 600 and 100 were found to work well. Consequently, we scale the
                # indices now for an arbitrary record length.
                k1, k2 = int(600/2**14*event.shape[-1]), int(100/2**14*event.shape[-1])

                if int(self._t_max[ic]) > k1:
                    self._t_min_1[ic] = k1 + np.argmin(self._event_nobl[ic, k1:int(self._t_max[ic])]) # start at 600 so we don't end up out of bounds when averaging later
                else:
                    self._t_min_1[ic] = k1
                if int(self._t_max[ic]) < event.shape[-1]:
                    self._t_min_2[ic] = max(k1,int(self._t_max[ic])) + np.argmin(self._event_nobl[ic, max(k1,int(self._t_max[ic])):])
                else:
                    self._t_min_2[ic] = event.shape[-1]
                # here, the 600 catches the case of a faulty or non-event with tmax before sample #600
                # take the values for the average not around the minima as the minima are subject to fluctuations.
                # also don't take values after the minima as the (/another) pulse (pileup) might come into play there.
                self._baseline_mean_1[ic] = np.mean(self._event_nobl[ic, self._t_min_1[ic]-k1:self._t_min_1[ic]-k2])
                self._baseline_mean_2[ic] = np.mean(self._event_nobl[ic, self._t_min_2[ic]-k1:self._t_min_2[ic]-k2])
                self._flux_loss[ic] = self._baseline_mean_1[ic]-self._baseline_mean_2[ic]

            elif self._method == "slope":
                self._slope[ic] = self._lin_drift[ic] * event.shape[-1]
                self._flux_loss[ic] = -self._slope[ic]

            elif self._method == "true_ph":
                if self._true_pulseheight is None:
                    raise ValueError('True pulse height needs to be provided for the fqlc method "true_ph".')

                self._flux_loss[ic] = self._true_pulseheight[ic] - self._ph[ic]

            else:
                raise ValueError('Choose fqlc method from "mmd" (minimum-minimum difference), "slope" or "true_ph".')

            if self._fql_voltage is not None: # FQL voltage is known and provided -> correct only for integer multiples
                self._flux_loss[ic] = self._fql_voltage[ic] * np.ceil((self._flux_loss[ic] - self._thresh[ic])/self._fql_voltage[ic])

            if self._return_shift_value:
                return self._flux_loss[ic]

            if self._flux_loss[ic] >= self._thresh[ic]: # only correct actual fql, not baseline drifts or the like
                mp = MainParameters(bcs={'length': 1}, fbl=dict(model=0, where=1/8))(event[ic]) # recalculate onset without smoothing to be more precise
                #self._t0[ic] = mp(event[ic])[2]
                self._t0[ic] = min(mp[10], mp[12])#mp[1]#0]
                #self._t0[ic] += event.shape[-1] // 4
                self._corrected_event[ic, int(self._t0[ic])+1:] += self._flux_loss[ic]

        if was_1d:
            return self._corrected_event[0]
        elif orig_shape is not None:
            self._corrected_event = self._corrected_event.reshape(orig_shape)
        return self._corrected_event

    def preview(self, event):
        self(event)
        t = (np.arange(event.shape[-1])-event.shape[-1]/4)*0.04 # convert samples to ms

        if event.ndim > 1:
            event.reshape(-1, event.shape[-1])
            d = {}
            ax = {}
            for ic in range(event.shape[0]):
                d[f"channel {ic} uncorrected"] = [t, self._event_nobl[ic]]
                d[f"channel {ic} FQL-corrected"] = [t, self._corrected_event[ic]]
        else:
            d = {'Event': [t, self._event_nobl[0]],
                 'FQL corrected event': [t, self._corrected_event[0]]}
        ax = {"xaxis": {"label": "Time [ms]"},
              "yaxis": {"label": "Voltage [V]"}}

        return dict(axes=ax, line=d)


    @property
    def batch_support(self):
        return 'full'
