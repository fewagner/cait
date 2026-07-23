import numpy as np

from ..functionbase import ScalarFncBaseclass
from ..processing.boxcarsmoothing import BoxCarSmoothing
from ..processing.fluxquantumlosscorrection import FluxQuantumLossCorrection
from ..processing.removebaseline import RemoveBaseline
from .calcmp import CalcMP


class SaturationTime(ScalarFncBaseclass):
    """
    Calculates the saturation time for a pulse (one channel), given by the time that the pulse spends above the saturation level, calculated around the pulse maximum. The point in time where the pulse falling flank of the pulse dips below the saturation level is calculated after moving-average smoothing, if the parameter "smoothing_length" is set to a positive (integer) value, while the corresponding value for the rising edge is always calculated without smoothing, due to the rise typically being much faster than the decay. If the pulse does not fall below the saturation level before and/or after its maximum, a saturation time of -1 is returned as a flag that reconstruction did not work, while keeping the float nature of the return values for a smoother workflow.

    :param saturation_level_mode: The mode defining how the saturation level given in the parameter saturation_level is interpreted. Passing "relative" interprets the saturation_level relative to the ph of the considered pulse (e.g. saturation_level=0.9 to calculate the time that the pulse is above 90% of its pulse height). Passing "absolute" interprets the saturation_level as an absolute voltage value. Defaults to "relative".
    :type saturation_level_mode": str
    :param saturation_level: The saturation level, either relative to the pulse height of the considered event, or as an absolute voltage value, as defined by the parameter saturation_level_mode. Defaults to 0.9.
    :type saturation_level: float
    :param smoothing_length: Length of the moving average (box-car-) smoothing to be applied before finding the moment where the pulse decays below the saturation level. Defaults to None.
    :type smoothing_length: int
    :param peak_bounds: Peak bounds, relative to the record window, for the calculation of main parameters, defining where the maximum of the pulse can be found. The saturation time will be calculated around this maximum, so the choice of peak bounds can exclude pileups or SQUID resets. Defaults to (0.23, 0.29).
    :type peak_bounds: tuple
    :param fqlc_method: One of three methods for the flux quantum loss (FQL) correction that is applied before the calculation of the saturation time: "mmd", "slope" or "true_ph". "mmd" (mininmum-minimum difference) calculates the FQL as difference between the minima before and after the pulse (with some fluctuation mitigation). "slope" calculates the FQL as the slope of the event. "true_ph" assumes that the true pulse height (without FQL) is known - and provided in the parameter true_pulseheight - and calculates the FQL as the difference between true and apparent pulse height. Defaults to the recommended "mmd".
    :type fqlc_method: str
    :param fql_voltage: If the voltage drop of an FQL is known and provided here, the correction shift will be an integer multiple of this value, instead of any value. In this case, the number of FQ lost is determined by subtracting the threshold (param "fqlc_thresh") from the calculated shift and then rounding up to an integer multiple of fql_voltage. Defaults to None.
    :type fql_voltage: float, optional
    :param fqlc_thresh: Minimum shift value to accept as FQL and correct for. Smaller or negative values are assumed to not stem from FQLs. If no value is given, fqlc_thresh defaults to one third of fql_voltage. If no value is given for fql_voltage either, fqlc_thresh defaults to 0.2 V.
    :type fqlc_thresh: float, optional
    :param true_pulseheight: The known true pulse height as defined by the maximum possible pulse height for fully saturated pulses. Necessary for fqlc_method "true_ph". Defaults to None.
    :type true_pulseheight: float, optional
    
    :return: Saturation time in ms.
    :rtype: float

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

        # Preview the working of the SaturationTime calculation
        vai.Preview(events, vai.SaturationTime())

        # Or calculate it for all events.
        sat_times = vai.apply(vai.SaturationTime(), events)

        # Check the distribution of saturation time values
        vai.Histogram(sat_times)

    .. image:: media/SaturationTime_preview.png
    """
    _outputs = [
        ("sat_time", float),
    ]

    def __init__(self, 
                 saturation_level_mode: str = "relative", 
                 saturation_level: float = 0.9, 
                 smoothing_length: int = None, 
                 peak_bounds: tuple = (.23, .29), 
                 fqlc_method: str = "mmd", 
                 fql_voltage: float = None, 
                 fqlc_thresh: float = None, 
                 true_pulseheight: float = None):
        self._mp = CalcMP(peak_bounds=peak_bounds)
        self._rm_bl_standard = RemoveBaseline()
        self._saturation_level_mode = saturation_level_mode
        self._saturation_level = saturation_level
        self._smoothing_length = smoothing_length
        self._fqlc_method = fqlc_method
        self._fql_voltage = fql_voltage
        self._fqlc_thresh = fqlc_thresh
        self._true_pulseheight = true_pulseheight

        if self._saturation_level_mode not in ["relative", "absolute"]:
            raise ValueError('Choose saturation level mode from "relative" or "absolute".')
        
        if self._fqlc_thresh is None:
            if self._fql_voltage is not None:
                self._fqlc_thresh = self._fql_voltage/3
            else:
                self._fqlc_thresh = 0.2
        self._fqlc = FluxQuantumLossCorrection(method=self._fqlc_method, 
                                               fql_voltage=self._fql_voltage, 
                                               thresh=self._fqlc_thresh, 
                                               true_pulseheight=self._true_pulseheight)

        if self._smoothing_length is not None:
            self._BCSmoothing = BoxCarSmoothing(length=self._smoothing_length)

    def __call__(self, event):
        event = self._rm_bl_standard(event)
        self._ev_fqlc = self._fqlc(event)
        self._shift = self._ev_fqlc[-1] - event[-1]
        # Not calculated using self._ev_fqlc, as the standard baseline removal in CalcMP can lead to the wrong pulseheight if there are pre-trigger pileups.
        self._ph, self._t0, _, self._t_max, _, _, self._t_end, _, _ = self._mp(event) 
        self._ph += self._shift

        # If the saturation level mode is "relative", multiply by the ph to get an absolute value
        if self._saturation_level_mode == "relative": 
            self._absolute_saturation_level = self._saturation_level * self._ph
        else:
            self._absolute_saturation_level = self._saturation_level
        
        if np.any(np.flip(self._ev_fqlc[:int(self._t_max)]) < self._absolute_saturation_level):
            self._t_start = self._t_max - np.argmax(np.flip(self._ev_fqlc[:int(self._t_max)]) < self._absolute_saturation_level)
        # Pulse does not go below saturation level before t_max
        else:
            return -1 
        
        if self._smoothing_length is not None:
            self._ev_fqlc_presmoothing = np.copy(self._ev_fqlc)
            self._ev_fqlc = self._BCSmoothing(self._ev_fqlc_presmoothing)
        
        # Catches cases where the pulse does not fall below the saturation level
        if np.any(self._ev_fqlc[int(self._t_start):] < self._absolute_saturation_level): 
            self._t_dec = np.argmax(self._ev_fqlc[int(self._t_max):] < self._absolute_saturation_level) + int(self._t_max)
        # Pulse does not decay below saturation level (within record window)
        else:
            return -1 
        
        # Converts array indices to ms for a sampling rate of 25kHz
        return 0.04 * (self._t_dec - self._t_start) 

        
    def preview(self, event):
        self(event)
        self._tax = np.arange(0,len(event)*0.04, 0.04)-len(event)/4*0.04
        if self._smoothing_length is not None and self._smoothing_length > 0:
            d = {'Event': [self._tax, event],
                 'Shifted, smoothed event': [self._tax, self._ev_fqlc],
                 'Saturation time': [[self._tax[int(self._t_start)], self._tax[self._t_dec]], [self._absolute_saturation_level, self._absolute_saturation_level]]}
        else:
            d = {'Event': [self._tax, event],
                 'Shifted event': [self._tax, self._ev_fqlc],
                 'Saturation time': [[self._tax[int(self._t_start)], self._tax[self._t_dec]], [self._absolute_saturation_level, self._absolute_saturation_level]]}
        ax = {'xaxis': {'label': "Time [ms]"}, 'yaxis': {'label': "Voltage [V]"}}
    
        return dict(line = d, scatter = {'Saturation level': [[self._tax[self._t_dec], self._tax[int(self._t_start)]], [self._absolute_saturation_level, self._absolute_saturation_level]], 'ph': [[self._tax[int(self._t_max)]], [self._ph]]}, axes=ax)
        
    def batch_support(self):
        return "none"