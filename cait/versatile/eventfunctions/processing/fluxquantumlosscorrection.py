import numpy as np

from ..functionbase import FncBaseClass
from ..scalarfunctions.calcmp import CalcMP

class RemoveBaseline_VoltageMinimum(FncBaseClass):
    """
    Alternative method to remove the baseline, subtracting a constant value. The constant value is close to the minimum of the voltage trace before the pulse (with some fluctuation mitigation). This works better than the standard method when there is a pileup in the pre-trigger region. This code only works for one channel at a time.
    Should be integrated into the standard vai.RemoveBaseline() method, but I ran into troubles because the present variant does not support multiple channels.

    :return: Event with baseline removed.
    :rtype: numpy.ndarray
    """
    def __init__(self):
        self._mp = CalcMP()

    def __call__(self, event):
        _, self._t0, _, _, _, _, _, _, _ = self._mp(event) # get onset
        if int(self._t0) > 600:
            self._t_min = 600 + np.argmin(event[600:int(self._t0)]) # get baseline value as average of samples before the minimum. Minimum likely to sit at negative noise fluctuation, so do not include in average. Average before and not after to not average over part of the pulse.
        else:
            self._t_min = 600
        self._bl_value = np.mean(event[self._t_min-600:self._t_min-100])
        
        self._shifted_event = event - self._bl_value

        return self._shifted_event
    
    @property
    def batch_support(self):
        return 'none'
        
    def preview(self, event) -> dict:
        self(event)
        d = {'event': [None, event],
                 'baseline removed': [None, self._shifted_event]}
        return dict(line = d)


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
    """
    def __init__(self, method: str = "mmd", fql_voltage: float = None, thresh: float = 0.2, true_pulseheight: float = None, return_shift_value: bool = False):
        self._remove_baseline = RemoveBaseline_VoltageMinimum()
        self._mp = CalcMP()
        self._method = method
        self._thresh = thresh
        self._true_pulseheight = true_pulseheight
        self._fql_voltage = fql_voltage
        self._return_shift_value = return_shift_value

    def __call__(self, event):
        self._ph, self._t0, _, self._t_max, _, _, self._t_end, _, self._lin_drift = self._mp(event)
        self._event_nobl = self._remove_baseline(event)

        if self._method == "mmd": # find minima of voltage trace before and after pulse, take difference
            if int(self._t_max) > 600:
                self._t_min_1 = 600 + np.argmin(self._event_nobl[600:int(self._t_max)]) # start at 600 so we don't end up out of bounds when averaging later
            else:
                self._t_min_1 = 600
            if int(self._t_max) < len(event):
                self._t_min_2 = max(600,int(self._t_max)) + np.argmin(self._event_nobl[max(600,int(self._t_max)):])
            else:
                self._t_min_2 = len(event)
            # here, the 600 catches the case of a faulty or non-event with tmax before sample #600
            # take the values for the average not around the minima as the minima are subject to fluctuations.
            # also don't take values after the minima as the (/another) pulse (pileup) might come into play there.
            self._baseline_mean_1 = np.mean(self._event_nobl[self._t_min_1-600:self._t_min_1-100])
            self._baseline_mean_2 = np.mean(self._event_nobl[self._t_min_2-600:self._t_min_2-100])
            self._flux_loss = self._baseline_mean_1-self._baseline_mean_2
        elif self._method == "slope":
            self._slope = self._lin_drift * event.shape[-1]
            self._flux_loss = -self._slope
        elif self._method == "true_ph":
            if self._true_pulseheight is None:
                raise ValueError("True pulse height needs to be provided for the fqlc method \"true_ph\".")
                return None
            self._flux_loss = self._true_pulseheight - self._ph
        else:
            raise ValueError("Choose fqlc method from \"mmd\" (minimum-minimum difference), \"slope\" or \"true_ph\"")
            return None
        
        self._corrected_event = self._event_nobl.copy()
        
        if self._fql_voltage is not None: # FQL voltage is known and provided -> correct only for integer multiples
            self._flux_loss = self._fql_voltage * np.ceil((self._flux_loss - self._thresh)/self._fql_voltage)
        
        if self._return_shift_value:
            return self._flux_loss
            
        if self._flux_loss >= self._thresh: # only correct actual fql, not baseline drifts or the like
            self._mp = CalcMP(box_car_smoothing={'length': 1}) # recalculate onset without smoothing to be more precise
            _, self._t0, _, _, _, _, _, _, _ = self._mp(event) 
            self._corrected_event[int(self._t0)+1:] += self._flux_loss

        return self._corrected_event
        
    def preview(self, event):
        self(event)
        t = (np.arange(len(event))-len(event)/4)*0.04 # convert samples to ms

        d = {'Event': [t, self._event_nobl],
             'FQL corrected event': [t, self._corrected_event]}
        ax = {"xaxis": {"label": "Time [ms]"}, "yaxis": {"label": "Voltage [V]"}}

        return dict(axes=ax, line = d)
        
    def batch_support(self):
        return 'none'