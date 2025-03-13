import numpy as np

from ..functionbase import FncBaseClass
from ..scalarfunctions.calcmp import CalcMP

class RemoveBaseline_new(FncBaseClass):
    """
    Alternative method to remove the baseline, subtracting a constant value. The constant value is close to the minimum of the voltage trace before the pulse (with some fluctuation mitigation). This works better than the standard method when there is a pileup in the pre-trigger region.
    Does NOT work for more than one channel yet! Should be integrated into the standard vai.RemoveBaseline() method, but I ran into troubles because the present variant does not support multiple channels.
    """
    def __init__(self):
        self._mp = CalcMP()

    def __call__(self, event):
        _, self._t0, _, _, _, _, _, _, _ = self._mp(event) # get onset
        self._tmin = 600 + np.argmin(event[600:int(self._t0)]) # get baseline value as average of samples before the minimum. Minimum likely to sit at negative noise fluctuation, so do not include in average. Average before and not after to not average over part of the pulse.
        self._bl_value = np.mean(event[self._tmin-600:self._tmin-100])
        
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


class FQLC(FncBaseClass):
    """
    Correct event for flux quantum loss (FQL). Works only for one channel.
    
    :param method: One of three methods: "mmd", "slope" or "satv". "mmd" (mininmum-minimum difference) calculates the FQL as difference between the minima before and after the pulse (with some fluctuation mitigation). "slope" calculates the FQL as the slope of the event. "satv" assumes that the true pulse height (without) FQL is known - and given in the argument sat_v - and calculates the FQL as the difference between true and apparent pulse height. Defaults to the recommended method, "mmd".
    :type method: str
    :param thresh: Minimum shift value to accept and correct for. Smaller or negative values are assumed to not stem from FQLs.
    :type thresh: float
    :param sat_v: The known true pulse height as defined by the saturation level. Necessary for method "satv".
    :type sat_v: float
    :param known_fql_V: If the voltage drop of a FQL is known and provided here, the correction shift will be an integer multiple of this value, instead of any value. Defaults to None.
    :type known_fql_V: float
    :param val_not_ev: If True, not the shifted event but the shift value is returned. Defaults to False.
    :type val_not_ev: bool

    :return: Event with FQL corrected or value of shift (if val_not_ev is set to True)
    :rtype: Union[numpy.ndarray, float]
    """
    def __init__(self, method: str = "mmd", thresh: float = 0.5, sat_v: float = 4.00, known_fql_V: float=None, val_not_ev=False):
        self._remove_baseline = RemoveBaseline_new()
        self._mp = CalcMP()
        self._method = method
        self._thresh = thresh
        self._sat_v = sat_v
        self._known_fql_V = known_fql_V
        self._val_not_ev = val_not_ev

    def __call__(self, event):
        self._ph, self._t0, _, self._t_max, _, _, self._t_end, _, self._lin_drift = self._mp(event)
        self._event_nobl = self._remove_baseline(event)

        if self._method == "mmd":
            self._tmin1 = 600 + np.argmin(self._event_nobl[600:int(self._t_max)]) #start at 600 so we don't end up out of bounds when averaging later
            self._tmin2 = max(600,int(self._t_max)) + np.argmin(self._event_nobl[max(600,int(self._t_max)):])
            #here, the 600 catches the case of a faulty or non-event with tmax before sample #600
            # take the values for the average not around the minima as the minima are subject to fluctuations.
            # also don't take values after the minima as the (/another) pulse (pileup) might come into play there.
            self._blavg1 = np.mean(self._event_nobl[self._tmin1-600:self._tmin1-100])
            self._blavg2 = np.mean(self._event_nobl[self._tmin2-600:self._tmin2-100])
            flux_loss = self._blavg1-self._blavg2 #minmindiff approach
        elif self._method == "slope":
            self.slope = self._lin_drift * event.shape[-1] # lin_drift times length
            flux_loss = -self.slope # slope
        elif self._method == "satv":
            flux_loss = self._sat_v - self._ph # known satV
        else:
            print("Choose method from \"mmd\" (minmindiff), \"slope\" or \"satv\"")
            flux_loss = self._blavg1-self._blavg2 #minmindiff approach
        


        self._shifted_event = self._event_nobl.copy()
        
        if self._known_fql_V is not None: # FQL voltage is known and provided -> correct only for integer multiples
            flux_loss = self._known_fql_V * np.ceil(flux_loss/self._known_fql_V - self._thresh)
        
        if self._val_not_ev:
            return flux_loss
            
        if flux_loss >= self._thresh: #only correct actual fql, not baseline drifts or the like
            self._mp = CalcMP(box_car_smoothing={'length': 1}) # recalculate t0 without smoothing to be more precise
            _, self._t0, _, _, _, _, _, _, _ = self._mp(event) 
            self._shifted_event[int(self._t0)+1:] += flux_loss

        return self._shifted_event
        
    def preview(self, event):
        self(event)
        t = (np.arange(len(event))-len(event)/4)*0.04 # samples to ms

        d = {'Event': [t, self._event_nobl],
             'FQL corrected event': [t, self._shifted_event]}
        ax = {"xaxis": {"label": "Time [ms]"}, "yaxis": {"label": "Voltage [V]"}}

        return dict(axes=ax, line = d)
        
    def batch_support(self):
        return 'none'