from typing import List
from typing import Union
from functools import partial

import numpy as np
import cait.versatile as vai

from ..functionbase import FncBaseClass
from ..processing.fluxquantumlosscorrection import FluxQuantumLossCorrection
from ..processing.fluxquantumlosscorrection import RemoveBaseline_VoltageMinimum
from .calcmp import CalcMP
from .templatefit import shift_arrays, _TemplateCacheSimple, _TemplateCachePoly


class All_Peak_Positions(FncBaseClass):
    """
    This is a copy-paste of the class "NPeaks", which calculates the number of peaks found, but it instead returns the indices of the peaks found. The peaks are found in an event by applying a moving z-score trigger to the trace.

    :param window_size: The size of the sliding window for the z-score trigger. If it's an integer, this will be the number of samples in the window, if it's a float, the number will be scaled to the record_length of the events (e.g. if 1/20, the window will be 1/20th of the record_length). The larger the window, the more robust the trigger is. However, you will miss potential triggers in the beginning of the event because the first sample that can reliably searched after applying the moving z-score is at ``window_size``. Defaults to 1/20
    :type window_size: Union[int, float], optional
    :param threshold: The threshold (in sigmas) for the trigger, defaults to 3.5
    :type threshold: float, optional

    :return: Indices of peaks found.
    :rtype: numpy.ndarray
    """
    def __init__(self, window_size: Union[int, float] = 1/20, threshold: float = 3.5):
        self._window_size = window_size
        self._threshold = threshold

        if isinstance(window_size, int):
            self._trigger = partial(vai.trigger_zscore, 
                                    record_length=window_size,
                                    threshold=threshold)
        else:
            self._trigger = None

        self._trigger_inds = list()

    def __call__(self, event):
        if np.array(event).ndim > 1:
            raise NotImplementedError(f"Multi-channel events are not supported by {self.__class__.__name__}")
        if self._trigger is None:
            record_length = np.array(event).shape[-1]
            window_size = int(record_length*self._window_size)
            self._trigger = partial(vai.trigger_zscore, 
                                    record_length=window_size,
                                    threshold=self._threshold)
            
        self._trigger_inds, _ = self._trigger(event)
        return self._trigger_inds
    
    @property
    def batch_support(self):
        return 'none'
    
    def preview(self, event):
        n = len(self(event))
        x = np.arange(event.shape[-1])
        
        l = {'event': [x, event]}
        s = {'triggers': [x[self._trigger_inds] if n>0 else [],
                             event[self._trigger_inds] if n>0 else []]}
        return dict(line=l, scatter=s)


########################
### CLASS DEFINITION ###
########################
class TemplateFit_FQLC(FncBaseClass):
    """
    This is an extended copy of the class "TemplateFit". Perform a template fit for single-channel data, i.e. fit a numeric SEV to data with possibility to also specify a polynomial baseline model and a truncation limit. Additionally allows for correction of flux quantum losses (FQLs) and for auto-detection of post-pulse-pileups, excluding pileup-affected parts of the voltage trace from the fit.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    :param sev: The template (SEV) to use in the fit.
    :type sev: np.ndarray
    :param bl_poly_order: The baseline model to use in the fit. Has to be a non-zero integer or None. If 0, a constant offset is fitted, if 1, a linear baseline is assumed, etc. As a constant baseline is always removed before the fit in the process of flux quantum loss correction, setting bl_poly_order to None is effectively the same as setting it to 0. Defaults to None.
    :type bl_poly_order: int
    :param truncation_limit: If not None, a truncated fit is performed: all samples between the first and the last sample above 'truncation_limit' are ignored in the fit. To determine these samples, the baseline of the event is removed by fitting a polynomial of order 'bl_poly_order' to the beginning of the record window. Defaults to None, i.e. not performing a truncated fit.
    :type truncation_limit: float
    :param xdata: The x-data array used to evaluate the baseline model. If None, the default ``xdata=np.linspace(0, 1, len(sev))`` is used, defaults to None.
    :type xdata: np.ndarray
    :param fit_onset: If True, the onset value is fitted. If False, the event is fitted as is, defaults to True
    :type fit_onset: bool
    :param max_shift: The maximum shift value (in samples) to search for a minimum. The onset fit will search the minimum for shifts in ``(-max_shift, +max_shift)``.
    :type max_shift: int
    :param fqlc_method: One of three methods for the flux quantum loss (FQL) correction that is applied before the calculation of the saturation time: "mmd", "slope" or "true_ph". "mmd" (mininmum-minimum difference) calculates the FQL as difference between the minima before and after the pulse (with some fluctuation mitigation). "slope" calculates the FQL as the slope of the event. "true_ph" assumes that the true pulse height (without FQL) is known - and provided in the parameter true_pulseheight - and calculates the FQL as the difference between true and apparent pulse height. Defaults to the recommended "mmd".
    :type fqlc_method: str
    :param fql_voltage: If the voltage drop of an FQL is known and provided here, the correction shift will be an integer multiple of this value, instead of any value. In this case, the number of FQ lost is determined by subtracting the threshold (param "fqlc_thresh") from the calculated shift and then rounding up to an integer multiple of fql_voltage. Defaults to None.
    :type fql_voltage: float, optional
    :param fqlc_thresh: Minimum shift value to accept as FQL and correct for. Smaller or negative values are assumed to not stem from FQLs. If no value is given, fqlc_thresh defaults to one third of fql_voltage. If no value is given for fql_voltage either, fqlc_thresh defaults to 0.2 V.
    :type fqlc_thresh: float, optional
    :param true_pulseheight: The known true pulse height as defined by the maximum possible pulse height for fully saturated pulses. Necessary for fqlc_method "true_ph". Defaults to None.
    :type true_pulseheight: float, optional
    :param pileup_trigger_rms: The threshold (in sigmas) for a moving z-score trigger to find pileup pulses. All samples after the start of a pileup are disregarded for the fit, starting pileup_buffer_samples before the sample that triggers. If None, no pileup detection is carried out. Defaults to 8.
    :type pileup_trigger_rms: float
    :param pileup_trigger_window_size: The size of the sliding window for the pileup-finding z-score trigger. If it's an integer, this will be the number of samples in the window, if it's a float, the number will be scaled to the record_length of the events (e.g. if 1/20, the window will be 1/20th of the record_length). The larger the window, the more robust the trigger is. Defaults to 1/20.
    :type pileup_trigger_window_size: Union[int, float], optional
    :param pileup_buffer_samples: Number of samples to exclude from the fit before the first pileup trigger (the pileup pulse typically starts a bit earlier than it is above the trigger threshold). Thus all, samples after the triggering sample minus pileup_buffer_samples are excluded. Defaults to 50.
    :type pileup_buffer_samples: int
    
    :return: Tuple of fit result, optimal shift, RMS value and a flag indicating whether the result should be discarded due to problems in the fitting procedure ``([amplitude, constant_bl_coeff, linear_bl_coeff, ...], shift, rms)``. If you set ``fit_onset=False``, the ``shift`` value will just be 0. If the fit fails, the discard flag is set to True.
    :rtype: Tuple[np.ndarray, int, float, bool]

    """
    def __init__(self, 
                 sev: np.ndarray,
                 bl_poly_order: int = None,
                 truncation_limit: float = None,
                 xdata: List[float] = None,
                 fit_onset: bool = True,
                 max_shift: int = 50,
                 fqlc_method: str = "mmd",
                 fql_voltage=None,
                 fqlc_thresh=None,
                 true_pulseheight: float = None,
                 pileup_trigger_rms=8,
                 pileup_trigger_window_size=1/20,
                 pileup_buffer_samples=50):
        if np.array(sev).ndim>1:
            raise ValueError(f"{self.__class__.__name__} can only process single-channel data. Multi-dimensional templates are not supported.")
        if not (isinstance(bl_poly_order, int) or bl_poly_order is None):
            raise TypeError(f"'bl_poly_order' has to be a non-zero integer or None, not {type(bl_poly_order)}.")
        elif isinstance(bl_poly_order, int) and bl_poly_order<0:
            raise TypeError(f"'bl_poly_order' has to be a non-negative integer, not {bl_poly_order}.")
        
        self._sev = np.array(sev)
        self._bl_poly_order = bl_poly_order
        self._truncation_limit = truncation_limit
        self._xdata = np.linspace(0, 1, self._sev.shape[-1]) if xdata is None else xdata
        self._max_shift = max_shift
        self._fqlc_method = fqlc_method
        self._fql_voltage = fql_voltage
        self._fqlc_thresh = fqlc_thresh
        self._true_pulseheight=true_pulseheight
        self._pileup_trigger_rms = pileup_trigger_rms
        self._pileup_buffer_samples = pileup_buffer_samples
        
        self._rm_bl = RemoveBaseline_VoltageMinimum()
        self._mp = CalcMP()

        if self._pileup_trigger_rms is not None:
            self._find_peaks = All_Peak_Positions(threshold=pileup_trigger_rms, window_size=pileup_trigger_window_size)

        if bl_poly_order is None:
            self._mode = 'simple'
            self._solver = _TemplateCacheSimple(sev=self._sev, 
                                                fit_onset=fit_onset, 
                                                max_shift=self._max_shift)
        else:
            self._mode = 'poly'
            self._solver = _TemplateCachePoly(sev=self._sev, 
                                              xdata=self._xdata, 
                                              order=bl_poly_order, 
                                              fit_onset=fit_onset, 
                                              max_shift=self._max_shift)

        if self._fqlc_thresh is None:
            if self._fql_voltage is not None:
                self._fqlc_thresh = self._fql_voltage/3
            else:
                self._fqlc_thresh = 0.2
        self._fqlc = FluxQuantumLossCorrection(method=self._fqlc_method, fql_voltage=self._fql_voltage, thresh=self._fqlc_thresh, true_pulseheight=self._true_pulseheight)

    def __call__(self, event):
        self._ev_fqlc = self._fqlc(event) # flux quantum loss corrected event

        self._fitpar_length = 1
        if self._bl_poly_order is not None:
            self._fitpar_length = self._bl_poly_order+2
        if event.shape != self._sev.shape: # Return empty results of correct dimensions and a discard flag. Not raising an error, because it can happen that events have irregular lengths and this treatment still allows for bulk processing of large amounts of data.
            if self._fitpar_length == 1:
                return 0, 0, 0, True
            else:
                return [0,]*self._fitpar_length, 0, 0, True

        self._ph, self._t0, _, self._t_max, _, _, self._t_end, _, _ = self._mp(event)
        
        #------------truncated fit range-----------
        self._below_truncation_limit = None
        self._below_truncation_limit = np.ones(self._ev_fqlc.shape[0], dtype=bool)
        if self._truncation_limit is not None:
            self._below_truncation_limit = (self._ev_fqlc < self._truncation_limit).flatten()

        #-----------pileup mitigation---------------
        #if 2 or more peaks above 5 sigma are found, everything after the 2nd peak's start is discarded
        self._no_pileup = np.ones(self._ev_fqlc.shape[0], dtype=bool)
        discard = False

        if self._pileup_trigger_rms is not None:
            self._peak_pos = self._find_peaks(self._ev_fqlc)
            if len(self._peak_pos)>=2:
                if self._peak_pos[1] > self._ev_fqlc.shape[0]/4 + self._max_shift: # could be that a peak is found before the actual event, then the event peak would be second, in that case we don't want to incluse the event pulse from the fit.
                    self._no_pileup[self._peak_pos[1]-self._pileup_buffer_samples:] = False # pulse starts earlier than it is above 5 sigma
                else: # pileup before the event pulse, discard fit
                    discard = True

        if not np.any(self._below_truncation_limit*self._no_pileup): # nothing left to fit to
            if self._fitpar_length == 1: # return empty results of correct dimensions and a discard flag
                return 0, 0, 0, True
            else:
                return [0,]*self._fitpar_length, 0, 0, True
            
        fitpars, shift, rms = self._solver(self._ev_fqlc, flag=self._below_truncation_limit*self._no_pileup)

        return fitpars, shift, rms, discard
    
    @property
    def batch_support(self):
        return "none"
    
    def preview(self, event):
        fitpars, shift, rms, discard = self(event)
        if discard:
            shift = 0
            fitpars=[0]
            if self._mode == 'simple':
                fitpars=0

        shifted_sev, shifted_x = shift_arrays(self._sev, self._xdata, j=shift)

        t = (np.arange(len(event))-len(event)/4)*0.04 # samples to ms

        if self._mode == 'simple':
            fit_sev = fitpars*shifted_sev
        else:
            fit_sev = fitpars[0]*shifted_sev + np.sum(
                [fitpars[k+1]*shifted_x**k for k in range(len(fitpars)-1)],
                axis=0)
        
        l = {"Event": [t, event], "Shifted event": [t, self._ev_fqlc], "Template fit": [t, fit_sev]}
        if self._truncation_limit is not None:
            truncation_line = self._truncation_limit *np.ones_like(event) # self._truncation_limit + event - self._rm_bl(event)
            l["Truncation limit"] = [t, truncation_line]
        #l["rms"] = [t, rms *np.ones_like(event)]

        s = {"Triggers": [t[self._peak_pos].tolist(), self._ev_fqlc[self._peak_pos].tolist()]}
        if len(self._peak_pos)>=2:
            s["Pileup start"] = [[t[self._peak_pos[1]-self._pileup_buffer_samples]], [self._ev_fqlc[self._peak_pos[1]-self._pileup_buffer_samples]]]
        else:
            s["Pileup start"] = [[],[]]
        
        ax = {"xaxis": {"label": "Time [ms]"}, "yaxis": {"label": "Voltage [V]"}}
        
        return dict(axes=ax, line=l, scatter=s)