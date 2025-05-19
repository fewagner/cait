from typing import List, Union

import numpy as np
import cait.versatile as vai

from ..processing.fluxquantumlosscorrection import FluxQuantumLossCorrection
from ..processing.removebaseline import RemoveBaseline
from .calcmp import CalcMP
from .templatefit import shift_arrays, TemplateFit

########################
### CLASS DEFINITION ###
########################
class TemplateFit_FQLC(TemplateFit):
    """
    This class extends :class:`cait.versatile.TemplateFit`. Perform a template fit for single-channel data, i.e. fit a numeric SEV to data and additionally allow for correction of flux quantum losses (FQLs) and for auto-detection of post-pulse-pileups, excluding pileup-affected parts of the voltage trace from the fit.

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

        # The template fit function
        f = vai.TemplateFit_FQLC(sev, truncation_limit=SHIFT)

        # Preview the working of the FQL corrected TemplateFit
        vai.Preview(events, f)

        # Or calculate the fit values.
        # Make sure to use the 'discard' flag to exclude invalid results!
        fitpars, shift, rms, discard = vai.apply(f, events)

        # Check the distribution of pulse heights
        vai.Histogram(fitpars[~discard])

    .. image:: media/TemplateFit_FQLC_preview.png
    """
    def __init__(self, 
                 sev: np.ndarray,
                 bl_poly_order: int = None,
                 truncation_limit: float = None,
                 xdata: List[float] = None,
                 fit_onset: bool = True,
                 max_shift: int = 50,
                 fqlc_method: str = "mmd",
                 fql_voltage: float = None,
                 fqlc_thresh: float = None,
                 true_pulseheight: float = None,
                 pileup_trigger_rms: float = 8,
                 pileup_trigger_window_size: Union[int, float] = 1/20,
                 pileup_buffer_samples: int = 50):
        if np.array(sev).ndim>1:
            raise ValueError(f"{self.__class__.__name__} can only process single-channel data. Multi-dimensional templates are not supported.")
        
        # Handle as much as possible in existing TemplateFit
        super().__init__(sev=sev, 
                         bl_poly_order=bl_poly_order, 
                         truncation_limit=truncation_limit, 
                         xdata=xdata, 
                         fit_onset=fit_onset, 
                         max_shift=max_shift)
        
        # Values needed in addition to those defined by the superclass
        self._bl_poly_order = bl_poly_order
        self._max_shift = max_shift
        self._pileup_trigger_rms = pileup_trigger_rms
        self._pileup_buffer_samples = pileup_buffer_samples
        self._pileup_trigger_window_size = pileup_trigger_window_size
        self._mp = CalcMP()
        
        if fqlc_thresh is None:
            if fql_voltage is not None:
                fqlc_thresh = fql_voltage/3
            else:
                fqlc_thresh = 0.2

        self._fqlc = FluxQuantumLossCorrection(method=fqlc_method, 
                                               fql_voltage=fql_voltage, 
                                               thresh=fqlc_thresh, 
                                               true_pulseheight=true_pulseheight)
        
        # Override the baseline subtraction defined in the superclass
        self._rm_bl = RemoveBaseline(dict(model="voltage_minimum"))

    def __call__(self, event):
        self._ev_fqlc = self._fqlc(event) # flux quantum loss corrected event

        self._fitpar_length = 1
        if self._bl_poly_order is not None:
            self._fitpar_length = self._bl_poly_order+2

        # Return empty results of correct dimensions and a discard flag. Not raising an error, because it can happen that events have irregular lengths and this treatment still allows for bulk processing of large amounts of data.
        if event.shape != self._sev.shape: 
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
            if isinstance(self._pileup_trigger_window_size, int):
                window_size = self._pileup_trigger_window_size
            else:
                window_size = int(event.shape[-1]*self._pileup_trigger_window_size)

            self._peak_pos = vai.trigger_zscore(self._ev_fqlc, 
                                                window_size, 
                                                self._pileup_trigger_rms)[0]

            if len(self._peak_pos)>=2:
                # Could be that a peak is found before the actual event, then the event peak would be second, in that case we don't want to incluse the event pulse from the fit.
                if self._peak_pos[1] > self._ev_fqlc.shape[0]/4 + self._max_shift: 
                    self._no_pileup[self._peak_pos[1]-self._pileup_buffer_samples:] = False # pulse starts earlier than it is above 5 sigma
                else: # pileup before the event pulse, discard fit
                    discard = True

        # Nothing left to fit to
        if not np.any(self._below_truncation_limit*self._no_pileup): 
            # Return empty results of correct dimensions and a discard flag
            if self._fitpar_length == 1: 
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