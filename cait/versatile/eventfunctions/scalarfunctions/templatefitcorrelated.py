from typing import List

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import LinAlgError

from ..functionbase import FncBaseClass
from ..processing.removebaseline import RemoveBaseline
from .templatefit import _TemplateCacheSimple, _TemplateCachePoly, shift_arrays

############################
###### HELPER CLASSES ######
############################
class _TemplateCacheCorrelated:
    """
    Helper class that performs the correlated template fit for multiple channels with a (non-)trivial baseline. It caches the matrices used for solving the minimization problem for the regular (not truncated) fit to increase computational efficiency.
    See https://edoc.ub.uni-muenchen.de/23762/ and https://mediatum.ub.tum.de/?id=1294132 for details.

    :param sev: The reference event (at least two channels).
    :type sev: np.ndarray
    :param xdata: The x-data to use for the baseline model evaluation (exactly 1-d).
    :type xdata: np.ndarray
    :param order: A list of the orders of the baseline polynomials to be fitted for each channel separately. If an entry is None, this channel's baseline will not be fitted.
    :type order: List[int]
    :param fit_onset: List of as many entries as there are channels in 'sev'. For any True, the onset value for the respective channel is fitted. If False, the respective channel just moves together with the fitted channels (but does not contribute to the minimization)
    :type fit_onset: List[bool]
    :param max_shift: The maximum shift value (in samples) to search for a minimum. The onset fit will search the minimum for shifts in ``(-max_shift, +max_shift)``. Defaults to 50 samples
    :type max_shift: int, optional
    """
    def __init__(self, 
                 sev: np.ndarray, 
                 xdata: np.ndarray, 
                 order: List[int], 
                 fit_onset: List[bool],
                 max_shift: int = 50):
        
        self._sev = sev
        self._n_channels = sev.shape[0]
        self._xdata = xdata
        self._order = order
        self._fit_onset = fit_onset
        self._max_shift = max_shift
        
        # Construct template fits for each channel separately
        self._template_fits = []

        for i in range(self._n_channels):
            if self._order[i] is None:
                self._template_fits.append(
                    _TemplateCacheSimple(
                        sev=self._sev[i],
                        fit_onset=self._fit_onset[i],
                        max_shift=self._max_shift
                    )
                )
            else:
                self._template_fits.append(
                    _TemplateCachePoly(
                        sev=self._sev[i],
                        xdata=self._xdata,
                        order=self._order[i],
                        fit_onset=self._fit_onset[i],
                        max_shift=self._max_shift
                    )
                )
                
        # Determine output shape (pad with zeros if channels have different baseline models)
        # [[ph_ch0, poly0_ch0, poly1_ch0, poly2_ch0, poly3_ch0]
        #  [ph_ch1, poly0_ch1, poly1_ch1, poly2_ch1, poly3_ch1]]
        non_none_orders = [x for x in self._order if x is not None]
        max_order = np.max(non_none_orders) if len(non_none_orders)>0 else -1
        self._output_shape = (self._n_channels, 2+max_order)
        
        # Define error outputs (if fit fails, these will be the
        # fit results that tell you that the fit failed)
        self._error_output = (np.zeros(self._output_shape), 0, -404*np.ones(self._n_channels))

    def __call__(self, ev: np.ndarray, flag: np.ndarray = None):
        """
        Performs the correlated template fit including a (non-)trivial baseline.
        See https://edoc.ub.uni-muenchen.de/23762/ and https://mediatum.ub.tum.de/?id=1294132 for details.

        :param ev: The event to be fitted (at least 2 channels)
        :type ev: np.ndarray
        :param flag: The flag (for each channel) to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing on either channel
        :type flag: np.ndarray, optional
        
        :return: Tuple of fit result, optimal shift, and RMS value ``([[amplitude_ch0, constant_bl_coeff_ch0, linear_bl_coeff_ch0, ...], [amplitude_ch1, constant_bl_coeff_ch1, linear_bl_coeff_ch1, ...], ...], shift, [rms_ch0, rms_ch1, ...])``. If the fit was unsuccessful, the RMS values are set to -404 and all fit parameters are 0. If different orders of baseline polynomials are used for different channels, the output is extended to the largest used polynomial order (i.e. unused coefficients are 0).
        :rtype: Tuple[np.ndarray, int, np.ndarray]
        """
        flag = [None]*self._n_channels if flag is None else flag
        
        try:
            if any(self._fit_onset):
                res = minimize(self._chij2, 
                            x0=0, 
                            args=(ev, flag), 
                            method="Powell", 
                            bounds=[(-self._max_shift, self._max_shift)])
                opt_shift = int(res.x[0])
            else:
                opt_shift = 0

            results = self._solve(j=opt_shift, ev=ev, flag=flag)
            
            opt_param = np.zeros(self._output_shape)
            rms = np.zeros(self._n_channels)
            
            for i, result in enumerate(results):
                op, r = result
                rms[i] = r
                # make sure it's a 1d array
                op = np.array(op)[None].flatten()
                opt_param[i, :len(op)] = op[:len(op)]

        except LinAlgError:
            opt_param, opt_shift, rms = self._error_output
        except ValueError as err:
            if err.args[0] == "array must not contain infs or NaNs":
                opt_param, opt_shift, rms = self._error_output
            elif err.args[0] == "zero-size array to reduction operation maximum which has no identity":
                opt_param, opt_shift, rms = self._error_output
            else:
                raise err
        
        return opt_param, opt_shift, rms
    
    def _solve(self, j: int, ev: np.ndarray, flag: list):
        """
        Solve the minimization problem for a given shift value j.

        :param j: The shift.
        :type j: int
        :param ev: The event to be fitted.
        :type ev: np.ndarray
        :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
        :type flag: np.ndarray, optional
        
        :return: Tuple of (fit parameters, rms).
        :rtype: Tuple[float, float]
        """
        # call the solvers of the underlying classes and return results 
        # as a list where each entry is a tuple that corresponds to 
        # (fitresult, rms) of the individual channels
        return [
            tf._solve(j=j, ev=ev[i], flag=flag[i]) 
            for i, tf in enumerate(self._template_fits)
        ]
    
    def _chij2(self, j: int, ev: np.ndarray, flag: list):
        """
        Returns the chi-squared value for a given shift after fitting ``sev`` to ``ev``. The result includes the sum of the chi-squares of all channels whose onset is to be fitted (i.e. chi-squared is minimized together if multiple channels use fit_onset=True).
        See https://edoc.ub.uni-muenchen.de/23762/ and https://mediatum.ub.tum.de/?id=1294132 for details.

        :param j: The shift.
        :type j: int
        :param ev: The event to be fitted.
        :type ev: np.ndarray
        :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
        :type flag: np.ndarray, optional
        
        :return: The chi-squared value.
        :rtype: float
        """
        # call the chij2 functions of the underlying classes and sum
        # the ones for which the onset should be fitted. Returns the
        # sum of the RMSs
        return sum([ 
            tf._chij2(j=j, ev=ev[i], flag=flag[i]) 
            for i, tf in enumerate(self._template_fits)
            if self._fit_onset[i]
        ])
    
########################
### CLASS DEFINITION ###
########################
class TemplateFitCorrelated(FncBaseClass):
    """
    Perform a correlated template fit for multi-channel data, i.e. fit a numeric SEV to data with possibility to also specify a polynomial baseline model (for each channel individually) and a truncation limit (for each channel individually).
    The 'correlated' in this context means that you can choose which of the channel's onset should be fitted (possibly multiple, see below).
    See https://edoc.ub.uni-muenchen.de/23762/ and https://mediatum.ub.tum.de/?id=1294132 for details.

    :param sev: The template (SEV) to use in the fit (at least two channels).
    :type sev: np.ndarray
    :param bl_poly_order: List of the baseline models to use in the fit (one entry for each channel). Has to be a non-zero integer or None. If 0, a constant offset is fitted, if 1, a linear baseline is assumed, etc. If None, the baseline is assumed to be constantly 0 (here, it's the users responsibility to remove the baseline accordingly), defaults to 0, i.e. fitting a constant offset for all channels.
    :type bl_poly_order: List[int]
    :param truncation_limit: List with as many entries as there are channels. For each entry that is not None, a truncated fit is performed: all samples between the first and the last sample above 'truncation_limit' are ignored in the fit. To determine these samples, the baseline of the event is removed by fitting a polynomial of order 'bl_poly_order' to the beginning of the record window. Defaults to None, i.e. not performing a truncated fit in any channel.
    :type truncation_limit: List[float]
    :param xdata: The x-data array used to evaluate the baseline model (1-dimensional). If None, the default ``xdata=np.linspace(0, 1, sev.shape[-1])`` is used, defaults to None.
    :type xdata: np.ndarray
    :param fit_onset: List with as many entries as there are channels. For each entry that is True, the onset value of the respective channel is fitted. If False, the channel does not participate in the onset fit. If only one of the entries is True, this channel is the 'dominant' one, i.e. its onset is fitted and all other channels are moved (passively) in unison. If multiple are True, their chi-squared for the fit is combined, i.e. the template is still moved in unison, but the minimizer considers all channels. Defaults to True, i.e. fit onset in all channels
    :type fit_onset: List[bool]
    :param max_shift: The maximum shift value (in samples) to search for a minimum. The onset fit will search the minimum for shifts in ``(-max_shift, +max_shift)``.
    :type max_shift: int
    
    :return: Tuple of fit result, optimal shift, and RMS value ``([[amplitude_ch0, constant_bl_coeff_ch0, linear_bl_coeff_ch0, ...], [amplitude_ch1, constant_bl_coeff_ch1, linear_bl_coeff_ch1, ...]], shift, [rms_ch0, rms_ch1, ...])``. If you set ``fit_onset=False``, the ``shift`` value will just be 0. If the fit fails, all fit parameters are set to 0 and the RMS value is set to -404. If different orders of baseline polynomials are used for different channels, the output is extended to the largest used polynomial order (i.e. unused coefficients are 0).
    :rtype: Tuple[np.ndarray, int, np.ndarray]

    **Example:**

    .. code-block:: python

        import numpy as np
        import cait.versatile as vai

        # Get events and SEV from mock data (and select first channel)
        # Also add an artificial falling baseline to the events
        md = vai.MockData()
        it = md.get_event_iterator().with_processing(lambda x: x-np.linspace(0, 1, x.shape[-1]))
        sev = md.sev

        # Specify fit
        f = vai.TemplateFitCorrelated(sev=sev, bl_poly_order=[1, 3], fit_onset=[True, False])

        # Preview the working of the fit
        vai.Preview(it, f)

        # Fit all events in iterator
        fitpar, opt_shift, rms = vai.apply(f, it)
        # Plot fit amplitudes
        vai.Histogram({"channel 0": fitpar[:, 0, 0], "channel 1": fitpar[:, 1, 0]}, xlabel="Pulse Height")

    .. image:: media/TemplateFitCorrelated.png
    """
    def __init__(self, 
                 sev: np.ndarray,
                 bl_poly_order: List[int] = 0,
                 truncation_limit: List[float] = None,
                 xdata: List[float] = None,
                 fit_onset: List[bool] = True,
                 max_shift: int = 50
                 ):
        
        if not (np.array(sev).ndim == 2):
            raise ValueError(f"{self.__class__.__name__} can only process multi-channel data. Single-dimensional templates are not supported. Use TemplateFit for single-channel data.")
            
        if not (np.array(sev).shape[0] > 1):
            raise ValueError(f"'sev' has to have at least 2 channels. Got {sev.shape[0]}.")
            
        if xdata is not None and (np.array(xdata).ndim > 1):
            raise ValueError(f"'xdata' has to be none or 1-dimensional.")
            
        self._sev = np.array(sev)
        self._n_channels = self._sev.shape[0]
        self._xdata = np.linspace(0, 1, self._sev.shape[-1]) if xdata is None else np.array(xdata)
        self._bl_poly_order = [bl_poly_order]*self._n_channels if (isinstance(bl_poly_order, int) or bl_poly_order is None) else bl_poly_order
        self._truncation_limit = [truncation_limit]*self._n_channels if (isinstance(truncation_limit, int) or truncation_limit is None) else truncation_limit
        self._fit_onset = [fit_onset]*self._n_channels if isinstance(fit_onset, bool) else fit_onset
        self._max_shift = max_shift
        
        if not (len(self._bl_poly_order) == self._n_channels):
            raise ValueError(f"'bl_poly_order' has to be None, an integer, or a list/tuple the same length as there are channels in 'sev'. Got {len(self._bl_poly_order)} and {self._n_channels}.")
        
        if not (len(self._fit_onset) == self._n_channels):
            raise ValueError(f"'fit_onset' has to be a bool, or a list/tuple the same length as there are channels in 'sev'. Got {len(self._fit_onset)} and {self._n_channels}.")
        
        self._rm_bl = []
        for order in self._bl_poly_order:
            if order is None:
                self._rm_bl.append(RemoveBaseline(dict(model=0, where=1/8, xdata=None)))
            else:
                self._rm_bl.append(RemoveBaseline(dict(model=1, where=1/8, xdata=None)))
        
        self._solver = _TemplateCacheCorrelated(sev=self._sev, 
                                                xdata=self._xdata, 
                                                order=self._bl_poly_order, 
                                                fit_onset=self._fit_onset, 
                                                max_shift=self._max_shift)

    def __call__(self, event):
        if event.shape != self._sev.shape: 
            raise ValueError(f"{self.__class__.__name__} can only process events which have the same shape as the specified template.")
        
        below_truncation_limit = [None]*self._n_channels
        for i, trunc_lv in enumerate(self._truncation_limit):
            if trunc_lv is not None:
                flag = (self._rm_bl[i](event[i]) > trunc_lv).flatten()
                if any(flag):
                    start, end = np.argmax(flag), event[i].shape[-1] - np.argmax(flag[::-1]) - 1
                    below_truncation_limit[i] = np.ones_like(flag)
                    below_truncation_limit[i][start:end] = False
                
        return self._solver(event, flag=below_truncation_limit)
    
    @property
    def batch_support(self):
        return 'none'
    
    def preview(self, event):
        fitpars, shift, _ = self(event)

        shifted_sev, shifted_x = shift_arrays(self._sev, self._xdata, j=shift)
        shifted_x = np.array([shifted_x]*self._n_channels)

        fit_sev = fitpars[:,0][:,None]*shifted_sev + np.sum([
            fitpars[:,k+1][:,None]*shifted_x**k 
            for k in range(fitpars.shape[-1]-1)
            ], axis=0)
        
        d = dict()
        for i in range(self._n_channels):
            d[f"channel {i}"] = [self._xdata, event[i]]
            d[f"fit channel {i}"] = [shifted_x[i], fit_sev[i]]

        for i, trunc_lv in enumerate(self._truncation_limit):
            if trunc_lv is not None:
                truncation_line = trunc_lv + event[i] - self._rm_bl[i](event[i])
                d[f"trunc. lim. channel {i}"] = [self._xdata, truncation_line]

        return dict(line=d)