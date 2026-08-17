from typing import List

import numpy as np
from scipy.optimize import minimize, lsq_linear
from scipy.linalg import solve, LinAlgError

from ..functionbase import ScalarFncBaseclass
from ..processing.removebaseline import RemoveBaseline

import warnings
warnings.filterwarnings('ignore', r'Ill-conditioned matrix')

########################
### HELPER FUNCTIONS ###
########################

### SHIFTING ARRAYS FOR ONSET FIT ###
def shift_arrays(sev: np.ndarray, 
                 *others: np.ndarray, 
                 j: int = 0, 
                 flag: np.ndarray = None):
    """
    Shifts arrays and a reference SEV against each other by a number of samples such that the resulting arrays (when layed on top of each other) give the impression that the SEV was shifted by ``j`` samples relative to the other arrays.
    
    :param sev: The reference event which is shifted.
    :type sev: np.ndarray
    :param others: An arbitrary number of arrays against which ``sev`` is shifted.
    :type others: np.ndarray
    :param j: The number of samples by which to shift the arrays. Defaults to 0
    :type j: int, optional
    :param flag: If provided, the shifted arrays are also sliced according to the flag (has to have the same size as 'sev' and all other arrays). This is used to only consider a subset of data points in a truncated fit. Defaults to None, i.e. no slicing
    :type flag: np.ndarray, optional
    
    :return: The shifted (and if flag was given, also sliced) SEV and all other arrays as a tuple. The length of the arrays is ``original_length-j``, also possibly reduced by the flag.
    :rtype: Tuple[np.ndarray]
    """
    if j==0:  out = (sev, *others)
    elif j>0: out = (sev[..., :-j], *(data[..., j:] for data in others))
    else:     out = (sev[..., -j:], *(data[..., :j] for data in others))

    if flag is not None:
        if j==0:  f = flag
        elif j>0: f = flag[..., j:]
        else:     f = flag[..., :j]
        out = (elem[..., f] for elem in out)
    
    return out

############################
### HELPER CACHE CLASSES ###
############################

class _TemplateCacheSimple:
    """
    Performs the template fit in the simplified model where no baseline fit is performed.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    Helper class that performs the template fit in the simplified model where no baseline fit is performed. It caches parts of intermediate solutions for the minimization problem for the regular (not truncated) fit to increase computational efficiency.

    :param sev: The reference event.
    :type sev: np.ndarray
    :param fit_onset: If True, the onset value is fitted. If False, the event is fitted as is, defaults to True
    :type fit_onset: bool, optional
    :param max_shift: The maximum shift value (in samples) to search for a minimum. The onset fit will search the minimum for shifts in ``(-max_shift, +max_shift)``. Defaults to 50 samples
    :type max_shift: int, optional
    """
    def __init__(self, sev: np.ndarray, fit_onset: bool = True, max_shift: int = 50):
        self._sev = sev
        self._fit_onset = fit_onset
        self._max_shift = max_shift
        
        # Precalculate and cache for later access
        # This is the only strategy that worked with multiprocessing
        self._cache = dict()
        for i in range(-max_shift, max_shift+1):
            self._cache[i] = self._norm2_uncached(i, flag=None)

    def __call__(self, ev: np.ndarray, flag: np.ndarray = None):
        """
        Performs the template fit in the simplified model where no baseline fit is performed.
        See https://edoc.ub.uni-muenchen.de/23762/ for details.

        :param ev: The event to be fitted
        :type ev: np.ndarray
        :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
        :type flag: np.ndarray, optional
        
        :return: Tuple of fit result, optimal shift, and RMS value ``(amplitude, shift, rms)``. If the fit was unsuccessful, the RMS value is set to -404.
        :rtype: Tuple[float, int, float]
        """
        if self._fit_onset:
            res = minimize(self._chij2, 
                        x0=0, 
                        args=(ev, flag), 
                        method="Powell", 
                        bounds=[(-self._max_shift, self._max_shift)])
            opt_shift = int(res.x[0])
        else:
            opt_shift = 0
            
        opt_param, rms = self._solve(j=opt_shift, ev=ev, flag=flag)
        
        return opt_param, opt_shift, rms
    
    ### GIVEN A SHIFT, SOLVE EQUATION ###
    def _solve(self, j: int, ev: np.ndarray, flag: np.ndarray = None):
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
        s, y = shift_arrays(self._sev, ev, j=j, flag=flag)
        opt_param = np.sum(y*s)/self._norm2(j=j, flag=flag)
        rms = np.sqrt(self._chij2(j, ev, flag))
        
        return np.atleast_1d(opt_param), rms
        
    ### CHI SQUARED EQUATIONS FOR ONSET FIT ###
    def _chij2(self, j: int, ev: np.ndarray, flag: np.ndarray = None):
        """
        Returns the chi-squared value for a given shift after fitting ``sev`` to ``ev`` in the simplified model.
        See https://edoc.ub.uni-muenchen.de/23762/ for details.

        :param j: The shift.
        :type j: int
        :param ev: The event to be fitted.
        :type ev: np.ndarray
        :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
        :type flag: np.ndarray, optional
        
        :return: The chi-squared value.
        :rtype: float
        """
        # this ensures that it works for both integers as well as the
        # minimizer's output (length-1 array)
        j = int(np.round(np.array(j)[None].flatten()[0]))
        s, y = shift_arrays(self._sev, ev, j=j, flag=flag)

        return np.mean( ( y - np.sum(y*s)/self._norm2(j=j, flag=flag)*s )**2 )

    def _norm2(self, j: int, flag: np.ndarray = None):
        """
        Returns the squared norm of the SEV for a given shift value.

        :param j: The shift value.
        :type j: int
        :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
        :type flag: np.ndarray, optional

        :return: The squared norm of the (shifted and/or truncated) SEV.
        :rtype: float
        """
        return self._norm2_cached(j) if flag is None else self._norm2_uncached(j, flag)
    
    def _norm2_uncached(self, j: int, flag: np.ndarray):
        return np.sum( list(shift_arrays(self._sev, j=j, flag=flag))[0]**2 )
    
    def _norm2_cached(self, j: int):
        return self._cache[j]
    
class _TemplateCachePoly:
    """
    Helper class that performs the template fit including a non-trivial baseline. It caches the matrices used for solving the minimization problem for the regular (not truncated) fit to increase computational efficiency.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    :param sev: The reference event.
    :type sev: np.ndarray
    :param xdata: The x-data to use for the baseline model evaluation.
    :type xdata: np.ndarray
    :param order: The order of the baseline polynomial to be fitted.
    :type order: int
    :param fit_onset: If True, the onset value is fitted. If False, the event is fitted as is, defaults to True
    :type fit_onset: bool, optional
    :param max_shift: The maximum shift value (in samples) to search for a minimum. The onset fit will search the minimum for shifts in ``(-max_shift, +max_shift)``. Defaults to 50 samples
    :type max_shift: int, optional
    :param baseline_type: Baseline model type: 'polynomial', 'exponential', or 'polyexp' (polynomial+exponential). Default to 'polynomial'.
    :type baseline_type: str
    :param exp_tau: Exponential decay rate τ. Units are reciprocal to xdata units. Required for 'exponential' or 'polyexp' baselines (default to 80).
    :type exp_tau: float
    """
    def __init__(self, 
                 sev: np.ndarray, 
                 xdata: np.ndarray, 
                 order: int, 
                 fit_onset: bool = True,
                 max_shift: int = 50,
                 baseline_type: str = 'polynomial',
                 exp_tau: float = None
                ):
        self._sev = sev
        self._xdata = xdata
        self._order = order
        self._fit_onset = fit_onset
        self._max_shift = max_shift
        self._baseline_type = baseline_type
        self._exp_tau = exp_tau
        #self._sev_dt_us = sev._dt_us
        
        # Precalculate and cache for later access
        # This is the only strategy that worked with multiprocessing
        self._cache = dict()
        for i in range(-max_shift, max_shift+1):
            self._cache[i] = self._A_uncached(i, flag=None)
            
        # Define error outputs (if fit fails, these will be the
        # fit results that tell you that the fit failed)
        # number of parameters to return on failure (ampl + poly coeffs + optional exp coeff)

        if self._baseline_type == 'polynomial':
            self._n_params = self._order + 2        # [a, c0..c_order]
        elif self._baseline_type == 'polyexp':
            self._n_params = self._order + 3        # [a, c0..c_order, b_exp]
        else:
            raise ValueError("Unsupported baseline_type.")
        
        self._error_output = (np.zeros(self._n_params), 0, -404)

    def __call__(self, ev: np.ndarray, flag: np.ndarray = None):
        """
        Performs the template fit including a non-trivial baseline.
        See https://edoc.ub.uni-muenchen.de/23762/ for details.

        :param ev: The event to be fitted
        :type ev: np.ndarray
        :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
        :type flag: np.ndarray, optional
        
        :return: Tuple of fit result, optimal shift, and RMS value ``([amplitude, constant_bl_coeff, linear_bl_coeff, ...], shift, rms)``. If the fit was unsuccessful, the RMS value is set to -404.
        :rtype: Tuple[np.ndarray, int, float]
        """
        try:
            if self._fit_onset:
                res = minimize(self._chij2, 
                            x0=0, 
                            args=(ev, flag), 
                            method="Powell", 
                            bounds=[(-self._max_shift, self._max_shift)])
                opt_shift = int(res.x[0])
            else:
                opt_shift = 0

            opt_param, rms = self._solve(j=opt_shift, ev=ev, flag=flag)

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
    
    def _solve(self, j: int, ev: np.ndarray, flag: np.ndarray = None):
        """
        Solve the minimization problem for a given shift value j.

        :param j: The shift.
        :type j: int
        :param ev: The event to be fitted.
        :type ev: np.ndarray
        :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
        :type flag: np.ndarray, optional
        
        :return: Tuple of (fit parameters, rms).
        :rtype: Tuple[np.ndarray, float]
        """
        

        #""" Code including lsq_linear to find best positive b. Downside: longer_runtime
        if self._baseline_type == 'polynomial':
            opt_param = solve(
                self._A(j, flag),
                self._b(j, ev, flag),
                assume_a="sym"
            )
            rms = np.sqrt(self._chij2(j, ev, flag))
        
            return opt_param, rms
        
        elif self._baseline_type == 'polyexp':

            s, y, x = shift_arrays(self._sev, ev, self._xdata, j=j, flag=flag)
            # Build design matrix X to apply bounds
            e = np.exp(-(1. / self._exp_tau) * x)
            X = np.column_stack([
                s,
                *[x**k for k in range(self._order + 1)],
                e
            ])

            # All parameters are unconstrained except b_exp >= 0
            lower_bounds = np.full(X.shape[1], -np.inf)
            upper_bounds = np.full(X.shape[1],  np.inf)

            lower_bounds[-1] = 0.0

            result = lsq_linear(
                X,
                y,
                bounds=(lower_bounds, upper_bounds)
            )

            opt_param = result.x

            fit = X @ opt_param
            rms = np.sqrt(np.mean((y - fit)**2))

            return opt_param, rms
        
        """
        opt_param = solve(
            self._A(j, flag),
            self._b(j, ev, flag),
            assume_a="sym"
        )
        rms = np.sqrt(self._chij2(j, ev, flag))
        
        return opt_param, rms
        """

    
    def _chij2(self, j: int, ev: np.ndarray, flag: np.ndarray = None):
        """
        Returns the chi-squared value for a given shift after fitting ``sev`` to ``ev`` including a non-trivial baseline.

        :param j: The shift.
        :type j: int
        :param ev: The event to be fitted.
        :type ev: np.ndarray
        :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
        :type flag: np.ndarray, optional
        
        :return: The chi-squared value.
        :rtype: float
        """
        # this ensures that it works for both integers as well as the
        # minimizer's output (length-1 array)
        j = int(np.round(np.array(j)[None].flatten()[0]))
        s, y, x = shift_arrays(self._sev, ev, self._xdata, j=j, flag=flag)

        if self._baseline_type == 'polynomial':
            sol = solve(self._A(j, flag), self._b(j, ev, flag), assume_a="sym")
            return np.mean((y - sol[0]*s - np.sum([sol[k+1]*x**k for k in range(self._order+1)], axis=0))**2)
                
        elif self._baseline_type == 'polyexp':
            # Build design matrix X to apply bounds
            e = np.exp(-(1. / self._exp_tau) * x)
            X = np.column_stack([
                s,
                *[x**k for k in range(self._order + 1)],
                e
            ])

            lower_bounds = np.full(X.shape[1], -np.inf)
            upper_bounds = np.full(X.shape[1],  np.inf)
            # b_exp >= 0
            lower_bounds[-1] = 0.0

            result = lsq_linear(
                X,
                y,
                bounds=(lower_bounds, upper_bounds)
            )

            sol = result.x
            fit = X @ sol

            return np.mean((y - fit)**2)
        """ Code for free b_exp
        elif self._baseline_type == 'polyexp':
            sol = solve(self._A(j, flag), self._b(j, ev, flag), assume_a="sym")
            poly = np.sum([sol[k+1]*x**k for k in range(self._order+1)], axis=0)
            b_exp = sol[self._order+2]
            e = np.exp(-(1./self._exp_tau) * x)
            fit = sol[0]*s + poly + b_exp*e
            return np.mean((y - fit)**2)
        """
        
        #""" Code including lsq_linear to find best positive b. Downside: longer_runtime
        

    def _A(self, j: int, flag: np.ndarray = None):
        """
        Returns the matrix for the linear system of equations needed for the template fit including a non-trivial baseline. If flag is None, cached values (if available) for a given shift value ``j`` are returned.

        :param j: The shift value.
        :type j: int
        :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
        :type flag: np.ndarray, optional
        
        :return: The matrix for the linear system of equations.
        :rtype: np.ndarray
        """
        return self._A_cached(j) if flag is None else self._A_uncached(j, flag)

    def _A_uncached(self, j: int, flag: np.ndarray):
        s, x = shift_arrays(self._sev, self._xdata, j=j, flag=flag)
        
        if self._baseline_type == 'polynomial':
            return np.concatenate([
                np.array([ [np.mean(s**2)] + [np.mean(s*x**k) for k in range(self._order+1)] ]),
                np.array([ 
                    [np.mean(s*x**l)] + [np.mean(x**(k+l)) for k in range(self._order+1)]
                    for l in range(self._order+1)
                ])
            ])
        
        elif self._baseline_type == 'polyexp':
            # size: (2 + order+1) x (2 + order+1)
            # order of unknowns: [a, c0..c_order, b_exp]
            e = np.exp(-(1./self._exp_tau) * x)
            K = self._order + 2 + 1  # a + (order+1 poly) + b_exp
            A = np.zeros((K, K), dtype=float)

            # indices
            ia = 0
            i_poly0 = 1
            i_exp = 1 + (self._order + 1)

            # <s, s>
            A[ia, ia] = np.mean(s*s)
            # <s, x^k>
            for k in range(self._order+1):
                A[ia, i_poly0+k] = A[i_poly0+k, ia] = np.mean(s * x**k)
                # <s, e>
            A[ia, i_exp] = A[i_exp, ia] = np.mean(s * e)

            # <x^l, x^k>
            for l in range(self._order+1):
                for k in range(self._order+1):
                    A[i_poly0+l, i_poly0+k] = np.mean(x**(k+l))
            # <x^l, e>
            for l in range(self._order+1):
                A[i_poly0+l, i_exp] = A[i_exp, i_poly0+l] = np.mean((x**l) * e)

            # <e, e>
            A[i_exp, i_exp] = np.mean(e*e)

            return A
        
    def _A_cached(self, j: int):
        return self._cache[j]
    
    def _b(self, j: int, ev: np.ndarray, flag: np.ndarray = None):
        """
        Returns right hand side for the linear system of equations needed for the template fit including a non-trivial baseline.

        :param j: The shift value.
        :type j: int
        :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
        :type flag: np.ndarray, optional

        :return: The right hand side for the linear system of equations.
        :rtype: np.ndarray
        """
        s, y, x = shift_arrays(self._sev, ev, self._xdata, j=j, flag=flag)
        
        if self._baseline_type == 'polynomial':
            return np.array([np.mean(y*s)] + [np.mean(y*x**k) for k in range(self._order+1)])
            
        elif self._baseline_type == 'polyexp':
            # [ <y,s>, <y,x^0>,...,<y,x^order>, <y,e> ]
            e = np.exp(-(1./self._exp_tau) * x)
            return np.array(
                [np.mean(y*s)] +
                [np.mean(y*x**k) for k in range(self._order+1)] +
                [np.mean(y*e)]
            )
########################
### CLASS DEFINITION ###
########################
class TemplateFit(ScalarFncBaseclass):
    """
    Perform a template fit for single-channel data, i.e. fit a numeric SEV to data with possibility to also specify a polynomial baseline model and a truncation limit.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    :param sev: The template (SEV) to use in the fit.
    :type sev: np.ndarray
    :param bl_poly_order: The baseline model to use in the fit. Has to be a non-zero integer or None. If 0, a constant offset is fitted, if 1, a linear baseline is assumed, etc. If None, the baseline is assumed to be constantly 0 (here, it's the users responsibility to remove the baseline accordingly), defaults to 0, i.e. fitting a constant offset.
    :type bl_poly_order: int
    :param truncation_limit: If not None, a truncated fit is performed: all samples between the first and the last sample above 'truncation_limit' are ignored in the fit. To determine these samples, the baseline of the event is removed by fitting a polynomial of order 'bl_poly_order' to the beginning of the record window. Defaults to None, i.e. not performing a truncated fit.
    :type truncation_limit: float
    :param xdata: The x-data array used to evaluate the baseline model. If None, the default ``xdata=np.linspace(0, 1, len(sev))`` is used, defaults to None.
    :type xdata: np.ndarray
    :param fit_onset: If True, the onset value is fitted. If False, the event is fitted as is, defaults to True
    :type fit_onset: bool
    :param max_shift: The maximum shift value (in samples) to search for a minimum. The onset fit will search the minimum for shifts in ``(-max_shift, +max_shift)``.
    :type max_shift: int
    :param exp_tau: If not None, baseline fit will include an exponential decay with decay time constant τ. Units are reciprocal to xdata units. 
    :type exp_tau: float
    
    :return: Tuple of fit result, optimal shift, and RMS value ``([amplitude, constant_bl_coeff, linear_bl_coeff, ...], shift, rms)``. If you set ``fit_onset=False``, the ``shift`` value will just be 0. If the fit fails, all fit parameters are set to 0 and the RMS value is set to -404.
    :rtype: Tuple[np.ndarray, int, float]

    **Example:**

    .. code-block:: python

        import numpy as np
        import cait.versatile as vai

        # Get events and SEV from mock data (and select first channel)
        # Also add an artificial falling baseline to the events
        md = vai.MockData()
        it = md.get_event_iterator()[0].with_processing(lambda x: x-np.linspace(0, 1, len(x)))
        sev = md.sev[0]

        # Specify fit
        f = vai.TemplateFit(sev=sev, bl_poly_order=1)

        # Preview the working of the fit
        vai.Preview(it, f)

        # Fit all events in iterator
        fitpar, opt_shift, rms = vai.apply(f, it)
        # Plot fit amplitudes
        vai.Histogram(fitpar[:, 0])

    .. image:: media/TemplateFit.png
    """
    def __init__(self, 
                 sev: np.ndarray,
                 bl_poly_order: int = 0,
                 truncation_limit: float = None,
                 xdata: List[float] = None,
                 fit_onset: bool = True,
                 max_shift: int = 50,
                 exp_tau: float = None
                ):
        if np.array(sev).ndim>1:
            raise ValueError(f"{self.__class__.__name__} can only process single-channel data. Multi-dimensional templates are not supported. For correlated template fits (multi-dimensional), use TemplateFitCorrelated.")
        if not (isinstance(bl_poly_order, int) or bl_poly_order is None):
            raise TypeError(f"'bl_poly_order' has to be a non-zero integer or None, not {type(bl_poly_order)}.")
        elif isinstance(bl_poly_order, int) and bl_poly_order<0:
            raise TypeError(f"'bl_poly_order' has to be a non-negative integer, not {bl_poly_order}.")
        
        self._sev = np.array(sev)
        self._xdata = np.linspace(0, 1, self._sev.shape[-1]) if xdata is None else xdata
        self._truncation_limit = truncation_limit
        #self._baseline_type = baseline_type
        self._exp_tau = exp_tau

        if self._exp_tau is None:
            self.baseline_type = 'polynomial'
            self._mode = 'poly'
        elif isinstance(self._exp_tau, float):
            self.baseline_type = 'polyexp'
            self._mode = 'polyexp'
        else:
            raise ValueError("Invalid baseline type")
        
        if bl_poly_order is None:
            self._mode = 'simple'
            self._solver = _TemplateCacheSimple(sev=self._sev, 
                                                fit_onset=fit_onset, 
                                                max_shift=max_shift)
            self._rm_bl = RemoveBaseline(dict(model=0, where=1/8, xdata=None))
        else:                                         
            self._solver = _TemplateCachePoly(sev=self._sev,
                                                xdata=self._xdata,
                                                order=bl_poly_order,   # poly order for the poly part
                                                fit_onset=fit_onset,
                                                max_shift=max_shift,
                                                baseline_type=self.baseline_type,
                                                exp_tau=self._exp_tau)

                
            
                 
        self._rm_bl = RemoveBaseline(dict(model=1, where=1/8, xdata=None))

    def __call__(self, event):
        if event.shape != self._sev.shape: 
            raise ValueError(f"{self.__class__.__name__} can only process events which have the same shape as the specified template.")
        
        below_truncation_limit = None
        if self._truncation_limit is not None:
            flag = (self._rm_bl(event) > self._truncation_limit).flatten()
            if any(flag):
                start, end = np.argmax(flag), event.shape[-1] - np.argmax(flag[::-1]) - 1
                below_truncation_limit = np.ones_like(flag)
                below_truncation_limit[start:end] = False
                
        return self._solver(event, flag=below_truncation_limit)
    
    @property
    def batch_support(self):
        return 'none'
    
    def preview(self, event):
        fitpars, shift, rms = self(event)
        print(fitpars)

        shifted_sev, shifted_x = shift_arrays(self._sev, self._xdata, j=shift)

        if self._mode == 'simple':
            fit_sev = fitpars*shifted_sev
        elif self._mode == 'poly':
            fit_sev = fitpars[0]*shifted_sev + np.sum(
                [fitpars[k+1]*shifted_x**k for k in range(len(fitpars)-1)],
                axis=0)
        elif self._mode == 'polyexp':
            a = fitpars[0]
            poly = np.sum([fitpars[k+1]*shifted_x**k for k in range(len(fitpars)-2)], axis=0)
            b_exp = fitpars[-1]
            decay = np.exp(-(1./self._solver._exp_tau) * shifted_x)
            baseline_poly = poly
            baseline_exp  = b_exp * decay
            baseline_total = baseline_poly + baseline_exp
            fit_sev = a*shifted_sev + baseline_poly + baseline_exp
                      
        
        d = {"event": [self._xdata, event], "template fit": [shifted_x, fit_sev]}
        
        if self._mode == 'polyexp':
            d["polynomial + exp baseline"] = [shifted_x, baseline_total]
            
        
        if self._truncation_limit is not None:
            truncation_line = self._truncation_limit + event - self._rm_bl(event)
            d["truncation limit"] = [self._xdata, truncation_line]

        return dict(line=d)
