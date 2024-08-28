from typing import Union, List
#import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve

from ..functionbase import FncBaseClass
from ..processing.removebaseline import RemoveBaseline

# We do not use scipy's parameter error estimation anyways so we can suppress this warning
# warnings.filterwarnings("ignore", "Covariance of the parameters could not be estimated")

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

### SOLVE SIMPLIFIED PROBLEM ###
def simple_sol(s: np.ndarray, y: np.ndarray):
    """
    Solves for the amplitude in the simple model where no baseline fit is performed.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    :param s: The (shifted) reference event.
    :type s: np.ndarray
    :param y: The (shifted) event to be fitted.
    :type y: np.ndarray
    
    :return: The fit amplitude for ``y``.
    :rtype: float
    """
    return np.sum(y*s)/np.sum(s**2)

### SOLVE SYSTEM OF EQUATIONS ###
def A(s: np.ndarray, x: np.ndarray, order: int):
    """
    Returns the matrix for the linear system of equations needed for the template fit including a non-trivial baseline.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    :param s: The (shifted) reference event.
    :type s: np.ndarray
    :param x: The (shifted) baseline x-data.
    :type x: np.ndarray
    :param order: The order of the polynomial used to fit the baseline.
    :type order: int
    
    :return: The matrix for the linear system of equations.
    :rtype: np.ndarray
    """
    return np.concatenate([
        np.array([ [np.mean(s**2)] + [np.mean(s*x**k) for k in range(order+1)] ]),
        np.array([ 
            [np.mean(s*x**l)] + [np.mean(x**(k+l)) for k in range(order+1)]
            for l in range(order+1)
        ])
    ])

def b(s: np.ndarray, y: np.ndarray, x: np.ndarray, order: int):
    """
    Returns right hand side for the linear system of equations needed for the template fit including a non-trivial baseline.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    :param s: The (shifted) reference event.
    :type s: np.ndarray
    :param y: The (shifted) event to be fitted.
    :type y: np.ndarray
    :param x: The (shifted) baseline x-data.
    :type x: np.ndarray
    :param order: The order of the polynomial used to fit the baseline.
    :type order: int
    
    :return: The right hand side for the linear system of equations.
    :rtype: np.ndarray
    """
    return np.array([np.mean(y*s)] + [np.mean(y*x**k) for k in range(order+1)])

def poly_sol(s: np.ndarray, y: np.ndarray, x: np.ndarray, order: int):
    """
    Solves the linear system of equations needed for the template fit including a non-trivial baseline.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    :param s: The (shifted) reference event.
    :type s: np.ndarray
    :param y: The (shifted) event to be fitted.
    :type y: np.ndarray
    :param x: The (shifted) baseline x-data.
    :type x: np.ndarray
    :param order: The order of the polynomial used to fit the baseline.
    :type order: int
    
    :return: The solution of the linear system of equations (``[fit_amplitude, constant_bl_coeff, linear_bl_coeff, ...]``).
    :rtype: np.ndarray
    """
    return solve(A(s, x, order), b(s, y, x, order), assume_a="sym")

### CHI SQUARED EQUATIONS FOR ONSET FIT ###
def chij2_simple(j: int, sev: np.ndarray, ev: np.ndarray, flag: np.ndarray = None):
    """
    Returns the chi-squared value for a given shift after fitting ``sev`` to ``ev`` in the simplified model.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    :param j: The shift.
    :type j: int
    :param sev: The reference event.
    :type sev: np.ndarray
    :param ev: The event to be fitted.
    :type ev: np.ndarray
    :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
    :type flag: np.ndarray, optional
    
    :return: The chi-squared value.
    :rtype: float
    """
    s, y = shift_arrays(sev, ev, j=int(np.round(j)), flag=flag)

    return np.mean( (y - simple_sol(s, y)*s)**2 )

def chij2_poly(j: int, 
               sev: np.ndarray, 
               ev: np.ndarray, 
               xdata: np.ndarray, 
               order: int,
               flag: np.ndarray = None):
    """
    Returns the chi-squared value for a given shift after fitting ``sev`` to ``ev`` including a non-trivial baseline.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    :param j: The shift.
    :type j: int
    :param sev: The reference event.
    :type sev: np.ndarray
    :param ev: The event to be fitted.
    :type ev: np.ndarray
    :param xdata: The baseline x-data.
    :type xdata: np.ndarray
    :param order: The order of the polynomial for the baseline fit (0, 1, 2, ...).
    :type order: int
    :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
    :type flag: np.ndarray, optional
    
    :return: The chi-squared value.
    :rtype: float
    """
    s, y, x = shift_arrays(sev, ev, xdata, j=int(np.round(j)), flag=flag)
        
    sol = poly_sol(s, y, x, order)
    return np.mean((y - sol[0]*s - np.sum([sol[k+1]*x**k for k in range(order+1)], axis=0))**2)

### WRAPPERS ###
def template_fit_simple(ev: np.ndarray, 
                        sev: np.ndarray,
                        flag: np.ndarray = None,
                        fit_onset: bool = True, 
                        max_shift: int = 50):
    """
    Performs the template fit in the simplified model where no baseline fit is performed.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    :param ev: The event to be fitted
    :type ev: np.ndarray
    :param sev: The reference event.
    :type sev: np.ndarray
    :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
    :type flag: np.ndarray, optional
    :param fit_onset: If True, the onset value is fitted. If False, the event is fitted as is, defaults to True
    :type fit_onset: bool, optional
    :param max_shift: The maximum shift value (in samples) to search for a minimum. The onset fit will search the minimum for shifts in ``(-max_shift, +max_shift)``. Defaults to 50 samples
    :type max_shift: int, optional
    
    :return: Tuple of fit result, optimal shift, and RMS value ``(amplitude, shift, rms)``.
    :rtype: Tuple[float, int, float]
    """
    if fit_onset:
        res = minimize(chij2_simple, 
                       x0=0, 
                       args=(sev, ev, flag), 
                       method="Powell", 
                       bounds=[(-max_shift, max_shift)])
        shift = int(res.x)
    else:
        shift = 0
    
    return simple_sol(*shift_arrays(sev, ev, j=shift, flag=flag)), shift, np.sqrt(chij2_simple(shift, sev, ev, flag))

def template_fit_poly(ev: np.ndarray, 
                      sev: np.ndarray, 
                      order: int, 
                      xdata: np.ndarray = None, 
                      flag: np.ndarray = None,
                      fit_onset: bool = True,
                      max_shift: int = 50):
    """
    Performs the template fit including a non-trivial baseline.
    See https://edoc.ub.uni-muenchen.de/23762/ for details.

    :param ev: The event to be fitted
    :type ev: np.ndarray
    :param sev: The reference event.
    :type sev: np.ndarray
    :param order: The order of the polynomial for the baseline fit (0, 1, 2, ...).
    :type order: int
    :param xdata: The x-data to use for the baseline model evaluation. If None, the default ``xdata=np.linspace(0, 1, len(sev))`` is used, defaults to None.
    :type xdata: np.ndarray, optional
    :param flag: The flag to apply to the data (used for truncated fit). Defaults to None, i.e. no slicing
    :type flag: np.ndarray, optional
    :param fit_onset: If True, the onset value is fitted. If False, the event is fitted as is, defaults to True
    :type fit_onset: bool, optional
    :param max_shift: The maximum shift value (in samples) to search for a minimum. The onset fit will search the minimum for shifts in ``(-max_shift, +max_shift)``. Defaults to 50 samples
    :type max_shift: int, optional
    
    :return: Tuple of fit result, optimal shift, and RMS value ``([amplitude, constant_bl_coeff, linear_bl_coeff, ...], shift, rms)``.
    :rtype: Tuple[np.ndarray, int, float]
    """
    if xdata is None: xdata = np.linspace(0, 1, np.array(sev).shape[-1])
    
    if fit_onset:
        res = minimize(chij2_poly, 
                       x0=0, 
                       args=(sev, ev, xdata, order, flag), 
                       method="Powell", 
                       bounds=[(-max_shift, max_shift)])
        shift = int(res.x)
    else:
        shift = 0
    
    return  ( poly_sol(*shift_arrays(sev, ev, xdata, j=shift, flag=flag), order), shift,
              np.sqrt(chij2_poly(shift, sev, ev, xdata, order, flag)) )

########################
### CLASS DEFINITION ###
########################
class TemplateFit(FncBaseClass):
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
    
    :return: Tuple of fit result, optimal shift, and RMS value ``([amplitude, constant_bl_coeff, linear_bl_coeff, ...], shift, rms)``. If you set ``fit_onset=False``, the ``shift`` value will just be 0.
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
                 max_shift: int = 50
                 ):
        if np.array(sev).ndim>1:
            raise ValueError(f"{self.__class__.__name__} can only process single-channel data. Multi-dimensional templates are not supported.")
        if not (isinstance(bl_poly_order, int) or bl_poly_order is None):
            raise TypeError(f"'bl_poly_order' has to be a non-zero integer or None, not {type(bl_poly_order)}.")
        elif isinstance(bl_poly_order, int) and bl_poly_order<0:
            raise TypeError(f"'bl_poly_order' has to be a non-negative integer, not {bl_poly_order}.")
        
        self._sev = np.array(sev)
        self._xdata = np.linspace(0, 1, self._sev.shape[-1]) if xdata is None else xdata
        self._truncation_limit = truncation_limit

        if bl_poly_order is None:
            self._mode = 'simple'
            self._solver = template_fit_simple
            self._solver_args = dict(sev=self._sev, fit_onset=fit_onset, max_shift=max_shift)
            self._rm_bl = RemoveBaseline(dict(model=0, where=1/8, xdata=None))
        else:
            self._mode = 'poly'
            self._solver = template_fit_poly
            self._solver_args = dict(sev=self._sev, order=bl_poly_order, xdata=self._xdata,
                                     fit_onset=fit_onset, max_shift=max_shift)
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
                
        return self._solver(event, **self._solver_args, flag=below_truncation_limit)
    
    @property
    def batch_support(self):
        return 'none'
    
    def preview(self, event):
        fitpars, shift, rms = self(event)

        shifted_sev, shifted_x = shift_arrays(self._sev, self._xdata, j=shift)

        if self._mode == 'simple':
            fit_sev = fitpars*shifted_sev
        else:
            fit_sev = fitpars[0]*shifted_sev + np.sum(
                [fitpars[k+1]*shifted_x**k for k in range(len(fitpars)-1)],
                axis=0)
        
        d = {"event": [self._xdata, event], "template fit": [shifted_x, fit_sev]}
        if self._truncation_limit is not None:
            truncation_line = self._truncation_limit + event - self._rm_bl(event)
            d["truncation limit"] = [self._xdata, truncation_line]

        return dict(line=d)