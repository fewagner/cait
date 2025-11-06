from abc import ABC, abstractmethod

import numba
import numpy as np
import scipy as sp
from ipywidgets import widgets

from ...serialize import SerializingMixin
from ..plot import Viewer


##########################################
############ HELPER FUNCTIONS ############
##########################################
def find_root_secant(
    f: callable, # accepts and returns shape (N, M) arrays
    x0: np.ndarray, # shape (N, M)
    x1: np.ndarray, # shape (N, M)
    eps: float = 1e-4, 
    max_iter: int = 100,
):
    # Root finder based on the secant method (https://en.wikipedia.org/wiki/Secant_method)
    
    # Check if the initial guess is already a root and adjust the starting point
    # slightly (by 10% of the total (x0, x1) range) so that the algorithm doesn't get stuck.
    fx0, fx1 = f(x0), f(x1)
    solved_x0, solved_x1 = np.abs(fx0)<eps, np.abs(fx1)<eps
    x0[solved_x0] += (x1[solved_x0] - x0[solved_x0])/10
    x1[solved_x1] -= (x1[solved_x1] - x0[solved_x1])/10
    
    xn_m2, xn_m1 = x0, x1
    fn_m2, fn_m1 = f(xn_m2), f(xn_m1)

    for _ in range(max_iter):
        xn = xn_m1 - fn_m1*(xn_m1 - xn_m2)/(fn_m1 - fn_m2)
        fn = f(xn)

        converged = np.abs(fn) < eps
        if np.all(converged):
            break
        
        # Only update those which have not yet converged.
        # (To prevent numerical instabilities when continuing the
        # iteration for already converged values: 
        # Denominator becomes 0 when calculating xn, doesn't affect numerator)
        xn_m2, xn_m1 = xn_m1, xn
        fn_m2[~converged], fn_m1[~converged] = fn_m1[~converged], fn[~converged]

    xn[~converged] = np.nan
            
    return xn

@numba.njit
def _find_interval_ascending(x: np.ndarray,
                            xval: float,
                            extrapolate: bool = True):
    """
    Find an interval such that x[interval] <= xval < x[interval+1]. Assuming
    that x is sorted in the ascending order.
    If xval < x[0], then interval = 0, if xval > x[-1] then interval = n - 2.

    :param x: Piecewise polynomial breakpoints sorted in ascending order.
    :type x: np.ndarray, shape (M,)
    :param xval: Point to find.
    :type xval: float
    :param extrapolate: Whether to return the last of the first interval if the point is out-of-bounds.
    :type extrapolate: bool, optional

    :return: Suitable interval or -1 if NaN.
    :rtype: int
    """
    # Adapted from https://github.com/scipy/scipy/blob/v1.16.1/scipy/interpolate/_poly_common.pxi

    nx = x.shape[0]
    a, b = x[0], x[nx - 1]
    interval = 0

    if interval < 0 or interval >= nx:
        interval = 0

    if not (a <= xval <= b):
        # Out-of-bounds (or nan)
        if xval < a and extrapolate:
            # below
            interval = 0
        elif xval > b and extrapolate:
            # above
            interval = nx - 2
        else:
            # nan or no extrapolation
            interval = -1
    elif xval == b:
        # Make the interval closed from the right
        interval = nx - 2
    else:
        # Find the interval the coordinate is in
        # (binary search with locality)
        if xval >= x[interval]:
            low = interval
            high = nx - 2
        else:
            low = 0
            high = interval

        if xval < x[low+1]:
            high = low

        while low < high:
            mid = (high + low)//2
            if xval < x[mid]:
                # mid < high
                high = mid
            elif xval >= x[mid + 1]:
                low = mid + 1
            else:
                # x[mid] <= xval < x[mid+1]
                low = mid
                break

        interval = low

    return interval

@numba.njit(parallel=True)
def _ppolyval(c: np.ndarray,
              x: np.ndarray,
              xp: np.ndarray,
              extrapolate: bool,
             ):
    """
    Evaluate a piecewise polynomial.
    
    :param c: Coefficients local polynomials of order K-1 in L intervals. There are N polynomials in each interval. Coefficient of highest order-term comes first.
    :type c: np.ndarray, shape (K, L, N)
    :param x: Breakpoints of polynomials.
    :type x: np.ndarray, shape (L+1,)
    :param xp: Points to evaluate the piecewise polynomial at. M evaluations per polynomial.
    :type xp: np.ndarray, shape (N, M)
    :param extrapolate: Whether to extrapolate to out-of-bounds points based on first and last intervals, or to return NaNs.
    :type extrapolate: bool

    :return: Polynomials evaluated at ``xp``. 
    :rtype: np.ndarray, shape (N, M)
    """
    # Adapted from https://github.com/scipy/scipy/blob/v1.16.1/scipy/interpolate/_ppoly.pyx to evaluate the polynomial along the last dimension only.

    out = np.zeros(xp.shape)

    # For each polynomial (N)
    for i_poly in numba.prange(xp.shape[0]):
        # For each evaluation point of a fixed polynomial (M)
        for i_point in range(xp.shape[1]):
            xval = xp[i_poly, i_point]

            # Find correct interval
            i = _find_interval_ascending(x, xval, extrapolate)

            if i < 0:
                # xval was nan etc
                out[i_poly, i_point] = np.nan
                continue

            # Evaluate the local polynomial
            res = 0.0
            z = 1.0
            s = xval - x[i]

            for kp in range(c.shape[0]):
                res += c[c.shape[0] - kp - 1, i, i_poly] * z
                z *= s

            out[i_poly, i_point] = res
            
    return out

def _sanitize_inputs(arr1: np.ndarray, 
                     arr2: np.ndarray, 
                     arr3: np.ndarray, 
                     name1: str,
                     name2: str,
                     name3: str,
                     ):
    # Checks all input arrays for shape consistency and enforces shapes if necessary
    orig_shape_arr1, orig_shape_arr2, orig_shape_arr3 = np.shape(arr1), np.shape(arr2), np.shape(arr3)
    arr1 = np.atleast_1d(arr1).astype(float)
    arr2 = np.atleast_2d(arr2).astype(float)
    arr3 = np.atleast_2d(arr3).astype(float)

    if not np.ndim(arr1)==1: 
        raise ValueError(f"Array '{name1}' has to be 1d. Got shape {orig_shape_arr1}.")
    if np.ndim(arr2)>2: 
        raise ValueError(f"Array '{name2}' has to be at most 2d. Got shape {orig_shape_arr2}.")
    if np.ndim(arr3)>2: 
        raise ValueError(f"Array '{name3}' has to be at most 2d. Got shape {orig_shape_arr3}.")
    if arr1.shape[0] != arr2.shape[-1]:
        raise ValueError(f"The number of elements in '{name1}' must match the last dimension of '{name2}'. Need shapes (n_unique_tpas,) and (n_unique_tpas,) or (N, n_unique_tpas). Got shapes {orig_shape_arr1} and {orig_shape_arr2}.")
    if arr2.shape[0] != arr3.shape[0]:
        raise ValueError(f"The number of elements in the first dimension of '{name2}' must match the first dimension of '{name3}'. Need shapes (n_unique_tpas,) or (N, n_unique_tpas) and (M,) or (N, M). Got shapes {orig_shape_arr2} and {orig_shape_arr3}.")
    
    return (
        arr1, # shape (n_unique_tpas,)
        arr2, # shape (N, n_unique_tpas)
        arr3, # shape (N, M)
        )

##########################################
########## ABSTRACT BASE CLASS ###########
##########################################
class TransferFunction(SerializingMixin, ABC):
    """
    Abstract object describing the relation between pulse heights and testpulse equivalent pulse heights. This class is unaware of time, i.e. the interpolation/fit happens in the (TPE, PH)-plane only.

    To implement a specific model (e.g. piecewise cubic splines or polynomials), the following things need to be implemented:

    - :func:`TransferFunction.__init__`: Initialize the object with parameters that it needs (e.g. polynomial order) and call the constructor of the super class with those arguments.
    - :func:`TestpulseResponse.__call__`: see its docstring. Must perform input shape validation and raise a ValueError in case of shape mismatch (you can use the function '_sanitize_inputs' to perform those checks).
    - If the class attribute ``_PREVIEW_INPUTS`` is defined, it will be used for interactively changing the arguments passed to ``__init__`` in the preview methods. See below for how they must be structured. You can only allow a subset of the input arguments to be varied, but all field names in ``_PREVIEW_INPUTS`` must be input arguments to ``__init__``.

    This base class provides a default implementation for :func:`TestpulseResponse.inverse`, which calculates the inverse of the fit numerically using the Secant Method, given the function :func:`TestpulseResponse.__call__`. The class attribute ``_DEFAULT_SECANT_ROOT_FIND_ARGS`` stores the configuration details of this method. If you wish to implement your own inverse (e.g. because there is a more efficient or analytical way to do it) you can override :func:`TestpulseResponse.inverse` in the child class. Must perform input shape validation and raise a ValueError in case of shape mismatch (you can use the function '_sanitize_inputs' to perform those checks).

    The usage of this class is demonstrated below for the specific implementation of :class:`TFPchip`. See the docstrings of the child classes for more information on the specific behavior. 

    **Example:**

    .. code-block:: python
    
        import numpy as np
        import scipy as sp
        import cait.versatile as vai

        tpas = [1.0, 2.5, 4.0, 7.0]
        tp_phs_single = [0.8, 1.5, 1.9, 2.3]
        tp_phs_multi = [[0.8, 1.3, 1.9, 2.3], [0.7, 1.2, 1.8, 2.4]]
        tpes_single = np.linspace(0, 5, 100)
        tpes_multi = [np.linspace(0, 5, 100), np.linspace(1, 4, 100)]

        tf = vai.TFPchip()

        # Evaluate a single calibration (i.e. fit tpas and tp_phs)
        # and evaluate this calibration at tpes.
        # Shapes: tpas: (n_unique_tpas,)
        #         tp_phs: (n_unique_tpas,)
        #         tpes: (M,)
        #         output: (M,)
        ph_values_single = tf(tpas, tp_phs_single, tpes_single)

        # Evaluate a multiple calibrations at the same time (i.e. fit
        # tpas and rows of tp_phs SEPARATLY, then evaluate each row's
        # calibration with each row in tpes).
        # Shapes: tpas: (n_unique_tpas,)
        #         tp_phs: (N, n_unique_tpas)
        #         tpes: (N, M)
        #         output: (N, M)
        ph_values_multi = tf(tpas, tp_phs_multi, tpes_multi)

        # The object also has an inverse method:
        # Call tf.inverse() analogously to above to convert from
        # pulse height to TPE   

        # Preview what the transfer function does for different arguments.
        tf.preview(tpas, tp_phs_single)            

    .. automethod:: __call__
    """
    _PREVIEW_INPUTS = {}
    #{
    #    "arg1": {"dtype": bool, "default": False}, # Results in Checkbox
    #    "arg2": {"dtype": float, "default": 3.0, "domain": (1.0, 5.0)}, # Results in Slider
    #    "arg3": {"dtype": int, "default": 1, "domain": (0, 4)} # Results in Dropdown
    #}
    _DEFAULT_SECANT_ROOT_FIND_ARGS = {
        "eps": 1e-5,
        "max_iter": 100, 
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_kwargs = kwargs
        
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self._init_kwargs.items()])})"
    
    @abstractmethod
    def __call__(self, 
                 tpas: np.ndarray, 
                 tp_phs: np.ndarray, 
                 tpes: np.ndarray,
                 ):
        """
        Calculate pulse height values corresponding to testpulse-equivalent amplitudes ``tpes``, given the testpulse amplitudes ``tpas`` and corresponding testpulse pulse heights ``tp_phs`` that should be used for the fit.
        
        :param tpas: Testpulse amplitudes used for fit. Has to have as many elements as there are unique testpulse amplitudes.
        :type tpas: np.ndarray, shape (n_unique_tpas,)
        :param tp_phs: Testpulse pulse heights corresponding to ``tpas``. You can pass either as many as there are TPAs or a 2-d array with as many columns as there are TPAs. In the latter case, the fit is performed for each row. See also shape explanation of ``tp_phs``.
        :type tp_phs: np.ndarray, shape (n_unique_tpas,) or (N, n_unique_tpas)
        :param tpes: Testpulse-equivalent amplitudes for which to evaluate the calibration. If 2-d, the rows evaluate identical fits.
        :type tpes: np.ndarray, shape (M,) or (N, M)
        
        :return: Pulse heights for ``tpes``.
        :rtype: np.ndarray, same shape as tpes
        """
        ...

    def inverse(self, 
                tpas: np.ndarray, 
                tp_phs: np.ndarray,
                phs: np.ndarray,
                ):
        """
        Calculate testpulse-equivalent amplitudes corresponding to pulse heights ``phs``, given the testpulse amplitudes ``tpas`` and corresponding testpulse pulse heights ``tp_phs`` that should be used for the fit.
        
        :param tpas: Testpulse amplitudes used for fit. Has to have as many elements as there are unique testpulse amplitudes.
        :type tpas: np.ndarray, shape (n_unique_tpas,)
        :param tp_phs: Testpulse pulse heights corresponding to ``tpas``. You can pass either as many as there are TPAs or a 2-d array with as many columns as there are TPAs. In the latter case, the fit is performed for each row. See also shape explanation of ``tp_phs``.
        :type tp_phs: np.ndarray, shape (n_unique_tpas,) or (N, n_unique_tpas)
        :param phs: Pulse heights for which to evaluate the calibration. If 2-d, the rows evaluate identical fits.
        :type phs: np.ndarray, shape (M,) or (N, M)
        
        :return: Testpulse equivalent amplitudes for ``phs``.
        :rtype: np.ndarray, same shape as phs
        """
        in_shape = np.shape(phs)
        tpas, tp_phs, phs = _sanitize_inputs(tpas, tp_phs, phs, "tpas", "tp_phs", "phs")
        return np.reshape(
            find_root_secant(
                f=lambda x: self(tpas=tpas, tp_phs=tp_phs, tpes=x) - phs, 
                x0=np.zeros_like(phs), 
                x1=np.max(tpas)*np.ones_like(phs), 
                eps=self._DEFAULT_SECANT_ROOT_FIND_ARGS["eps"],
                max_iter=self._DEFAULT_SECANT_ROOT_FIND_ARGS["max_iter"],
            ),
        in_shape,
        )
        
    @classmethod
    def preview(cls, 
                tpas: np.ndarray, 
                tp_phs: np.ndarray, 
                n_grid_points: int = 100, 
                **viewer_kwargs,
                ):
        """
        Plot the testpulses defined by ``(tpas, tp_phs)`` and preview the fit result.
        
        :param tpas: Testpulse amplitudes used for the fit.
        :type tpas: np.ndarray, shape (N,)
        :param tp_phs: Pulse heights of testpulses used for the fit.
        :type tp_phs: np.ndarray, shape (N,)
        :param n_grid_points: Number of points used for plotting the fit.
        :type n_grid_points: int
        :param viewer_kwargs: Additional keyword arguments for :class:`cait.versatile.plot.viewer.Viewer`.
        :type viewer_kwargs: Any
        """
        if "backend" in viewer_kwargs.keys() and viewer_kwargs["backend"] != "plotly":
            raise NotImplementedError(f"Backend '{viewer_kwargs['backend']}' is currently not supported. Please use 'plotly'. ")
        if "xlabel" not in viewer_kwargs.keys():
            viewer_kwargs["xlabel"] = "Testpulse Amplitude"
        if "ylabel" not in viewer_kwargs.keys():
            viewer_kwargs["ylabel"] = "Testpulse Pulse Height"
            
        viewer = Viewer(**viewer_kwargs)
        viewer.add_scatter(x=tpas, y=tp_phs, name="Testpulses")
        viewer.add_line(x=[], y=[], name="Fit")
        fit_x = np.linspace(-0.05*np.max(tpas), 1.1*np.max(tpas), n_grid_points)
        
        inputs = dict()
        for k, v in cls._PREVIEW_INPUTS.items():
            if v["dtype"] is bool:
                inputs[k] = widgets.Checkbox(description=k, value=v["default"])
            elif v["dtype"] is float:
                inputs[k] = widgets.FloatSlider(description=k, value=v["default"], min=v["domain"][0], max=v["domain"][1])
            elif v["dtype"] is int:
                inputs[k] = widgets.Dropdown(description=k, value=v["default"], options=tuple(range(v["domain"][0], v["domain"][1]+1)))
        
        def _on_value_change(*args_not_used, **kwargs_not_used):
            kwargs = {k: v.value for k,v in inputs.items()}
            viewer.update_scatter(name="Fit", x=fit_x, y=cls(**kwargs)(tpas, tp_phs, fit_x))
            
        for i in inputs.values():
            i.observe(_on_value_change, "value")
        
        # Run once to calculate the fit for default arguments
        _on_value_change()
        
        return widgets.VBox(
            [
                widgets.HBox(list(inputs.values())),
                widgets.HBox([viewer.get_figure()])
            ]
        )   

##########################################
####### TRANSFER FUNCTION CHILDREN #######
##########################################
class TFUnity(TransferFunction):
    """Do-Nothing class that is used to validate the automatic tests for all children of :class:`TransferFunction`."""
    def __init__(self):
        super().__init__()
        
    def __call__(self, tpas: np.ndarray, tp_phs: np.ndarray, tpes: np.ndarray):
        in_shape = np.shape(tpes)
        tpas, tp_phs, tpes = _sanitize_inputs(tpas, tp_phs, tpes, "tpas", "tp_phs", "tpes")
        return np.reshape(tpes*np.ones_like(tpes), in_shape)

class TFPchip(TransferFunction):
    """
    Transfer function using a Piecewise Cubic Hermite Interpolating Polynomial (monotonic cubic splines).
    
    :param fix_at_yaxis: If True, the value when intercepting the y-axis is fixed to the value specified by ``y_intercept``. I.e. when True, the fit considers the additional point ``(0, y_intercept)``. Defaults to True.
    :type fix_at_yaxis: bool
    :param y_intercept: The y-intercept corresponding to the previous argument. Defaults to 0.
    :type y_intercept: float

    This example just demonstrates how the interpolation function looks. For a general description on how to use it, see :class:`cait.versatile.analysisobjects.transferfunction.TransferFunction`.

    **Example:**

    .. code-block:: python
    
        import numpy as np
        import scipy as sp
        import cait.versatile as vai

        tpas = [1.0, 2.5, 4.0, 7.0]
        tp_phs = [0.8, 1.5, 1.9, 2.3]

        vai.TFPchip().preview(tpas, tp_phs)

    .. image:: media/TFPchipPreview.png
    """
    _PREVIEW_INPUTS = {
        "fix_at_yaxis": {"dtype": bool, "default": True},
        "y_intercept": {"dtype": float, "default": 0, "domain": (-1, 1)},
    }
    def __init__(self, 
                 fix_at_yaxis: bool = True, 
                 y_intercept: float = 0,
                 ):
        super().__init__(fix_at_yaxis=fix_at_yaxis, 
                         y_intercept=y_intercept,
                         )
        self._fix_at_yaxis = fix_at_yaxis
        self._y_intercept = y_intercept
        
    def _pchip(self, tpas: np.ndarray, tp_phs: np.ndarray):
        if self._fix_at_yaxis:
            tpas = np.hstack(([0], tpas))
            tp_phs = np.hstack((self._y_intercept*np.ones((tp_phs.shape[0], 1)), tp_phs))

        if np.any(np.diff(tpas)<=0):
            raise ValueError(f"Input argument 'tpas' must be strictly monotonically increasing. Got {tpas}.")
    
        return sp.interpolate.PchipInterpolator(tpas, tp_phs, axis=-1)
    
    def __call__(self, tpas: np.ndarray, tp_phs: np.ndarray, tpes: np.ndarray):
        in_shape = np.shape(tpes) # Save shape for output
        tpas, tp_phs, tpes = _sanitize_inputs(tpas, tp_phs, tpes, "tpas", "tp_phs", "tpes")
        # Now, we have the following shapes:
        # tpas: (n_unique_tpas,), tp_phs: (N, n_unique_tpas), tpes: (N, M)
        iterp_objects = self._pchip(tpas, tp_phs)
        
        # Evaluating the PchipInterpolator at M points after interpolating 
        # multiple polynomials at the same time (in our case N times) 
        # results in an evaluation of all N polynomials for all points. 
        # We only want to evaluate a given polynomial for M points. For that
        # the call function of PchipInterpolator was modified and implemented
        # slightly differently in _ppolyval.
        return np.reshape(
            _ppolyval(iterp_objects.c, iterp_objects.x, tpes, extrapolate=True),
            in_shape,
        )
    
class TFPoly(TransferFunction):
    """
    Transfer function using a polynomial.
    
    :param poly_deg: The degree of the polynomial to use. Defaults to 3.
    :type poly_deg: int
    :param fix_at_yaxis: If True, the value when intercepting the y-axis is fixed to the value specified by ``y_intercept``. I.e. when True, the fit considers the point ``(0, y_intercept)`` as an additional point. Defaults to True.
    :type fix_at_yaxis: bool
    :param y_intercept: The y-intercept corresponding to the previous argument. Defaults to 0.
    :type y_intercept: float

    This example just demonstrates how the interpolation function looks. For a general description on how to use it, see :class:`cait.versatile.analysisobjects.transferfunction.TransferFunction`.

    **Example:**

    .. code-block:: python
    
        import numpy as np
        import scipy as sp
        import cait.versatile as vai

        tpas = [1.0, 2.5, 4.0, 7.0]
        tp_phs = [0.8, 1.5, 1.9, 2.3]

        vai.TFPoly().preview(tpas, tp_phs)

    .. image:: media/TFPolyPreview.png
    """
    _PREVIEW_INPUTS = {
        "fix_at_yaxis": {"dtype": bool, "default": True},
        "y_intercept": {"dtype": float, "default": 0, "domain": (-1, 1)},
        "poly_deg": {"dtype": int, "default": 3, "domain": (1, 4)},
    }
    def __init__(self, 
                 poly_deg: int = 3, 
                 fix_at_yaxis: bool = True, 
                 y_intercept: float = 0,
                 ):
        super().__init__(poly_deg=poly_deg, 
                         fix_at_yaxis=fix_at_yaxis,
                         y_intercept=y_intercept,
                         )
        
        self._poly_deg = poly_deg
        self._fix_at_yaxis = fix_at_yaxis
        self._y_intercept = y_intercept
        
    def _poly_fit(self, tpas: np.ndarray, tp_phs: np.ndarray):
        if self._fix_at_yaxis:
            tpas = np.hstack(([0], tpas))
            tp_phs = np.hstack((self._y_intercept*np.ones((tp_phs.shape[0], 1)), tp_phs))
            
        # Return coefficients in ascending order (i.e. constant coeffient first).
        return np.fliplr(np.polyfit(tpas, tp_phs.T, self._poly_deg).T)
        
    def __call__(self, tpas: np.ndarray, tp_phs: np.ndarray, tpes: np.ndarray):
        in_shape = np.shape(tpes) # Save shape for output
        tpas, tp_phs, tpes = _sanitize_inputs(tpas, tp_phs, tpes, "tpas", "tp_phs", "tpes")
        # Now, we have the following shapes:
        # tpas: (n_unique_tpas,), tp_phs: (N, n_unique_tpas), tpes: (N, M)
        poly_coeffs = self._poly_fit(tpas, tp_phs) # shape (N, poly_deg+1)
        
        return np.reshape(
            np.sum([
                poly_coeffs[:,k][:,None]*tpes**k 
                for k in range(self._poly_deg+1)
                ], axis=0),
            in_shape
        )