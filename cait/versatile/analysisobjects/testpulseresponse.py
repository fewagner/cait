import math
import warnings
import enum
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
@numba.njit
def gauss_weight(x: float, sig: float):
    return 1/(math.sqrt(2*math.pi)*sig)*math.exp(-1/2*(x/sig)**2)

@numba.njit(parallel=True)
def gauss_smoothing(x: np.ndarray, y: np.ndarray, width: float):
    # The current implementation is memory-bound, meaning that calculations
    # could be faster if memory handling is done more cleverly.
    # One idea would be to process the data in chunks of size < 10000 because
    # then it would fit into the high-speed memory.
    # Additionally, the array access is quite 'back-and-forth', i.e. we probably
    # get many cache misses.
    l = x.shape[0]
    arr_out = np.zeros((l,))

    for i in numba.prange(l):
        current_weight = gauss_weight(0, width)
        val = current_weight*y[i]
        weights_total = current_weight
        j = 0
        dt = 0

        # Go to the left of the node as long as we are within 3 sigmas (in time).
        # Add node value weighted by Gaussian.
        while dt < 3*width:
            # Prevent double counting.
            if i-j != i:
                current_weight = gauss_weight(dt, width)
                val += current_weight*y[i-j]
                weights_total += current_weight

            j += 1
            if j > i:
                break
            dt = x[i] - x[i-j]

        j = 0
        dt = 0
        # Go to the right of the node as long as we are within 3 sigmas (in time).
        # Add node value weighted by Gaussian.
        while dt < 3*width:
            # Prevent double counting.
            if i+j != i:
                current_weight = gauss_weight(dt, width)
                val += current_weight*y[i+j]
                weights_total += current_weight

            j += 1
            if i+j >= l-1:
                break

            dt = x[i+j] - x[i]

        arr_out[i] = val/weights_total

    return arr_out

def _check_npoints_and_outliers(phs: np.ndarray):
    """Check if data contains 5-sigma outliers or less than 10 datapoints and issue warnings accordingly. Return flag of outliers, i.e. True means outlier."""
    outlier_flag = np.zeros(phs.shape, dtype=bool)

    if np.size(phs) < 10:
        warnings.warn(f"You attempt to construct a TestpulseResponse with only {np.size(phs)} phs-points. This might result in an unreliable calibration. Make sure that your cuts didn't remove a substential number of testpulses.", category=UserWarning)

    if np.size(phs) > 10:
        # Normalize inputs:
        # Median filter removes trend, differences should detect spikes
        diffs = np.diff(phs - sp.signal.medfilt(phs, 9))
        # Get median and +-1 sigma of the diffs distribution
        qs = np.quantile(diffs, sp.stats.norm.cdf(np.array([-1., 0., 1.])))

        # Scale the diffs distribution to a standard normal distribution
        std_gauss = diffs - qs[1]
        std_gauss[diffs>qs[1]] /= qs[2] - qs[1]
        std_gauss[diffs<qs[1]] /= qs[1] - qs[0]

        outside_flag = np.abs(std_gauss) > 5
        n_outside = np.sum(outside_flag)

        if n_outside > 0:
            warnings.warn(f"There are {n_outside} outliers in the phs-points that you use to construct the TestpulseResponse. This might heavily impact the reliability of your calibration. Make sure that your cuts remove outliers BEFORE constructing the TestpulseResponse. Alternatively, you can set 'remove_outliers=True', although this is NOT RECOMMENDED because ideally, testpulses are cleaned BEFOREHAND.", category=UserWarning)

        outlier_flag[1:] = outside_flag

    return outlier_flag

def _sanitize_inputs_prepare(x: np.ndarray, tp_phs: np.ndarray, remove_outliers: bool = False):
    orig_shape_x, orig_shape_tp_phs = np.shape(x), np.shape(tp_phs)
    x = np.atleast_1d(x).astype(float)
    tp_phs = np.atleast_1d(tp_phs).astype(float)

    if not np.ndim(x)==1:
        raise ValueError(f"Array 'x' has to be 1d. Got shape {orig_shape_x}.")
    if not np.ndim(tp_phs)==1:
        raise ValueError(f"Array 'tp_phs' has to be 1d. Got shape {orig_shape_tp_phs}.")
    if x.shape != tp_phs.shape:
        raise ValueError(f"Arrays 'x' and 'tp_phs' have to have same shape. Got shapes {orig_shape_x} and {orig_shape_tp_phs}.")

    outlier_flag = _check_npoints_and_outliers(tp_phs)

    if remove_outliers:
        x, tp_phs = x[~outlier_flag], tp_phs[~outlier_flag]
        n_outliers = np.sum(outlier_flag)
        if n_outliers > 0:
            warnings.warn(f"{n_outliers} outlier(s) were automatically removed for TestpulseResponse. Remaining datapoints: {np.size(tp_phs)-n_outliers}", category=UserWarning)

    return x, tp_phs

def _sanitize_inputs_call(x: np.ndarray):
    orig_shape_x = np.shape(x)
    x = np.atleast_1d(x).astype(float)

    if not np.ndim(x)==1:
        raise ValueError(f"Array 'x' has to be 1d. Got shape {orig_shape_x}.")

    return x

##########################################
########## ABSTRACT BASE CLASS ###########
##########################################
class TestpulseResponse(SerializingMixin, ABC):
    """
    Abstract object describing the height of testpulses over x (usually time) for a single testpulse amplitude (TPA).

    :param remove_outliers: If True, pulse heights that deviate more than 5 sigma from the pulse heights trend will be removed. Note that we always recommend to REMOVE OUTLIERS MANUALLY before even starting the calibration! Defaults to False.
    :type remove_outliers: bool, optional

    To add a specific model (e.g. piecewise cubic splines or polynomials), the following things need to be implemented:

    - :func:`TestpulseResponse.__init__`: Initialize the object with parameters that it needs (e.g. polynomial order) and call the constructor of the super class with those arguments.
    - :func:`TestpulseResponse.prepare`: Takes a 1-d array of x-values (usually microsecond timestamps) of testpulses of A SINGLE testpulse amplitude and a 1-d array of the corresponding testpulse heights at these x-values (usually timestamps) and performs the fit over x. This method must return the object itself again. Must perform input shape validation, raise a ValueError in case of shape mismatch, and generate a UserWarning if there are outliers in the pulse-heights or less than 10 data points (you can use the function '_sanitize_inputs_prepare' to perform those checks).
    - :func:`TestpulseResponse.__call__`: Takes a 1-d array of x-values (usually microsecond timestamps) and evaluates the polynomial (which was pre-calculated in :func:`TestpulseResponse.prepare`). Must perform input shape validation and raise a ValueError in case of shape mismatch (you can use the function '_sanitize_inputs_call' to perform those checks).
    - If the class attribute ``_PREVIEW_INPUTS`` is defined, it will be used for interactively changing the arguments passed to ``__init__`` in the preview methods. See below for how they must be structured. You can only allow a subset of the input arguments to be varied, but all field names in ``_PREVIEW_INPUTS`` must be input arguments to ``__init__``.

    The usage of this class is demonstrated below for the specific implementation of :class:`TPRPoly`. See the docstrings of the child classes for more information on the specific behavior.

    **Example:**

    .. code-block:: python

        import numpy as np
        import scipy as sp
        import cait.versatile as vai

        start_us = 1426321613000000
        ts = np.sort(sp.stats.randint.rvs(low=start_us, high=start_us+3*1e6*3600, size=1000))
        tp_phs = sp.stats.norm.rvs(loc=1.3, scale=0.05, size=len(ts))

        selected_times = np.linspace(start_us, start_us+10000, 100)

        # Instantiate the object (this does not yet do any fitting).
        tpr = vai.TPRPoly()

        # Calculate the fit (this is done only once. Afterwards, the
        # object can be used for as many evaluations as you like).
        tpr.prepare(ts, tp_phs)

        # Get testpulse pulse heights at selected values
        tph_at_selected_times = tpr(selected_times)

        # Every TestpulseResponse object can be plotted to interactively
        # try out different configuration parameters (even before instantiating):
        vai.TPRPoly.preview(ts, tp_phs)

    .. automethod:: __call__
    """
    _PREVIEW_INPUTS = {}
    #{
    #    "arg1": {"dtype": bool, "default": False}, # Results in Checkbox
    #    "arg2": {"dtype": float, "default": 3.0, "domain": (1.0, 5.0)}, # Results in Slider
    #    "arg3": {"dtype": int, "default": 1, "domain": (0, 4)} # Results in Dropdown
    #}

    class Extrapolation(enum.IntEnum):
        CONST = enum.auto()
        EXT = enum.auto()


    def __init__(self, remove_outliers: bool = False, extrapolation = Extrapolation.CONST, **kwargs):
        super().__init__(remove_outliers=remove_outliers, **kwargs)
        self._init_kwargs = dict(remove_outliers=remove_outliers, extrapolation=extrapolation, **kwargs)
        self._bounds = None

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self._init_kwargs.items()])})"

    @abstractmethod
    def prepare(self, x: np.ndarray, tp_phs: np.ndarray):
        """
        Precalculate the fit over x (usually time) given x-values (usually testpulse timestamps) and testpulse heights.

        It is the caller's responsibility to only pass values corresponding to a single testpulse amplitude (TPA), AND that this data has been cleaned from outliers.

        :param x: Independent variable of testpulses for the calibration (usually (microsecond) timestamps).
        :type x: np.ndarray, shape (N,)
        :param tp_phs: Pulse heights of testpulses used for the calibration.
        :type tp_phs: np.ndarray, shape (N,)

        :return: This instance.
        :rtype: TestpulseResponse
        """
        ...

    @abstractmethod
    def __call__(self, x: np.ndarray):
        """
        Return predicted testpulse heights for given x (usually timestamps).

        To use this method, 'prepare' has to be called beforehand.

        :param x: Independent variable for where to evaluate the testpulse heights (usually (microsecond) timestamps).
        :type x: np.ndarray, shape (N,)

        :return: Predicted testpulse heights at given x-values.
        :rtype: np.ndarray, same shape as x
        """
        ...

    @classmethod
    def preview(cls,
                x: np.ndarray,
                tp_phs: np.ndarray,
                n_grid_points: int = 1000,
                **viewer_kwargs):
        """
        Plot the testpulses defined by ``(x, tp_phs)`` and preview the fit result.

        :param ts: (Microsecond) timestamps of testpulses used for the calibration.
        :type ts: np.ndarray, shape (K,)
        :param tp_phs: Pulse heights of testpulses used for the calibration.
        :type tp_phs: np.ndarray, shape (K,)
        :param n_grid_points: Number of points used for plotting the fit.
        :type n_grid_points: int
        :param viewer_kwargs: Additional keyword arguments for :class:`cait.versatile.plot.viewer.Viewer`.
        :type viewer_kwargs: Any
        """
        if "backend" in viewer_kwargs.keys() and viewer_kwargs["backend"] != "plotly":
            raise NotImplementedError(f"Backend '{viewer_kwargs['backend']}' is currently not supported. Please use 'plotly'. ")
        if "xlabel" not in viewer_kwargs.keys():
            viewer_kwargs["xlabel"] = "x"
        if "ylabel" not in viewer_kwargs.keys():
            viewer_kwargs["ylabel"] = "Testpulse Pulse Height"

        viewer = Viewer(**viewer_kwargs)
        viewer.add_scatter(x=x, y=tp_phs, name="Testpulses")
        viewer.add_line(x=[], y=[], name="Fit")
        fit_x = np.linspace(np.min(x), np.max(x), n_grid_points)

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
            viewer.update_scatter(name="Fit", x=fit_x, y=cls(**kwargs).prepare(x, tp_phs)(fit_x))

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


    def impose_extrapolation(self, x):
        if self._bounds is None:
            return x

        em = self._init_kwargs['extrapolation']
        if em not in vars(self.Extrapolation).values():
            raise ValueError(f"extrapolation must be one of "
                             f'{str([f"Extrapolation.{x.name}" for x in self.Extrapolation])}'
                             f' (i.e. {str([x.value for x in self.Extrapolation])})'
                             f", got {em}"
                             )
        x = np.array(x)
        if em == self.Extrapolation.CONST:
            # If there is a fit method with bounds, clip the input to the bounds
            x = np.clip(x, self._bounds[0], self._bounds[1])
        return x


##########################################
############# CHILD CLASSES ##############
##########################################
class TPRUnity(TestpulseResponse):
    """
    Do-Nothing class that is used to validate the automatic tests for all children of :class:`TestpulseResponse`.

    :param kwargs: Additional keyword arguments for see :class:`cait.versatile.analysisobjects.testpulseresponse.TestpulseResponse`.
    :type poly_order: Any
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare(self, x: np.ndarray, tp_phs: np.ndarray):
        x, tp_phs = _sanitize_inputs_prepare(x, tp_phs, self._init_kwargs["remove_outliers"])
        self._mean_ph = np.mean(tp_phs)
        return self

    def __call__(self, x: np.ndarray):
        in_shape = np.shape(x)
        x = _sanitize_inputs_call(x)
        return np.reshape(self._mean_ph*np.ones_like(x), in_shape)

class TPRPoly(TestpulseResponse):
    """
    Fit testpulse heights (globally) with a polynomial.

    :param poly_order: Order of the polynomial. Defaults to 1 (i.e. a linear polynomial).
    :type poly_order: int, optional
    :param kwargs: Additional keyword arguments for see :class:`cait.versatile.analysisobjects.testpulseresponse.TestpulseResponse`.
    :type poly_order: Any

    This example just demonstrates how the interpolation function looks. For a general description on how to use it, see :class:`cait.versatile.analysisobjects.testpulseresponse.TestpulseResponse`.

    **Example:**

    .. code-block:: python

        import numpy as np
        import scipy as sp
        import cait.versatile as vai

        start_us = 1426321613000000
        ts = np.sort(sp.stats.randint.rvs(low=start_us, high=start_us+3*1e6*3600, size=1000))
        tp_phs = sp.stats.norm.rvs(loc=1.3, scale=0.05, size=len(ts))

        vai.TPRPoly.preview(ts, tp_phs)

    .. image:: media/TPRPolyPreview.png
    """
    _PREVIEW_INPUTS = {
        "poly_order": {"dtype": int, "default": 1, "domain": (0, 4)},
    }
    def __init__(self, poly_order: int = 1, **kwargs):
        super().__init__(poly_order=poly_order, **kwargs)
        self._poly_order = poly_order
        self._fit_poly = None

    def prepare(self, x: np.ndarray, tp_phs: np.ndarray):
        x, tp_phs = _sanitize_inputs_prepare(x, tp_phs, self._init_kwargs["remove_outliers"])
        self._bounds = (x.min(), x.max())

        # Scale so that x-axis has values in (0, 1) to improve stability
        self._scale_ts = lambda X: (X-np.min(x))/(np.max(x)-np.min(x))
        self._fit_poly = np.poly1d(np.polyfit(self._scale_ts(x), tp_phs, self._poly_order))

        return self

    def __call__(self, x: np.ndarray):
        if self._fit_poly is None:
            raise Exception("Need to run prepare first.")

        x = self.impose_extrapolation(x)

        return np.reshape(
            self._fit_poly(self._scale_ts(_sanitize_inputs_call(x))),
            np.shape(x),
        )

class TPRCubicSpline(TestpulseResponse):
    """
    Fit testpulse heights with piecewise cubic polynomials. Optionally, Gaussian data smoothing is available before the spline interpolation is performed.

    The smoothing can be specified in units different from the x-units using the ``param_scale`` argument. I.e. if the x values are microsecond timestamps but you want to specify the smoothing width (kernel length) in terms of hours, you can set ``param_scale=1e-6/3600`` which converts microseconds to hours.

    :param kernel_length: Size of the Gaussian smoothing (1 sigma of the distribution). Units of ``kernel_length`` are determined by ``param_scale``. Defaults to 0.5, i.e. with ``param_scale=1e-6/3600``, this corresponds to half an hour.
    :type kernel_length: float, optional
    :param param_scale: Conversion factor between the argument ``kernel_length`` and the units of the input x-values. E.g. If the x-values are given in microseconds, but you want to specify the ``kernel_length`` in hours, you can set ``param_scale=1e-6/3600``. Defaults to ``1e-6/3600``.
    :type param_scale: float, optional
    :param kwargs: Additional keyword arguments for see :class:`cait.versatile.analysisobjects.testpulseresponse.TestpulseResponse`.
    :type poly_order: Any

    This example just demonstrates how the interpolation function looks. For a general description on how to use it, see :class:`cait.versatile.analysisobjects.testpulseresponse.TestpulseResponse`.

    **Example:**

    .. code-block:: python

        import numpy as np
        import scipy as sp
        import cait.versatile as vai

        start_us = 1426321613000000
        ts = np.sort(sp.stats.randint.rvs(low=start_us, high=start_us+3*1e6*3600, size=1000))
        tp_phs = sp.stats.norm.rvs(loc=1.3, scale=0.05, size=len(ts))

        vai.TPRCubicSpline.preview(ts, tp_phs)

    .. image:: media/TPRCubicSplinePreview.png
    """
    _PREVIEW_INPUTS = {
        "kernel_length": {"dtype": float, "default": 0.5, "domain": (0, 3)},
    }
    def __init__(self, kernel_length: int = 0.5, param_scale: float = 1e-6/3600, **kwargs):
        super().__init__(kernel_length=kernel_length, param_scale=param_scale, **kwargs)

        if kernel_length<0:
            raise ValueError(f"Kernel length has to be a non-negative float. Not {kernel_length}.")

        self._kernel_length = kernel_length
        self._scale = param_scale
        self._fit_poly = None

    def prepare(self, x: np.ndarray, tp_phs: np.ndarray):
        # Sanitize data
        x, tp_phs = _sanitize_inputs_prepare(x, tp_phs, self._init_kwargs["remove_outliers"])
        self._bounds = (x.min(), x.max())

        sort_inds = np.argsort(x)
        x, tp_phs = x[sort_inds], tp_phs[sort_inds]

        if np.any(np.diff(x)<=0):
            raise ValueError("For Cubic Spline interpolation, the testpulse x-values have to be unique (no duplications allowed) and monotonically increasing.")

        if self._kernel_length > 0:
            tp_phs = gauss_smoothing((x-np.min(x))*self._scale, tp_phs, self._kernel_length)

        self._fit_poly = sp.interpolate.CubicSpline(x, tp_phs, extrapolate=True)

        return self

    def __call__(self, x: np.ndarray):
        if self._fit_poly is None:
            raise Exception("Need to run prepare first.")

        x = self.impose_extrapolation(x)
        return np.reshape(
            self._fit_poly(_sanitize_inputs_call(x)),
            np.shape(x),
        )
