import copy
import datetime

import numpy as np
from ipywidgets import widgets

from ...serialize import SerializingMixin
from ..plot import Viewer
from .testpulseresponse import TestpulseResponse
from .transferfunction import TransferFunction


def _sanitize_inputs(arr1: np.ndarray, arr2: np.ndarray, name1: str, name2: str):
    # Check inputs and standardize arrays
    orig_shape_arr1, orig_shape_arr2 = np.shape(arr1), np.shape(arr2)
    arr1 = np.atleast_1d(arr1)
    arr2 = np.atleast_2d(arr2).T if np.ndim(arr2)<2 else np.atleast_2d(arr2)

    if not np.ndim(arr1)==1: 
        raise ValueError(f"Array '{name1}' has to be 1d. Got shape {orig_shape_arr1}.")
    if np.ndim(arr2)>2: 
        raise ValueError(f"Array '{name2}' has to be at most 2d. Got shape {orig_shape_arr2}.")
    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError(f"The number of elements in '{name1}' must match the first dimension of '{name2}'. Need shapes (N,) and (N,) or (N, M). Got shapes {orig_shape_arr1} and {orig_shape_arr2}.")
    
    return (
        arr1, # shape (N,)
        arr2, # shape (N, M)
    )

class EnergyCalibration(SerializingMixin):
    """
    Energy calibration for fixed testpulse amplitudes (``tpas``) and testpulse pulse heights (``tp_phs``) which vary with an independent variable (``tp_x``). This independent variable is usually the (microsecond) timestamp of the testpulses but can in principle be any quantity.

    NOTICE: The array ``tp_phs`` must be cleaned from outliers BEFORE starting the calibration!
    
    :param tp_x: The independent variable which the testpulse pulse heights ``tp_phs`` depend on. Usually the (microsecond) timestamps of the testpulses.
    :type tp_x: np.array, shape (N,)
    :param tp_phs: The testpulse pulse heights which vary with ``tp_x``.
    :type tp_phs: np.array, shape (N,)
    :param tpas: The testpulse amplitudes of each testpulse. This array is used to divide different testpulse amplitudes into separate calibrations.
    :type tpas: np.array, shape (N,)
    :param testpulse_response: The description for how to fit the set of ``(tp_x, tp_phs)`` points (for each TPA individually).
    :type testpulse_response: TestpulseResponse
    :param transfer_function: The description for how to fit ``(tpas, tp_phs)`` values for a fixed x-value.
    :type transfer_function: TransferFunction
    :param max_x_gap: If ``<np.inf``, a separate calibration is started if there are gaps in the independent variable ``tp_x`` which are larger than this value. Units of ``max_x_gap`` are determined by ``param_scale``. Usually, this would be used when ``tp_x`` are (microsecond) timestamps and there is no data for > half an hour. In such a case, set ``max_x_gap=0.5`` and ``param_scale=1e-6/3600``. Defaults to ``np.inf``, i.e. this functionality is disabled. 
    :type max_x_gap: float, optional
    :param param_scale: Conversion factor between the argument ``max_x_gap`` and the units of the input x-values. E.g. If the x-values are given in microseconds, but you want to specify the ``max_x_gap`` in hours, you can set ``param_scale=1e-6/3600``. Defaults to ``1e-6/3600``.
    :type param_scale: float, optional

    **Example:**

    .. code-block:: python
    
        import numpy as np
        import scipy as sp
        import cait.versatile as vai

        # Define a random state (so that this example is reproducible)
        rng_seed = 137

        # Create mock data. In your analysis, you would of course already
        # have timestamps, testpulseamplitudes and testpulse heights.
        N = 5000
        start_us = 1426321613000000
        ts = np.hstack([sp.stats.randint.rvs(low=start_us,
                                            high=start_us+3*1e6*3600, 
                                            size=N, 
                                            random_state=rng_seed), 
                        sp.stats.randint.rvs(low=start_us+4*1e6*3600,
                                            high=start_us+7*1e6*3600, 
                                            size=N,
                                            random_state=rng_seed)])
        tpas = sp.stats.randint.rvs(low=1, high=6, size=2*N, random_state=rng_seed)
        tp_phs = sp.stats.norm.rvs(loc=np.sqrt(0.3*tpas), scale=0.05, size=2*N, random_state=rng_seed)

        # Create the EnergyCalibration object.
        # Choose a testpulse respose object from ['TPRCubicSpline', 'TPRPoly'].
        # Choose a transfer function object from ['TFPoly', 'TFPchip'].
        # Here, we choose piecewise cubic splines with Gaussian smoothing and 
        # piecewise cubic Hermite polynomials.
        my_ecal = vai.EnergyCalibration(
            tp_x=ts,
            tp_phs=tp_phs,
            tpas=tpas,
            testpulse_response=vai.TPRCubicSpline(kernel_length=0.1),
            transfer_function=vai.TFPchip(),
            max_x_gap=0.5, # start new calibrations for segments which are separated by > 0.5 hours
        )

        # Open plot that lets you interactively check what your calibration is doing.
        # You can change the parameters of the TestpulseResponse and TransferFunction.
        # You can click at any time in the left plot to see what the transfer function 
        # looks like at that time.
        my_ecal.preview()

        # To convert TPE values to PH values, call the object:
        timestamps_of_interest = ts[100:200]
        tpes_of_interest = np.linspace(2, 3, len(timestamps_of_interest))
        output_phs = my_ecal(timestamps_of_interest, tpes_of_interest)   

        # To convert PH values to TPE values, call its inverse:
        timestamps_of_interest = ts[100:200]
        phs_of_interest = output_phs
        output_tpes = my_ecal.inverse(timestamps_of_interest, phs_of_interest)   

        # ... in 'output_tpes' you should have recovered the initial 'tpes_of_interest'.
        # Of course, in a real-world scenario, you would only be interested in either 
        # of the two directions.               

    .. image:: media/ECalPreview.png

    .. automethod:: __call__
    """
    def __init__(self,
                 tp_x: np.ndarray,
                 tp_phs: np.ndarray,
                 tpas: np.ndarray,
                 testpulse_response: TestpulseResponse,
                 transfer_function: TransferFunction,
                 max_x_gap: float = np.inf,
                 param_scale: float = 1e-6/3600,
                 ):
        # For the SerializingMixin to remember the inputs
        super().__init__(tp_x=tp_x, 
                         tp_phs=tp_phs, 
                         tpas=tpas, 
                         testpulse_response=testpulse_response,
                         transfer_function=transfer_function,
                         max_x_gap=max_x_gap,
                         param_scale=param_scale,
                         )
        
        for obj, target, name in zip(
            [testpulse_response, transfer_function],
            [TestpulseResponse, TransferFunction],
            ["testpulse_response", "transfer_function"]
            ):
            if not isinstance(obj, target):
                raise TypeError(f"Input argument '{name}' must be an instance of '{target.__class__.__name__}', not {type(obj)}")
        
        for arr, name in zip([tp_x, tpas, tp_phs], ["tp_x", "tpas", "tp_phs"]):
            if not np.ndim(arr)==1:
                raise ValueError(f"Input argument '{name}' has to be 1d. Got shape {np.shape(arr)}.")
        
        if len(set([np.shape(x) for x in [tp_x, tpas, tp_phs]])) > 1:
            raise ValueError(f"Input arguments 'tp_x', 'tpas', and 'tp_phs' have to have the same shape. Got {np.shape(tp_x)}, {np.shape(tpas)}, and {np.shape(tp_phs)}.")
        
        if max_x_gap<=0:
            raise ValueError(f"Input argument 'max_x_gap' has to be positive. Not {max_x_gap}.")
        
        # Sanitize data
        tp_x, tpas, tp_phs = np.array(tp_x), np.array(tpas), np.array(tp_phs)
        sort_inds = np.argsort(tp_x)
        tp_x, tpas, tp_phs = tp_x[sort_inds], tpas[sort_inds], tp_phs[sort_inds]
        
        # Determine unique TPA list
        self._unique_tpas = np.sort(np.unique(tpas))
        
        # Separate into continuous segments
        if max_x_gap < np.inf:
            # 'scaled_x' is usually 'hours'
            scaled_x = (tp_x - np.min(tp_x))*param_scale
            anchor_inds = np.nonzero(np.diff(scaled_x)>max_x_gap)[0] + 1
            self._x_blocks = np.split(tp_x, anchor_inds)
            self._tpa_blocks = np.split(tpas, anchor_inds)
            self._ph_blocks = np.split(tp_phs, anchor_inds)
        else:
            anchor_inds = np.array([0])
            self._x_blocks = [tp_x]
            self._tpa_blocks = [tpas]
            self._ph_blocks = [tp_phs]
        
        # Save 'grid' of domains for later.
        # (start0, end0, start1, end1, ...)
        block_domains = []
        for current_x in self._x_blocks:
            block_domains.append((np.min(current_x), np.max(current_x)))
            
        self._domain_grid = np.array(block_domains).flatten()
            
        # Create testpulse response for each domain with the initial testpulse response object.
        # (Can be changed later - in preview - through self._update_tpr)
        self._update_tpr(testpulse_response)
        
        # Set transfer function to the initial transfer function object.
        # (Can be changed later - in preview - through self._update_tf)
        self._update_tf(transfer_function)
        
        # Save Testpulse Response and Transfer Function objects for plotting and __repr__
        self._init_tpr_obj = testpulse_response
        self._init_tf_obj = transfer_function

        # Save input data for plotting
        self._tp_x, self._tp_phs, self._scale = tp_x, tp_phs, param_scale

    def __repr__(self):
        return f"{self.__class__.__name__}(TPR={self._init_tpr_obj}, TF={self._init_tf_obj})"
        
    def __call__(self, x: np.ndarray, tpes: np.ndarray):
        """
        Convert testpulse equivalent heights to pulse heights at given x-values (usually timestamps).

        :param x: Independent variable (usually (microsecond) timestamps) for which to evaluate the calibration.
        :type x: np.ndarray, shape (N,)
        :param tpes: Testpulse equivalent heights at x-values for which to calculate the pulse height. If 2D, the rows are evaluated at identical x-values.
        :type tpes: np.ndarray, shape (N,) or (N, M)

        :return: The pulse height for pulses at ``x`` with testpulse equivalent heights ``tpes``.
        :rtype: np.ndarray, shape (N,) or (N, M)
        """
        in_shape = np.shape(tpes)
        x, tpes = _sanitize_inputs(x, tpes, "x", "tpes") # guaranteed shapes (N,) and (N, M)
        tphs, mask_valid = self.get_tp_phs(x)
        
        # x-values (usually timestamps) which fall outside of domains are marked invalid (NaN)
        out = np.nan*np.ones_like(tpes)
        out[mask_valid, ...] = self._tf(self._unique_tpas,
                                        tphs[mask_valid, ...], 
                                        tpes[mask_valid, ...])
        
        return np.reshape(out, in_shape)
    
    def inverse(self, x: np.ndarray, phs: np.ndarray):
        """
        Convert pulse heights to testpulse equivalent heights at given x-values (usually timestamps).

        :param x: Independent variable (usually (microsecond) timestamps) for which to evaluate the calibration.
        :type x: np.ndarray, shape (N,)
        :param phs: Heights of pulses at x-values for which to calculate the testpulse equivalent height. If 2D, the rows are evaluated at identical times.
        :type phs: np.ndarray, shape (N,) or (N, M)

        :return: The testpulse equivalent energies for pulses at ``x`` with pulse heights ``phs``.
        :rtype: np.ndarray, shape (N,) or (N, M)
        """
        in_shape = np.shape(phs)
        x, phs = _sanitize_inputs(x, phs, "x", "phs") # guaranteed shapes (N,) and (N, M)
        tphs, mask_valid = self.get_tp_phs(x)
        
        # x-values (usually timestamps) which fall outside of domains are marked in valid (NaN)
        out = np.nan*np.ones_like(phs)
        out[mask_valid, ...] = self._tf.inverse(self._unique_tpas, 
                                                tphs[mask_valid, ...], 
                                                phs[mask_valid, ...])
        
        return np.reshape(out, in_shape)
    
    def _update_tpr(self, tpr_obj: TestpulseResponse):
        """Update the currently used TestpulseResponse object. Called by ``.preview`` to dynamically modify fit parameters."""
        # For each segment, separate into TPA and initialize TestpulseResponse object 
        self._tprs = []
        for current_x, current_tpas, current_phs in zip(self._x_blocks, self._tpa_blocks, self._ph_blocks):
            self._tprs.append([
                copy.copy(tpr_obj).prepare(
                    x=current_x[current_tpas==tpa], 
                    tp_phs=current_phs[current_tpas==tpa]
                )
                for tpa in self._unique_tpas
            ])
        return self
    
    def _update_tf(self, tf_obj: TransferFunction):
        """Update the currently used TransferFunction object. Called by ``.preview`` to dynamically modify fit parameters."""
        self._tf = tf_obj
        return self

    def _get_tpr_plot_dict(self, n_fit_grid_points: int, downsample_factor: int): 
        x_start, x_stop = np.min(self._tp_x), np.max(self._tp_x)
        x_len = x_stop - x_start
        x_fit = np.linspace(x_start-0.01*x_len, x_stop+0.01*x_len, n_fit_grid_points)
        y_fit, _ = self.get_tp_phs(x_fit)

        return {
            "scatter": {
                "Testpulses": [
                    np.atleast_1d(self._tp_x[::downsample_factor]),
                    np.atleast_1d(self._tp_phs[::downsample_factor])
                ]
            },
            "line": {
                f"TPA {tpa:.2g}": [
                    np.atleast_1d(x_fit),
                    np.atleast_1d(fit)
                ] 
                for tpa, fit in zip(self._unique_tpas, y_fit.T)
            },
        }
        
    def _get_tf_plot_dict(self, x: float, n_fit_grid_points: int):
        if np.ndim(x)>0:
            raise ValueError(f"Input argument 'x' must be scalar. Got shape {np.shape(x)}.")
        
        tpe_start, tpe_stop = 0, np.max(self._unique_tpas)
        tpe_len = tpe_stop - tpe_start
        tpe_fit = np.linspace(tpe_start-0.01*tpe_len, tpe_stop+0.01*tpe_len, n_fit_grid_points)
        y_fit = np.reshape(self(np.atleast_1d(x), np.atleast_2d(tpe_fit)), np.shape(tpe_fit))
        
        tp_phs = np.reshape(self.get_tp_phs(x)[0], np.shape(self._unique_tpas))

        return {
            "line": {
                "Fit": [
                    np.atleast_1d(tpe_fit),
                    np.atleast_1d(y_fit)
                ]
            },
            "scatter": {
                f"TPA {tpa:.2g}": [
                    np.atleast_1d(tpa),
                    np.atleast_1d(tp_ph)
                ]
                for tpa, tp_ph in zip(self._unique_tpas, tp_phs)
            }
        }

    def get_tp_phs(self, x: np.ndarray):
        """
        Return the testpulse heights evaluated at x (usually (microsecond) timestamps) for each unique TPA (columns).
        
        :param x: The independent variable (usually microsecond timestamp) for which to return the testpulse pulse heights.
        :type x: np.ndarray, shape (N,)
        
        :return: Pulse heights at ``x`` for each unique testpulse amplitude (TPA). Invalid values (which fall outside the calibration range) are marked as ``np.nan``. Furthermore, a flag of valid ``x`` values is returned.
        :rtype: Tuple[np.ndarray, np.ndarray], shape (N, n_unique_tpas) and (N,)
        """
        # Sanitize input
        x = np.atleast_1d(x)
        if x.ndim!=1: 
            raise ValueError(f"Array 'x' has to be 1d. Got shape {x.shape}.")

        out = np.nan*np.ones((x.shape[0], len(self._unique_tpas)))
        
        # Bin pulse_ts into the previously obtained domains to know
        # which testpulse response object we have to call
        domain_inds = np.digitize(x, self._domain_grid)
        mask_valid = np.mod(domain_inds, 2) != 0
        falls_into_domain = np.array((domain_inds[mask_valid]-1)/2, dtype=int)
        
        # Select the correct domains for all x-values (usually timestamps)
        # and evaluate the testpulse response for all testpulses. 
        # Create an array with rows for each TPA and columns for each x-value.
        tphs = np.zeros((len(self._unique_tpas), len(falls_into_domain)))
        for domain_ind in np.unique(falls_into_domain):
            current_mask = falls_into_domain==domain_ind
            tphs[:, current_mask] = np.array([
                tpr(x[mask_valid][current_mask])
                for tpr in self._tprs[domain_ind]
            ])
        
        out[mask_valid, :] = tphs.T
        
        return out, mask_valid
        
    def plot_testpulse_response(self, 
                                n_fit_grid_points: int = 1000,
                                downsample_factor: int = 1,
                                **viewer_kwargs,
                                ):
        """
        Plot the testpulse response at a given x-value (usually microsecond timestamp).

        :param n_fit_grid_points: Number of points used for plotting the fits. Defaults to 1000.
        :type n_fit_grid_points: int, optional
        :param downsample_factor: Downsample the ``(tp_x, tp_phs)`` points by this factor. Defaults to 1, i.e. no downsampling.
        :type downsample_factor: int, optional
        :param viewer_kwargs: Additional keyword arguments for :class:`cait.versatile.plot.viewer.Viewer`.
        :type viewer_kwargs: Any
        """
        return Viewer(
            data=self._get_tpr_plot_dict(n_fit_grid_points, downsample_factor),
            xlabel="x", 
            ylabel="Pulse Height", 
            **viewer_kwargs
        )

    def plot_transfer_function(self, 
                               x: float, 
                               n_fit_grid_points: int = 1000,
                               **viewer_kwargs,
                               ):
        """
        Plot the testpulse response at a given x-value (usually microsecond timestamp).

        :param x: Number of points used for plotting the fits. Defaults to 1000.
        :type x: float
        :param n_fit_grid_points: Number of points used for plotting the fits. Defaults to 1000.
        :type n_fit_grid_points: int, optional
        :param viewer_kwargs: Additional keyword arguments for :class:`cait.versatile.plot.viewer.Viewer`.
        :type viewer_kwargs: Any
        """
        return Viewer(
            data=self._get_tf_plot_dict(x, n_fit_grid_points),
            xlabel="Testpulse-equivalent amplitude",
            ylabel="Pulse Height",
            **viewer_kwargs
        )

    def preview(self, 
                n_fit_grid_points: int = 1000, 
                downsample_factor: int = 1,
                **viewer_kwargs,
                ):
        """
        Plot the testpulse response over time and the transfer function for selected times. 

        :param n_fit_grid_points: Number of points used for plotting the fits. Defaults to 1000.
        :type n_fit_grid_points: int, optional
        :param downsample_factor: Downsample the ``(tp_x, tp_phs)`` points by this factor. Defaults to 1, i.e. no downsampling.
        :type downsample_factor: int, optional
        :param viewer_kwargs: Additional keyword arguments for :class:`cait.versatile.plot.viewer.Viewer`.
        :type viewer_kwargs: Any
        """
        if "backend" in viewer_kwargs.keys() and viewer_kwargs["backend"] != "plotly":
            raise NotImplementedError(f"Backend '{viewer_kwargs['backend']}' is currently not supported. Please use 'plotly'.  To plot using other backends, use 'plot_testpulse_response' and 'plot_transfer_function'.")
        
        if "width" not in viewer_kwargs.keys():
            viewer_kwargs["width"] = 550
        
        tpr_viewer = Viewer(
            xlabel="x",
            ylabel="Pulse Height",
            **viewer_kwargs
        )
        tf_viewer = Viewer(
            xlabel="Testpulse-equivalent Height",
            **viewer_kwargs
        )

        # Already draw pulse height scatter (will not be redrawn later) and fits (will be redrawn)
        tpr_viewer.plot(self._get_tpr_plot_dict(n_fit_grid_points, downsample_factor))

        self._current_plot_x = np.min(self._tp_x)

        #scale_time = lambda X: (X-first_data_x)*self._scale   
        #f"Time (h) since {np.array([first_data_x], dtype='datetime64[us]')[0].astype(datetime.datetime).strftime('%d-%b-%Y, %H:%M:%S')}, ({first_data_x})", 
                           
        tpr_inputs, tf_inputs = dict(), dict()
        
        for inputs, obj in zip([tpr_inputs, tf_inputs], [self._init_tpr_obj, self._init_tf_obj]):
            for k, v in obj._PREVIEW_INPUTS.items():
                desc = k
                val = obj._init_kwargs[k] if k in obj._init_kwargs.keys() else v["default"]
                if v["dtype"] is bool:
                    inputs[k] = widgets.Checkbox(
                        description=desc, 
                        value=val,
                    )
                elif v["dtype"] is float:
                    inputs[k] = widgets.FloatSlider(
                        description=desc, 
                        value=val, 
                        min=v["domain"][0], 
                        max=v["domain"][1],
                        step=(v["domain"][1]-v["domain"][0])/1000
                    )
                elif v["dtype"] is int:
                    inputs[k] = widgets.Dropdown(
                        description=desc, 
                        value=val, 
                        options=tuple(range(v["domain"][0], v["domain"][1]+1)),
                    )
                       
        def _on_tpr_value_change(*args, **kwargs):
            new_kwargs = self._init_tpr_obj._init_kwargs.copy()
            for k, v in tpr_inputs.items():
                new_kwargs[k] = v.value

            new_tpr_obj = self._init_tpr_obj.__class__(**new_kwargs)
            self._update_tpr(new_tpr_obj)
            new_plot_dict = self._get_tpr_plot_dict(n_fit_grid_points, downsample_factor)

            # Update only lines (the fits) and don't re-draw scatter (the data)
            tpr_viewer.plot(dict(line=new_plot_dict["line"]))

            # Also update TF plot
            _on_tf_value_change()
            
        def _on_tf_value_change(*args, **kwargs):
            new_kwargs = self._init_tf_obj._init_kwargs.copy()
            for k, v in tf_inputs.items():
                new_kwargs[k] = v.value

            new_tf_obj = self._init_tf_obj.__class__(**new_kwargs)
            self._update_tf(new_tf_obj)
            new_plot_dict = self._get_tf_plot_dict(self._current_plot_x, n_fit_grid_points)

            # Update both lines (the fits) and re-draw scatter
            tf_viewer.plot(new_plot_dict)
            
        def _on_tpr_click(trace, points, state):
            ind = points.point_inds[0]
            self._current_plot_x = self._tp_x[ind]
            _on_tf_value_change()
            
        for i in tpr_inputs.values():
            i.observe(_on_tpr_value_change, "value")
        for i in tf_inputs.values():
            i.observe(_on_tf_value_change, "value")
        
        # Run once to calculate TF fits and plot
        _on_tf_value_change()

        # Only with plotly backend: enable clicking the TPR plot
        if tpr_viewer.backend == "plotly":
            tpr_viewer.get_artist("Testpulses").on_click(_on_tpr_click)
        
        return widgets.HBox([
            widgets.VBox([
                widgets.HBox(list(tpr_inputs.values())),
                widgets.HBox([tpr_viewer.get_figure()])
            ]),
            widgets.VBox([
                widgets.HBox(list(tf_inputs.values())),
                widgets.HBox([tf_viewer.get_figure()])
            ]),
        ])