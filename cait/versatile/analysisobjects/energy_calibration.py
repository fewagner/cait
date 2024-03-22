from typing import List

import numpy as np

from ...calibration import PulserModel
from ..plot import Viewer

class ECal:
    """
    Object for Energy Calibration of a detector. In the constructor `timestamps`, `testpulse_amplitudes` and `pulse_heights` are used to fit the drifts of each testpulse amplitude. Calling the object afterwards lets you predict the testpulse-amplitude-equivalent for given pulse heights at given times.

    :param timestamps: The microsecond timestamps of the testpulses.
    :type timestamps: List[int]
    :param testpulse_amplitudes: The testpulse amplitude values of the testpulses.
    :type testpulse_amplitudes: List[float]
    :param pulse_heights: The recorded pulse heights of the testpulses.
    :type pulse_heights: List[float]
    :param start_saturation: Test pulses with average pulse heights above this value are excluded from the fit. Note that for *predicting* testpulse-amplitude-equivalents later, this does not apply, i.e. they are still extrapolated and it is the user's responsibility to account for saturation effects in these pulse heights.
    :type start_saturation: float
    :param exclude_tpas: List of testpulse amplitudes to exclude from the fit. Defaults to [], i.e. no exclusions.
    :type exclude_tpas: List[float], optional
    :param max_dist: The maximal distance in hours between two test pulses such that they are included still in the same regression. For testpulses further apart a new regression is started. Defaults to 0.5, i.e. half an hour.
    :type max_dist: float, optional
    :param fit_method: Interpolation method for fitting testpulses. Either of ['linear', 'tree']. Defaults to 'linear'.
    :type fit_method: str, optional
    :param predict_method: Method for predicting testpulse-equivalent-amplitudes in the TPA/PH plane. Options are ('polynomial', <order of polynomial>) and ('interpolation', <kind of interpolation>), where <kind of interpolation> is passed on to `scipy.interpolate.interp1d` and can be any of ['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']. Defaults to ('interpolation', 'linear').
    :type predict_method: tuple, optional
    """
    def __init__(self,
                 timestamps: List[int],
                 testpulse_amplitudes: List[float],
                 pulse_heights: List[float] ,
                 start_saturation: float,
                 exclude_tpas: List[float] = [],
                 max_dist: float = 0.5,
                 fit_method: str = 'linear',
                 predict_method: tuple = ('interpolation', 'linear')
                 ):
        
        if fit_method not in ["linear", "tree"]: 
            raise NotImplementedError(f"Unsupported fit method '{fit_method}'. Choose either of ['linear', 'tree'].")
        if type(predict_method) is not tuple:
            raise TypeError(f"'predict_method' has to be a tuple, not {type(predict_method)}.")
        if predict_method[0] not in ['polynomial', 'interpolation']:
            raise NotImplementedError(f"Unsupported predict method '{predict_method[0]}'. Choose either of ['polynomial', 'interpolation'].")
        if predict_method[0] == "polynomial" and type(predict_method[1]) is not int:
            raise TypeError(f"Predict method 'polynomial' requires an integer (the order of the polynomial) as a second argument, not {type(predict_method[1])}")
        if predict_method[0] == "interpolation" and predict_method[1] not in ['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']:
            raise ValueError(f"Predict method 'interpolation' requires either of ['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'] as a second argument, not {predict_method[1]}.")
        
        # Save first testpulse timestamp to have identical hours-axis later for predicting
        self.first_tp_timestamp = np.min(timestamps)
        # Convert timestamps to hours
        hours = (timestamps - np.min(timestamps))/1e6/3600

        self.pulser_model = PulserModel(start_saturation=start_saturation, max_dist=max_dist)
        self.pulser_model.fit(tphs=pulse_heights, 
                              tpas=testpulse_amplitudes, 
                              tp_hours=hours, 
                              exclude_tpas=exclude_tpas, 
                              interpolation_method=fit_method)
        
        self.predict_method = predict_method
        
    def __call__(self, timestamps: List[int], pulse_heights: List[float]):
        """
        Predict testpulse-amplitude-equivalents for given `pulse_heights` recorded at `timestamps`.

        :param timestamps: The microsecond timestamps of the pulses.
        :type timestamps: List[int]
        :param pulse_heights: The recorded pulse heights.
        :type pulse_heights: List[float]

        :return: Tuple of testpulse-amplitude-equivalents and their 1-sigma-uncertainties
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # Error handling for methods other than 'polynomial' and 'interpolation' is done in __init__
        if self.predict_method[0] == "polynomial":
            kwargs = {"use_interpolation": False, "poly_order": self.predict_method[1]}
        elif self.predict_method[0] == "interpolation":
            # poly_order has no effect here but is a required argument of predict anyways
            kwargs = {"use_interpolation": True, "kind": self.predict_method[1], "poly_order": 1}

        # Convert timestamps to hours
        hours = (np.array(timestamps) - self.first_tp_timestamp)/1e6/3600

        _, _, tpa_equivalent, tpa_equivalent_sigma = self.pulser_model.predict(evhs=pulse_heights, 
                                                                               ev_hours=hours,
                                                                               **kwargs)
        
        return tpa_equivalent, tpa_equivalent_sigma
    
    def show_regression(self, **kwargs):
        """
        Plot the result of the regression.

        :param kwargs: Keyword arguments passed on to `cait.versatile.Viewer`.
        :type kwargs: Any
        """

        scatter_x = self.pulser_model.tp_hours
        scatter_y = self.pulser_model.tphs

        fit_x, fit_y, fit_sigma_x, fit_sigma_y = [], [], [], []

        if self.pulser_model.interpolation_method == "linear":
            markers_x = list(set([x_pos for interval in self.pulser_model.intervals for x_pos in interval]))

            for i, iv in enumerate(self.pulser_model.intervals):
                t = np.linspace(iv[0], iv[1], 100)
                for m in range(len(self.pulser_model.all_linear_tpas[i])):
                    lower, y, upper = self.pulser_model.all_regs[i][m].y_sigma(t)

                    fit_x += list(t) + [None]
                    fit_y += list(y) + [None]
                    fit_sigma_x += list(t) + [None] + list(t) + [None]
                    fit_sigma_y += list(lower) + [None] + list(upper) + [None]

        elif self.pulser_model.interpolation_method == "tree":
            markers_x = None

            t = np.linspace(0, self.pulser_model.tp_hours[-1], 100)
            for m in range(len(self.pulser_model.linear_tpas)):
                lower = self.pulser_model.lower_regs[m].predict(t.reshape(-1, 1))
                y = self.pulser_model.mean_regs[m].predict(t.reshape(-1, 1))
                upper = self.pulser_model.upper_regs[m].predict(t.reshape(-1, 1))

                fit_x += list(t) + [None]
                fit_y += list(y) + [None]
                fit_sigma_x += list(t) + [None] + list(t) + [None]
                fit_sigma_y += list(lower) + [None] + list(upper) + [None]

        viewer = Viewer(**kwargs)
        viewer.add_scatter(x=scatter_x, y=scatter_y, name='testpulses')
        viewer.add_line(fit_x, fit_y, "regression")
        viewer.add_line(fit_sigma_x, fit_sigma_y, "$1\sigma$") 
        if markers_x is not None: viewer.add_vmarker(markers_x, (np.min(scatter_y), np.max(scatter_y)))

        viewer.set_xlabel("Time (h)")
        viewer.set_ylabel("Pulse Height (V)")
        viewer.show()
        
        return viewer
    
    def show_plane(self, hours: float = 0.5, **kwargs):
        """
        Plot the TPA-PH plane for a given time. If no time is specified, the middle of the first testpulse section is used (If there are no gaps in the testpulses, this amounts to the middle of the entire time range).

        :param hours: The time (in hours after first timestamp) for which to plot the polynomial.
        :type hours: float
        :param kwargs: Keyword arguments passed on to `cait.versatile.Viewer`.
        :type kwargs: Any
        """
        phs = np.linspace(0, self.pulser_model.start_saturation, 100)
        timestamps = np.ones_like(phs)*int(self.first_tp_timestamp + hours*3600*1e6)
        tpas, tpa_sigmas = self(timestamps, phs)

        fit_sigma_x = list(phs) + [None] + list(phs)
        fit_sigma_y = list(tpas-tpa_sigmas) + [None] + list(tpas+tpa_sigmas)

        viewer = Viewer(**kwargs)
        viewer.add_line(phs, tpas, "mean")
        viewer.add_line(fit_sigma_x, fit_sigma_y, "$1\sigma$")

        viewer.set_xlabel("Pulse Height (V)")
        viewer.set_ylabel("Testpulse Amplitude (V)")
        viewer.show()
        
        return viewer