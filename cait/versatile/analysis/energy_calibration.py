from typing import List

import numpy as np

from ...calibration import PulserModel

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
    :param interpolation_method: Interpolation method. Either of ['linear', 'tree']. Defaults to 'linear'.
    :type interpolation_method: str, optional
    """
    def __init__(self,
                 timestamps: List[int],
                 testpulse_amplitudes: List[float],
                 pulse_heights: List[float] ,
                 start_saturation: float,
                 exclude_tpas: List[float] = [],
                 max_dist: float = 0.5,
                 interpolation_method: str = 'linear',
                 ):
        # Save first testpulse timestamp to have identical hours-axis later for predicting
        self.first_tp_timestamp = np.min(timestamps)
        # Convert timestamps to hours
        hours = (timestamps - np.min(timestamps))/1e6/3600

        self.pulser_model = PulserModel(start_saturation=start_saturation, max_dist=max_dist)
        self.pulser_model.fit(tphs=pulse_heights, 
                              tpas=testpulse_amplitudes, 
                              tp_hours=hours, 
                              exclude_tpas=exclude_tpas, 
                              interpolation_method=interpolation_method)
        
    def __call__(self, timestamps: List[int], pulse_heights: List[float], poly_order: int = 1):
        """
        Predict testpulse-amplitude-equivalents for given `pulse_heights` recorded at `timestamps`.

        :param timestamps: The microsecond timestamps of the pulses.
        :type timestamps: List[int]
        :param pulse_heights: The recorded pulse heights.
        :type pulse_heights: List[float]
        :param poly_order: The order of the polynomial to fit in the TPA/PH plane. Defaults to 1.
        :type poly_order: int, optional

        :return: Tuple of testpulse-amplitude-equivalents and their 1-sigma-uncertainties
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # Convert timestamps to hours
        hours = (np.array(timestamps) - self.first_tp_timestamp)/1e6/3600

        _, _, tpa_equivalent, tpa_equivalent_sigma = self.pulser_model.predict(evhs=pulse_heights, 
                                                                               ev_hours=hours,
                                                                               poly_order=poly_order)
        
        return tpa_equivalent, tpa_equivalent_sigma