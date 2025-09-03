from typing import List, Union

import numpy as np
from scipy.integrate import trapezoid

from ..functionbase import FncBaseClass
from ..processing.boxcarsmoothing import BoxCarSmoothing
from .fitbaseline import FitBaseline


class MainParameters(FncBaseClass):
    """
    Calculate main parameters for an event.  These parameters are:

        - **pulse height** (V): Height of the event
        - **peak_position** (samples): Position of the peak.
        - **onset** (samples): Start of the pulse, which is assumed to be where the trace rises to 20% of the pulse height.  This value is shifted relative to 1/4 of the record length, so should be negative for normal pulses.
        - **rise time** (samples): Time from onset to reach 80% of the pulse height.
        - **decay time** (samples): Time (after pulse maximum) from 90% of pulse height to 36.8% (1/e) of pulse height.
        - **baseline slope** (V/samples): Slope of baseline prior to onset, calculated msing a fraction of the record length.
        - **baseline offset** (V): Offset of the baseline, averaged over a fraction of the record length.
        - **baseline difference** (V): Difference between the right and left edges of the trace, averaged over a fraction of the record length.
        - **baseline rms** (V): RMS noise of the baseline over a fraction of the record length.
        - **minimum derivative** (V): Most negative difference between two samples.
        - **minimum derivative index** (sample): index of the minimum derivative.
        - **maximum derivative** (V): Most positive difference between two samples.
        - **maximum derivative index** (sample): index of the maximum derivative.
        - **maximum** (V): Maximum minus minimum for the entire trace.
        - **integral** (V ms): Integral of the trace.
        - **variance** (V²): Variance of the entire trace.

    The time-based parameters (peak position, onset, rise time, decay time) are all also returned converted to ms.
    In addition, some parameters are calculated in the way that **CAT** does. These parameters differ in the following:

        - **onset** (ms or samples): Start of the pulse, found from when the trace is above 3 times the baseline RMS noise.
        - **rise time** (ms or samples): Calculated from 10% to 90% of the pulse height (and indendent of offset).
        - **decay time** (ms or samples): Calculated from 90% to 10% of the pulse height (and indendent of offset).

    :param dt_us: The microsecond time base of the recording.  If provided, relevant output (e.g. rise time) will be in units of seconds. If not provided, these will be in terms of samples.
    :type dt_us: int, optional
    :param peak_loc: Location to search for, or used as, the peak. If a tuple of floats, this is the fraction of the record length in which the peak is searched, e.g. a tuple of ``(0, 1/2)`` searches the first half of the trace, while ``(0, 1)`` searches the whole trace.  If a float, the peak is always assumed to be at that fraction of the record length, e.g. ``1/4`` will take the pulse maximum to be at one quarter of the record length. If an integer, this is the location of the peak in samples.  Defaults to ``(1/5, 2/5)``.
    :type peak_loc: tuple, float, int, optional
    :param bcs: Keyword arguments for :class:`cait.versatile.BoxCarSmoothing`. See its docstring for details.
    :type bcs: dict
    :param fbl: Keyword arguments for :class:`cait.versatile.FitBaseline`. See its docstring for details.
    :type fbl: dict

    :return: Main parameters as described above
    :type: Tuple[np.ndarray]

    .. code-block:: python

        import cait.versatile as vai

        events = vai.MockData().get_event_iterator()

        f = vai.MainParameters(dt_us=events.dt_us)

        # Preview the working of the calculation
        vai.Preview(events, f)

        # Apply function to all events (this returns a tuple of arrays)
        mp = vai.apply(f, events)

        # Generate a directory where the keys correspond to the names
        # of the main parameters and the values to the calculated values
        mp_dict = {k: v for k, v in zip(f.names, mp)}

    .. image:: media/MainParameters_preview.png
    """
    params = (
            ("pulse_height", float),
            ("peak_position", int),
            ("peak_position_rel", float),
            ("onset", int),
            ("onset_ms", float),
            ("rise_time", int),
            ("rise_time_ms", float),
            ("decay_time", int),
            ("decay_time_ms", float),
            ("rms", float),
            ("baseline_slope", float),
            ("baseline_offset", float),
            ("baseline_difference", float),
            ("min_deriv", float),
            ("min_deriv_index", int),
            ("max_deriv", float),
            ("max_deriv_index", int),
            ("maximum", float),
            ("integral", float),
            ("variance", float),
            # CAT parameters
            ("onset_CAT", float),
            ("onset_CAT_ms", float),
            ("rise_time_CAT", float),
            ("rise_time_CAT_ms", float),
            ("decay_time_CAT", float),
            ("decay_time_CAT_ms", float),
            )


    def __init__(
            self,
            dt_us: int = None,
            peak_loc: Union[List[float], List[int], float, int] = [1/5, 2/5],
            bcs = dict(length=50),
            fbl = dict(model=1, where=1/8),
            ):
        super().__init__()

        if not isinstance(peak_loc, (tuple, list, np.ndarray, int, float)):
            raise ValueError("Argument peak_loc must be array-like, int, or "
                             f"float, got {type(peak_loc)}")

        # Check `peak_loc` for proper input
        # Integer check
        if isinstance(peak_loc, int) and peak_loc < 0:
            raise ValueError("Only integers greater than zero are "
                             f"allowed for `peak_loc`, got {peak_loc}")

        # Float check
        elif isinstance(peak_loc, float) and (peak_loc < 0 or peak_loc > 1):
            raise ValueError("Only floats between 0 and 1 are "
                             f"allowed for `peak_loc`, got {peak_loc}")

        # Array-like check
        elif isinstance(peak_loc, (tuple, list, np.ndarray)):
            peak_loc = np.array(peak_loc)

            if len(peak_loc) != 2:
                raise ValueError("If `peak_loc` is array-like, it must have "
                                 f"length 2, got len(peak_loc) == {len(peak_loc)}")

            if np.any(peak_loc < 0):
                raise ValueError("All entries of `peak_loc` must be "
                                 f"greater than 0, got {peak_loc}")

            if not peak_loc[0] < peak_loc[1]:
                raise ValueError("If `peak_loc` is array-like, it must be "
                                 f"strictly increasing, got {peak_loc}")

            # Float checks
            if isinstance(peak_loc[0], float):
                if not np.all((peak_loc >= 0) & (peak_loc <= 1)):
                    raise ValueError("Only arrays of floats between 0 and 1 are "
                                     f"allowed for `peak_loc`, got {peak_loc}")
            # Integer checks
            elif isinstance(peak_loc[0], int):
                if not np.all(peak_loc >= 0):
                    raise ValueError("Only arrays of positive integers are "
                                     f"allowed for `peak_loc`, got {peak_loc}")


        self._dt_us = dt_us
        self._peak_loc = peak_loc

        self._fitbaseline = FitBaseline(**fbl)
        self._smoothing = BoxCarSmoothing(**bcs)


    def __call__(self, event):
        # This prevents the user's event from being modified
        event = np.array(event)
        # Expand dims if the event is 1-D (we can then work on it as though
        # there were more than one event, makes assumptions easier)
        was_single_channel = False
        if event.ndim == 1:
            was_single_channel = True
            event = np.expand_dims(event, 0)

        # Used for calculations.  If dt_us is set, then output has some values
        # with units in terms of seconds (e.g. rise time in [V/s]), otherwise
        # in samples (e.g. [V/sample]).
        _dt = self._dt_us / 1000 if self._dt_us is not None else 1

        par, bl_rms = self._fitbaseline(event)
        bl_rms /= np.sqrt(self._fitbaseline.xdata[self._fitbaseline.where].shape[0])
        bl_rms = bl_rms.reshape(event.shape[:-1])
        bl_offset = self._fitbaseline.model(0, par).reshape(event.shape[:-1])
        # Since the baseline fit may be an arbitrary polynomial or an exponential,
        # approximate the slope using finite difference.
        x1 = self._fitbaseline.xdata[1]
        x0 = self._fitbaseline.xdata[0]
        bl_slope = (self._fitbaseline.model(x1, par) - self._fitbaseline.model(x0, par)) / (x1 - x0)
        bl_slope = bl_slope.reshape(event.shape[:-1])


        # After calculating the baseline parameters, we operate on the baseline-
        # subtracted event.
        event -= self._fitbaseline.model(self._fitbaseline.xdata, par)

        # Calculate parameters which need to be calculated pre-smoothing
        # TODO: choose whether evmax is calculated from the entire trace, or if it
        #   uses self._peak_loc.
        evmax = event.max(axis=-1) - event.min(axis=-1)  # max - min
        diff = np.diff(event, axis=-1)
        max_deriv_ind = diff.argmax(axis=-1)  # index of maximum derivative
        min_deriv_ind = diff.argmin(axis=-1)  # index of minimum derivative
        max_deriv = diff.max(axis=-1) / _dt  # maximum derivative
        min_deriv = diff.min(axis=-1) / _dt  # minimum derivative
        bl_diff = np.mean(event[..., -50:], axis=-1) - np.mean(event[..., :50], axis=-1)  # baseline difference


        # The rest of the parameters are calculated from the smoothed event
        # (unless the `length` kwarg is set to 1)
        event = self._smoothing(event)

        # Pulse height is simply the maximum in the search interval
        if isinstance(self._peak_loc, int):
            self._peak_pos = np.full(event.shape[:-1], self._peak_loc)
            ph = event[..., self._peak_loc]
        elif isinstance(self._peak_loc, float):
            ind = int(np.round(self._peak_loc * event.shape[-1]))
            self._peak_pos = np.full(event.shape[:-1], ind)
            ph = event[..., ind]
        elif isinstance(self._peak_loc, (tuple, list, np.ndarray)):
            # peak_loc is always an array, and its type should already be set
            bounds = np.array(self._peak_loc)
            if bounds.dtype == float:
                bounds = (bounds * event.shape[-1]).astype(int)
            self._peak_pos = np.argmax(event[..., bounds[0]:bounds[1]], axis=-1) + bounds[0]
            index = tuple(np.indices(event.shape[:-1])) + (self._peak_pos,)
            ph = event[index].reshape(event.shape[:-1])

        # Onset is the last sample above 3x the baseline RMS, searching
        # backwards from the peak.
        # ---
        # Rise time is the amount of time (or number of samples) to go from
        # 20% of the pulse height to 80% of the pulse height.
        # DIFFERS FROM CAT: CAT is 10% -> 90%.
        # ---
        # Decay time is the amount of time (or number of samples) to go from
        # 90% of the pulse height to 1/e of the pulse height.
        # DIFFERS FROM CAT: CAT is 90% -> 10%.
        # ---
        _x = np.tile(np.arange(event.shape[-1]), (*event.shape[:-1], 1))
        mask_pp = _x < np.expand_dims(self._peak_pos, -1)
        ph_exp = np.expand_dims(ph, axis=-1)  # Used a lot below

        self._rs = np.argmax(_x * (mask_pp & (event < 0.2*ph_exp)), axis=-1)
        self._re = np.argmax(_x * (mask_pp & (event < 0.8*ph_exp)), axis=-1)
        self._ds = np.argmax(_x * (~mask_pp & (event > 0.9*ph_exp)), axis=-1)
        self._de = np.argmax(_x * (~mask_pp & (event > 1/np.e*ph_exp)), axis=-1)
        self._os = self._rs - event.shape[-1] // 4

        osc = np.argmax(_x * (mask_pp & (event < 3*np.expand_dims(bl_rms, axis=-1))), axis=-1)
        rsc = np.argmax(_x * (mask_pp & (event < 0.1*ph_exp)), axis=-1)
        rec = np.argmax(_x * (mask_pp & (event < 0.9*ph_exp)), axis=-1)
        dsc = self._ds
        dec = np.argmax(_x * (~mask_pp & (event > 0.1*ph_exp)), axis=-1)

        # Integral may be useful in rejecting pileup.
        integral = trapezoid(event, dx=_dt, axis=-1)
        variance = np.var(event, axis=-1)

        # Done this way to be able to mask out the rise/decay times which
        # weren't able to be found.
        rise_time = np.full(self._rs.shape, -1, dtype=float)
        decay_time = np.full(self._rs.shape, -1, dtype=float)
        rcond = (self._rs > 0) & (self._re > 0)
        dcond = (self._ds > 0) & (self._de > 0)
        rise_time[rcond] = (self._re - self._rs)[rcond]
        decay_time[dcond] = (self._de - self._ds)[dcond]

        rise_time_cat = np.full(rsc.shape, -1, dtype=float)
        decay_time_cat = np.full(rsc.shape, -1, dtype=float)
        rcond = (rsc > 0) & (rec > 0)
        dcond = (dsc > 0) & (dec > 0)
        rise_time_cat[rcond] = (rec - rsc)[rcond]
        decay_time_cat[dcond] = (dec - dsc)[dcond]

        out = [
                ph,
                self._peak_pos,
                self._peak_pos * _dt,
                self._os,
                self._os * _dt,
                rise_time,
                rise_time.astype(float) * _dt,
                decay_time,
                decay_time * _dt,
                bl_rms,
                bl_slope,
                bl_offset,
                bl_diff,
                min_deriv,
                min_deriv_ind,
                max_deriv,
                max_deriv_ind,
                evmax,
                integral,
                variance,
                osc,
                osc * _dt,
                rise_time_cat,
                rise_time_cat * _dt,
                decay_time_cat,
                decay_time_cat * _dt,
                ]

        # Return values
        # This should fix batch issues...
        if was_single_channel:
            out = np.squeeze(out)

        return tuple(out)


    @property
    def batch_support(self):
        return 'full'


    def preview(self, event) -> dict:
        unsmoothed = event.copy()
        _ = self(event)
        mp = np.array([self._peak_pos, self._os, self._rs, self._re, self._ds, self._de]).squeeze()

        _dt = self._dt_us if self._dt_us is not None else 1
        x = np.arange(event.shape[-1]) * _dt

        if event.ndim > 1:
            d1 = {}
            d2 = {}
            for i in range(event.shape[0]):
                d1[f"channel {i}"] = [x, unsmoothed[i]]
                d1[f"channel {i} bcs"] = [x, self._smoothing(unsmoothed[i])]
                d2[f"MP channel {i}"] = [x[mp[:, i]], unsmoothed[i][mp[:, i]]]
        else:
            d1 = {'event': [x, event], 'event bcs': [x, self._smoothing(event)]}
            d2 = {'MP': [x[mp], self._smoothing(event)[mp]]}

        return dict(line=d1, scatter=d2, axes=dict(xaxis={"label": "time (ms)" if self._dt_us is not None else "index"}))


    @property
    def names(self):
        return (x[0] for x in MainParameters.params)


    @property
    def types(self):
        return (x[1] for x in MainParameters.params)
