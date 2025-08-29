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
    :type bcs: dict

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
            peak_loc: Union[List[float], float, int] = [1/5, 2/5],
            bcs = dict(length=50),
            fbl = dict(model=1, where=1/8),
            ):
        super().__init__()

        if not isinstance(peak_loc, (tuple, list, np.ndarray, int, float)):
            raise ValueError("Argument peak_loc must be array-like, int, or "
                             f"float, got {type(peak_loc)}")

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

        orig_shape = None
        if event.ndim == 3:
            orig_shape = event.shape
            event = event.reshape(-1, event.shape[-1])

        # Used for calculations.  If dt_us is set, then output has some values
        # with units in terms of seconds (e.g. rise time in [V/s]), otherwise
        # in samples (e.g. [V/sample]).
        _dt = self._dt_us / 1000 if self._dt_us is not None else 1

        par, bl_rms = self._fitbaseline(event)
        bl_rms /= np.sqrt(self._fitbaseline.xdata[self._fitbaseline.where].shape[0])
        bl_offset = self._fitbaseline.model(0, par)
        # Since the baseline fit may be an arbitrary polynomial or an exponential,
        # approximate the slope using finite difference.
        x1 = self._fitbaseline.xdata[1]
        x0 = self._fitbaseline.xdata[0]
        bl_slope = (self._fitbaseline.model(x1, par) - self._fitbaseline.model(x0, par)) / (x1 - x0)


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
            self._peak_pos = np.full(event.shape[0], self._peak_loc)
            ph = event[..., self._peak_loc]
        elif isinstance(self._peak_loc, float):
            self._peak_pos = np.full(event.shape[0], int(np.round(self._peak_loc * event.shape[-1])))
            ph = event[..., self._peak_pos[0]]
        else:
            self._peak_pos = np.argmax(event, axis=-1)
            ph = event.max(axis=-1)

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
        # Note we do this in a loop instead of cutting, so that we catch the
        # FIRST instance of each case (i.e. avoids issues with e.g. pileup).
        self._os = -1 * np.ones(event.shape[:-1], dtype=int)  # rise start
        self._rs = -1 * np.ones(event.shape[:-1], dtype=int)  # rise start
        self._re = -1 * np.ones(event.shape[:-1], dtype=int)  # rise end
        self._ds = -1 * np.ones(event.shape[:-1], dtype=int)  # decay start
        self._de = -1 * np.ones(event.shape[:-1], dtype=int)  # decay end

        # CAT parameters
        osc = -1 * np.ones(event.shape[:-1], dtype=int)  # onset
        rsc = -1 * np.ones(event.shape[:-1], dtype=int)  # onset
        rec = -1 * np.ones(event.shape[:-1], dtype=int)  # onset
        dsc = -1 * np.ones(event.shape[:-1], dtype=int)  # onset
        dec = -1 * np.ones(event.shape[:-1], dtype=int)  # onset


        # Create mask of correct shape
        # Two cases:
        #  - event.ndim == 2: add last dimension
        #  - event.ndim == 3: add first and last dimension
        for ichan in range(event.shape[0]):
            _x = np.arange(event.shape[-1])
            mask_pp = _x < self._peak_pos[ichan]

            _rs = np.where(mask_pp & (event[ichan] < 0.2*ph[ichan]))
            _re = np.where(mask_pp & (event[ichan] < 0.8*ph[ichan]))
            _ds = np.where(~mask_pp & (event[ichan] > 0.9*ph[ichan]))
            _de = np.where(~mask_pp & (event[ichan] > 1/np.e*ph[ichan]))

            if len(_rs[0]):
                self._rs[ichan] = _rs[0][-1]
            if len(_re[0]):
                self._re[ichan] = _re[0][-1]
            if len(_ds[0]):
                self._ds[ichan] = _ds[0][-1]
            if len(_de[0]):
                self._de[ichan] = _de[0][-1]

            self._os[ichan] = (self._rs[ichan] - event.shape[-1]//4)

            _os = np.where(mask_pp & (event[ichan] < 3*bl_rms[ichan]))
            _rs = np.where(mask_pp & (event[ichan] < 0.1*ph[ichan]))
            _re = np.where(mask_pp & (event[ichan] < 0.9*ph[ichan]))
            _ds = np.where(~mask_pp & (event[ichan] > 0.9*ph[ichan]))
            _de = np.where(~mask_pp & (event[ichan] > 0.1*ph[ichan]))

            if len(_os[0]):
                osc[ichan] = _os[0][-1]
            if len(_rs[0]):
                rsc[ichan] = _rs[0][-1]
            if len(_re[0]):
                rec[ichan] = _re[0][-1]
            if len(_ds[0]):
                dsc[ichan] = _ds[0][-1]
            if len(_de[0]):
                dec[ichan] = _de[0][-1]

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
        if orig_shape is not None:
            out = np.array(out).reshape(len(out), -1, orig_shape[1])
        if was_single_channel:
            out = np.squeeze(out)
        return tuple(out)


    @property
    def batch_support(self):
        return 'full'


    def preview(self, event) -> dict:
        unsmoothed = event.copy()
        _ = self(event)
        mp = np.array([self._peak_pos, self._os, self._rs, self._re, self._ds, self._de])

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
