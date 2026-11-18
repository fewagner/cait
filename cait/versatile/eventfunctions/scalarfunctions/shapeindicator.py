import numpy as np
from numpy.fft import rfft
from scipy.interpolate import CubicSpline

from ..functionbase import ScalarFncBaseclass
from ..processing.optimumfiltering import OptimumFiltering
from ..processing.tukey import TukeyWindow


class ShapeIndicator(ScalarFncBaseclass):
    """
    Calculate the Shape Indicator (SI) of an event.

    The SI is the chi-squared per degree of freedom (chi2/ndf) of a fit of a cubic spline of the
    filtered standard event to the optimum-filtered event, evaluated in a window of 4 FWHM around
    the event peak, as defined in Di Domizio, Orio & Vignati, JINST 6 (2011) P02007, Eq. 6.1-6.2.
    Like in the paper, it has the form of a chi-squared but does not follow a chi-squared
    distribution, because the noise of the filtered trace is correlated:

    .. math::

        SI = \\sum_{i=0}^{L-1} \\frac{(y^f_i - f_i)^2}{\\sigma_L^2 (L - 2)}

    where :math:`y^f` is the filtered event, :math:`f` is the best-fit template
    (amplitude fitted in :math:`[A, A(1+\\epsilon)]`, position scanned in
    :math:`[-1, +1]` samples), :math:`L` is 4 times the FWHM of the filtered standard
    event (measured on its rising edge) and :math:`\\sigma_L^2` is the noise variance in
    a window of length :math:`L` (Eq. 6.2). Events shaped like the standard event
    give a low SI, non-physical events (spikes, vibrations, pile-up) give a much larger
    SI. The scale depends on the setup (see the note below).

    The function only supports single-channel events and expects events with their
    baseline removed (e.g. ``it.with_processing(vai.RemoveBaseline())``).

    :param of: The optimum filter of the channel, shape ``(M//2+1,)``.
    :type of: np.ndarray
    :param sev: The standard event of the channel, shape ``(M,)``.
    :type sev: np.ndarray
    :param nps: The noise power spectrum of the channel, shape ``(M//2+1,)``, as stored
        by :class:`cait.versatile.NPS`: a one-sided power spectral density in V^2/Hz. It is
        converted internally to the noise power per bin used in Eqs. 3.2 and 6.2.
    :type nps: np.ndarray
    :param window: If True, a Tukey window is applied to events (and the standard
        event) before filtering. Defaults to False.
    :type window: bool, optional
    :param dt_us: The microsecond timebase of the recording. Needed for the conversion of
        the NPS. If None, it is read from ``nps`` (which works if ``nps`` is a
        :class:`cait.versatile.NPS` object). Defaults to None.
    :type dt_us: int, optional

    :return: The Shape Indicator. NaN if the fit window exceeds the record window.
    :rtype: float

    .. note::
        The noise variance :math:`\\sigma_L^2` only includes the frequency bins
        :math:`k \\geq M/L`. If the record is much longer than the fit window (large
        :math:`M/L`), :math:`\\sigma_L^2` covers only part of the filtered-noise variance
        and the SI of pure noise lies above 1 (e.g. about 1.5 for
        :math:`M/L \\approx 100\\text{-}250`).
        Choose cuts from the SI distribution of your own clean events rather than
        from the value 1.

    **Example:**

    .. code-block:: python

        import cait.versatile as vai

        md = vai.MockData()
        it = md.get_event_iterator()[0].with_processing(vai.RemoveBaseline())

        f = vai.ShapeIndicator(of=md.of[0], sev=md.sev[0], nps=md.nps[0], dt_us=md.dt_us)

        # Calculate the SI of all events
        si = vai.apply(f, it)

        # Inspect the fits
        vai.Preview(it, f)
    """
    _outputs = [("shape_indicator", float)]

    def __init__(
        self,
        of: np.ndarray,
        sev: np.ndarray,
        nps: np.ndarray,
        window: bool = False,
        dt_us: int = None,
    ):
        # vai.NPS objects carry their timebase, so it is read before nps is cast to an array.
        if dt_us is None:
            dt_us = getattr(nps, "dt_us", None)
        if dt_us is None:
            raise ValueError(
                "The timebase of the NPS is needed to convert it from V^2/Hz to the noise "
                "power per bin: pass a cait.versatile.NPS object as 'nps' or give 'dt_us'."
            )
        self._dt_us = dt_us

        self._of = np.array(of)
        self._sev = np.array(sev, dtype=float)
        self._nps = np.array(nps, dtype=float)
        self._window = window

        # Variable names (h, L, sigma2_L, eps, k) follow the notation of the paper.
        n_samples = self._sev.shape[-1]
        if self._sev.ndim != 1 or self._of.ndim != 1 or self._nps.ndim != 1:
            raise ValueError(
                "ShapeIndicator supports single-channel input only: "
                "'of', 'sev' and 'nps' must be 1d."
            )
        if self._of.shape[-1] != n_samples // 2 + 1 or self._nps.shape[-1] != n_samples // 2 + 1:
            raise ValueError(
                f"Shape mismatch: for a SEV with {n_samples} samples, 'of' and 'nps' must have "
                f"{n_samples // 2 + 1} entries. "
                f"Got {self._of.shape[-1]} and {self._nps.shape[-1]}."
            )
        self._n_samples = n_samples

        self._optimum_filtering = OptimumFiltering(self._of)
        self._tukey = TukeyWindow() if window else None

        sev_filtered = self._filter(self._sev)

        # Normalisation constant h (Eq. 3.2). The paper does not fix the normalisation of the
        # DFT or of the noise power N. Here, N is the two-sided noise power per bin, which
        # sums to the noise variance (vai.NPS stores a one-sided PSD in V^2/Hz, hence
        # N = PSD * fs / (2 M)), and s is the DFT divided by the number of samples M, so that
        # h equals the variance of the filtered noise (Eq. 3.4). The sums run over the full
        # two-sided spectrum, as in Eq. 6.2 (k = M/L ... M - M/L). For a real signal this is
        # twice the one-sided sum (DC and Nyquist count once).
        sampling_frequency = 1e6 / dt_us
        sev_fft_norm = rfft(self._sev) / n_samples
        nps_per_bin = self._nps * sampling_frequency / (2 * n_samples)
        # |s|^2 / N for k >= 1 (DC is excluded)
        ratio = np.zeros(len(nps_per_bin))
        ratio[1:] = np.abs(sev_fft_norm[1:]) ** 2 / nps_per_bin[1:]

        def two_sided_sum(k_start):
            # sum of |s|^2 / N over k = k_start ... M - k_start (positive and negative frequencies)
            return 2 * np.sum(ratio[k_start:n_samples // 2]) + ratio[n_samples // 2]

        h = 1.0 / two_sided_sum(1)

        # Fit window L = 4 * FWHM of the filtered standard event. The FWHM is measured
        # on the rising edge only (and mirrored), because the decay tail of the filtered
        # SEV can lead to a spuriously large estimate.
        self._se_peak_pos = int(np.argmax(sev_filtered))
        se_peak_val = float(sev_filtered[self._se_peak_pos])
        left_idx = np.where(sev_filtered[:self._se_peak_pos] < se_peak_val / 2.0)[0]
        left_cross = left_idx[-1] if len(left_idx) > 0 else 0
        fwhm = 2 * (self._se_peak_pos - left_cross)
        self._fit_window = min(4 * fwhm, n_samples)
        self._half_window = self._fit_window // 2

        # Noise variance in a window of length L (Eq. 6.2): sum over the frequencies
        # that are visible in a window of that length.
        k_low = max(1, int(round(n_samples / self._fit_window)))
        self._sigma2_L = float(h ** 2 * two_sided_sum(k_low))

        # Continuous template (allows sub-sample evaluation) and maximum relative
        # amplitude spread due to digitisation (difference between the peak sample
        # and its neighbour).
        self._spline = CubicSpline(np.arange(n_samples), sev_filtered)
        self._eps = abs(sev_filtered[self._se_peak_pos]
                        - sev_filtered[min(self._se_peak_pos + 1, n_samples - 1)]) / se_peak_val

        self._reset_state()

    def _filter(self, trace: np.ndarray) -> np.ndarray:
        # Optional Tukey window followed by optimum filtering.
        if self._tukey is not None:
            trace = self._tukey(trace)
        return self._optimum_filtering(trace)

    def _reset_state(self):
        # Intermediate results of the last call (used by preview).
        self._filtered_event = None
        self._window_slice = None
        self._best_fit = None
        self._si = np.nan

    def __call__(self, event: np.ndarray) -> float:
        self._reset_state()

        event = np.asarray(event, dtype=float)
        if event.ndim != 1:
            raise ValueError(
                "ShapeIndicator supports single-channel events only. "
                "Select a channel first, e.g. 'iterator[0]'."
            )

        self._filtered_event = self._filter(event)

        # Locate the event peak, excluding the outer 1/8 of the record on each side
        # to avoid filter edge effects.
        search_start = int(self._n_samples / 8)
        search_end = self._n_samples - search_start
        peak_pos = int(np.argmax(self._filtered_event[search_start:search_end])) + search_start
        of_amplitude = float(self._filtered_event[peak_pos])

        # Fit window of length L centred on the event peak.
        window_start = peak_pos - self._half_window
        window_end = window_start + self._fit_window
        if window_start < 0 or window_end > self._n_samples:
            return self._si

        self._window_slice = slice(window_start, window_end)
        event_window = self._filtered_event[self._window_slice]

        # Scan sub-sample position shifts in [-1, +1] samples (21 steps of 0.1 samples)
        # and keep the one that minimises the chi-squared. This removes digitisation
        # artefacts from the SI.
        best_chi2 = np.inf
        for position_shift in np.linspace(-1.0, 1.0, 21):
            sample_positions = (self._se_peak_pos
                                + (np.arange(self._fit_window) - self._half_window)
                                + position_shift)
            template = self._spline(sample_positions)

            template_norm_sq = np.dot(template, template)
            if template_norm_sq == 0:
                continue

            # Least-squares amplitude, clamped to [A, A*(1+eps)] as in the paper.
            amplitude_fit = np.clip(
                np.dot(event_window, template) / template_norm_sq,
                of_amplitude,
                of_amplitude * (1.0 + self._eps),
            )

            residuals = event_window - amplitude_fit * template
            chi2 = np.sum(residuals ** 2) / (self._sigma2_L * (self._fit_window - 2))

            if chi2 < best_chi2:
                best_chi2 = chi2
                self._best_fit = amplitude_fit * template

        self._si = float(best_chi2)
        return self._si

    @property
    def batch_support(self) -> str:
        return 'none'

    def preview(self, event: np.ndarray) -> dict:
        si = self(event)

        # Preview keeps every trace whose name is new and only updates traces whose name already
        # exists. Trace names must therefore stay constant between events (the SI goes into
        # the x-axis label, which Preview leaves untouched).
        if self._window_slice is None:
            # Fit window outside of the record: show the filtered event, blank the fit traces.
            x = np.arange(self._n_samples)
            blank = np.full(self._n_samples, np.nan)
            return dict(
                scatter={'filtered event': [x, self._filtered_event],
                         'residuals (template - data)': [x, blank]},
                line={'best-fit template': [x, blank]},
                axes={"xaxis": {"label": "Sample index (SI = NaN)"},
                      "yaxis": {"label": "Filtered amplitude"}},
            )

        x = np.arange(self._window_slice.start, self._window_slice.stop)
        event_window = self._filtered_event[self._window_slice]
        return dict(
            scatter={'filtered event': [x, event_window],
                     'residuals (template - data)': [x, self._best_fit - event_window]},
            line={'best-fit template': [x, self._best_fit]},
            axes={"xaxis": {"label": f"Sample index (SI = {si:.2f})"},
                  "yaxis": {"label": "Filtered amplitude"}},
        )