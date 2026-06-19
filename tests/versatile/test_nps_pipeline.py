"""
Step-by-step guide: How get_clean_bs_idx works
===============================================

GOAL: Build synthetic ADC stream data from scratch, then walk through each
sub-function of the NPS baseline-selection pipeline so you can understand
what happens at every stage before running it on real data.

BACKGROUND
----------
A cait "stream" is a long 1-D voltage array sampled at constant rate (e.g.
25 kHz).  To compute the Noise Power Spectrum (NPS) we need baseline records
— fixed-length windows that are free of:

  * detector pulses (particle events, test pulses)
  * level shifts    (DC jumps from environmental disturbances)
  * decaying tails  (exponential afterglow from a recent large pulse)

get_clean_bs_idx finds the starting indices of such windows automatically.
It does so in five sequential steps:

  1. level_shift_detector       – splits stream at DC jumps
  2. apply_fourier_trigger       – discards regions around large pulses (FFT)
  3. apply_mean_trigger          – discards regions around small pulses (z-score)
  4. divide_array                – chops clean intervals into record-length blocks
  5. decaying_baseline_remover   – drops blocks whose start/end means differ

Run this file with:
    python test_nps_pipeline.py

It prints a summary of each step and saves test_nps_pipeline.png.
"""

# NOTE: on macOS, multiprocessing uses 'spawn' by default.
# All Pool calls must live inside  if __name__ == '__main__':
if __name__ == '__main__':

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')           # change to 'TkAgg' / 'Qt5Agg' for live windows
    import matplotlib.pyplot as plt

    from cait.versatile.functions.nps_auto.get_clean_bs_idx import (
        level_shift_detector,
        apply_fourier_trigger,
        apply_mean_trigger,
        divide_array,
        decaying_baseline_remover,
        get_clean_bs_idx,
    )

    # =========================================================
    # PARAMETERS
    # =========================================================
    SAMPLE_FREQ    = 25_000        # Hz  (standard VDAQ2 rate)
    DT_US          = int(1e6 / SAMPLE_FREQ)   # 40 µs / sample
    RECORD_LENGTH  = 2**15         # 32 768 samples ≈ 1.31 s per baseline record
    NOISE_SIGMA    = 0.003         # V   (typical phonon-channel baseline noise)

    STREAM_LENGTH  = 3_000_000     # samples  ≈ 120 s total

    # Pulse parameters – several amplitudes to probe trigger sensitivity
    #   large  (1.0 V  ≈ 333 σ) – Fourier trigger should catch these
    #   medium (0.1 V  ≈  33 σ) – mean trigger should catch these
    #   small  (0.02 V ≈   7 σ) – just above 6 σ threshold, borderline
    PULSE_AMPS     = [1.0, 0.1, 0.02]
    PULSE_TAU      = 5_000         # samples (τ ≈ 0.2 s at 25 kHz)
                                   # pulse decays to noise level after ~5τ = 25 000 samples

    PULSE_SPACING  = 200_000       # samples between consecutive pulse peaks

    # Level-shift parameters – a sudden DC offset halfway through the stream
    LEVEL_SHIFT_V  = 0.05          # V   (DC step, >> noise)
    LEVEL_SHIFT_I  = 1_500_000     # sample index where the shift occurs

    rng = np.random.default_rng(42)

    print("=" * 60)
    print("NPS Pipeline Step-by-Step Tutorial")
    print("=" * 60)
    print(f"  Sample rate  : {SAMPLE_FREQ} Hz  →  dt = {DT_US} µs")
    print(f"  Stream length: {STREAM_LENGTH:,} samples = {STREAM_LENGTH/SAMPLE_FREQ:.0f} s")
    print(f"  Record length: {RECORD_LENGTH} samples = {RECORD_LENGTH/SAMPLE_FREQ*1e3:.1f} ms")
    print(f"  Noise σ      : {NOISE_SIGMA} V")
    print(f"  Pulse amps   : {PULSE_AMPS} V  (τ = {PULSE_TAU} samples, {PULSE_TAU/SAMPLE_FREQ:.2f} s)")
    print(f"  Level shift  : +{LEVEL_SHIFT_V} V at sample {LEVEL_SHIFT_I:,}")


    # =========================================================
    # STEP 0 – Build the mock stream
    # =========================================================
    # The mock stream consists of three superimposed components:
    #
    #   (a) Gaussian white noise  – the detector baseline
    #   (b) Exponential pulses    – particle events we want to EXCLUDE
    #   (c) A DC step             – level shift we want to detect and split at
    #
    # In a real measurement this array would be read from a .csmpl binary file
    # (raw uint16 ADC counts converted to Volts).  Here we build it directly
    # as float64 so we can skip the file I/O and focus on the algorithm.
    print("\n--- Step 0: Building mock stream ---")

    stream = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=STREAM_LENGTH)

    # Inject level shift in the second half
    stream[LEVEL_SHIFT_I:] += LEVEL_SHIFT_V

    # Pulse peaks are offset by 150 k from multiples of 300 k so they do not
    # land inside the coarse scan windows used by level_shift_detector (which
    # steps in 300 k increments).  This keeps the level-shift detection clean.
    # Amplitudes cycle through PULSE_AMPS so each height is represented evenly.
    positions = (
        list(range(150_000, LEVEL_SHIFT_I,            PULSE_SPACING)) +
        list(range(LEVEL_SHIFT_I + 150_000,
                   STREAM_LENGTH - 5 * PULSE_TAU,     PULSE_SPACING))
    )
    pulses = [(p, PULSE_AMPS[i % len(PULSE_AMPS)]) for i, p in enumerate(positions)]

    for p, amp in pulses:
        length = min(5 * PULSE_TAU, STREAM_LENGTH - p)
        t      = np.arange(length, dtype=float)
        stream[p : p + length] += amp * np.exp(-t / PULSE_TAU)

    print(f"  Injected {len(pulses)} pulses (cycling through amplitudes {PULSE_AMPS} V):")
    for p, amp in pulses:
        print(f"    {p:>9,}  A = {amp:.2f} V  (tail ends ~{p + 5*PULSE_TAU:,})")
    print(f"  Level shift at sample {LEVEL_SHIFT_I:,}")
    print(f"  Clean gap between pulse tail and next pulse: "
          f"~{PULSE_SPACING - 5*PULSE_TAU:,} samples = "
          f"~{(PULSE_SPACING - 5*PULSE_TAU)/RECORD_LENGTH:.1f} record lengths")


    # =========================================================
    # STEP 1 – level_shift_detector
    # =========================================================
    # HOW IT WORKS
    # ------------
    # Pass 1 (coarse, step = 300 000 samples):
    #   Load a 32 000-sample window at each step; compute np.median(window).
    #   Keep a rolling buffer of the 5 most recent medians and record their
    #   std.  A DC jump causes the buffer std to spike sharply.
    #   scipy.signal.find_peaks identifies these spikes.
    #
    # Pass 2 (fine, step = 8 000 samples):
    #   Zoom into ±800 000 samples around each detected anomaly and find
    #   the exact sample where the median std is largest.
    #
    # OUTPUT: list of (start, end) tuples for level-shift-free segments
    #         that are ≥ record_length samples long.
    print("\n--- Step 1: level_shift_detector ---")
    intervals_ls = level_shift_detector(stream=stream, record_length=RECORD_LENGTH)
    print(f"  Found {len(intervals_ls)} level-shift-free interval(s):")
    for a, b in intervals_ls:
        print(f"    [{a:>9,} – {b:>9,}]  "
              f"duration = {(b-a)/SAMPLE_FREQ:.1f} s  "
              f"(~{(b-a)//RECORD_LENGTH} record lengths)")


    # =========================================================
    # STEP 2 – apply_fourier_trigger
    # =========================================================
    # HOW IT WORKS
    # ------------
    # For each clean interval, a short FFT window slides through in steps
    # ('stepsize').  Only the frequency components ≤ 25 Hz are summed —
    # at 25 kHz this effectively computes the local DC mean of each window.
    #
    # Why Fourier? The dc-mean time series is very smooth for pure noise
    # but jumps sharply when a large pulse enters the window.  Computing
    # the first difference ("delta stream") turns that jump into a spike.
    # search_deltastream applies a mean ± sigma*std trigger on that spike
    # and records a guard region of ±32 768 samples around each trigger.
    #
    # Note: the window_size and stepsize are automatically scaled by
    #   dyn_factor = sample_freq / 50 000
    # so the same defaults work across different ADC rates.
    # At 25 kHz: dyn_factor = 0.5, window_size → 150 samples.
    #
    # OUTPUT: list of (start, end) tuples for pulse-free sub-intervals.
    print("\n--- Step 2: apply_fourier_trigger ---")
    intervals_fourier = apply_fourier_trigger(
        stream=stream,
        tuples=intervals_ls,
        dt_us=DT_US,
        n_cores=1,          # 1 core keeps things simple for a tutorial
        sigma=8,
        window_size=300,
        window_size_mean=5,
        stepsize=300,
        record_length=RECORD_LENGTH,
    )
    print(f"  Intervals after Fourier trigger: {len(intervals_fourier)}")
    for a, b in intervals_fourier[:8]:
        print(f"    [{a:>9,} – {b:>9,}]  ({(b-a)/SAMPLE_FREQ:.2f} s)")
    if len(intervals_fourier) > 8:
        print(f"    … and {len(intervals_fourier)-8} more")


    # =========================================================
    # STEP 3 – apply_mean_trigger  (z-score trigger)
    # =========================================================
    # HOW IT WORKS
    # ------------
    # A second, finer pass using a moving z-score over 'window_size' samples
    # (default 1500).  The z-score of sample i is:
    #
    #   z[i] = (x[i] - rolling_mean[i]) / rolling_std[i]
    #
    # If |z[i]| > sigma (default 6), a pulse is declared and the surrounding
    # 32 768 + 8 000 samples are discarded.
    #
    # This step catches events that produce only a modest DC shift (fast
    # rise, or small amplitude) that the Fourier trigger might miss.
    #
    # OUTPUT: list of (start, end) tuples, similar to step 2.
    print("\n--- Step 3: apply_mean_trigger ---")
    intervals_mean = apply_mean_trigger(
        stream=stream,
        tuples=intervals_fourier,
        n_cores=1,
        sigma=6,
        window_size=1500,
        record_length=RECORD_LENGTH,
    )
    print(f"  Intervals after mean trigger: {len(intervals_mean)}")
    for a, b in intervals_mean[:8]:
        print(f"    [{a:>9,} – {b:>9,}]  ({(b-a)/SAMPLE_FREQ:.2f} s)")
    if len(intervals_mean) > 8:
        print(f"    … and {len(intervals_mean)-8} more")


    # =========================================================
    # STEP 4 – divide_array
    # =========================================================
    # HOW IT WORKS
    # ------------
    # Each surviving interval may span many record lengths.  divide_array
    # chops it into consecutive non-overlapping segments of exactly
    # record_length samples.  A small random offset is added to each
    # segment start to avoid systematic phase alignment with any residual
    # periodic noise.
    #
    # OUTPUT: list of (start, end) tuples where end - start == record_length.
    print("\n--- Step 4: divide_array ---")
    segments = divide_array(tuples=intervals_mean, record_length=RECORD_LENGTH)
    print(f"  Fixed-length segments: {len(segments)}")
    print(f"  Each segment: {RECORD_LENGTH} samples = {RECORD_LENGTH/SAMPLE_FREQ*1e3:.1f} ms")


    # =========================================================
    # STEP 5 – decaying_baseline_remover
    # =========================================================
    # HOW IT WORKS
    # ------------
    # Even after pulse vetoing, a segment that starts just after a large
    # pulse might still sit on the decaying tail.  For each segment this
    # function computes:
    #
    #   sdt_dev = (std(first 100 samples) + std(last 100 samples)) / 2
    #
    # and checks whether the means of the first and last 100 samples differ
    # by less than sdt_dev.  If the difference is larger, the segment is
    # still drifting and is discarded.
    #
    # OUTPUT: list of INTEGER starting indices (not tuples!) of accepted
    #         baseline records.  These are the indices you pass to the NPS
    #         calculator.
    print("\n--- Step 5: decaying_baseline_remover ---")
    clean_idx = decaying_baseline_remover(stream=stream, tuples=segments)
    print(f"  Good baseline start indices: {len(clean_idx)}")


    # =========================================================
    # STEP 6 – Full pipeline via get_clean_bs_idx
    # =========================================================
    # All five steps above are called internally by get_clean_bs_idx.
    # Running it should give the same (or near-identical) result.
    print("\n--- Step 6: get_clean_bs_idx (complete pipeline) ---")
    clean_idx_full = get_clean_bs_idx(
        stream=stream,
        record_length=RECORD_LENGTH,
        dt_us=DT_US,
        remove_decaying_baseline=True,
        n_cores=1,
    )
    print(f"  Good baseline start indices (full pipeline): {len(clean_idx_full)}")


    # =========================================================
    # STEP 7 – Sanity checks
    # =========================================================
    # Verify that the selected records are actually clean.
    # For each returned index we check two things:
    #
    #   A) No sample exceeds 5·NOISE_SIGMA after subtracting the record mean
    #      (a real pulse would give values >> 100·NOISE_SIGMA).
    #
    #   B) The record does not overlap any known pulse region.
    print("\n--- Step 7: Sanity checks ---")

    # Gaussian noise over 32 768 samples has an expected peak of ~4.7σ.
    # Using 6σ as threshold means random noise peaks will almost never
    # false-alarm, while a real pulse (>> 100σ) would definitely trigger.
    CLEAN_THRESH = 6 * NOISE_SIGMA          # 0.018 V

    bad_amplitude = 0
    for idx in clean_idx_full:
        record = stream[idx : idx + RECORD_LENGTH]
        # subtract local mean to remove the DC level-shift offset
        if np.max(np.abs(record - record.mean())) > CLEAN_THRESH:
            bad_amplitude += 1

    print(f"  Records with peak excursion > {CLEAN_THRESH:.4f} V : "
          f"{bad_amplitude} / {len(clean_idx_full)}")

    # Per-amplitude overlap check
    pulse_tails_by_amp = {}
    for p, amp in pulses:
        pulse_tails_by_amp.setdefault(amp, []).append((p, p + 5 * PULSE_TAU))

    total_overlap = 0
    for amp in PULSE_AMPS:
        tails = pulse_tails_by_amp.get(amp, [])
        missed = 0
        for idx in clean_idx_full:
            record_end = idx + RECORD_LENGTH
            if any(not (record_end < p_start or idx > p_end) for p_start, p_end in tails):
                missed += 1
        total_overlap += missed
        caught = "✓ caught" if missed == 0 else f"✗ {missed} record(s) leaked through"
        print(f"  A = {amp:.2f} V ({amp/NOISE_SIGMA:.0f} σ): {caught}")

    if bad_amplitude == 0 and total_overlap == 0:
        print("  ✓ All selected records are pulse-free")
    elif total_overlap == 0:
        print("  ✓ No overlap with pulse regions (amplitude outliers are random noise at ≥6σ)")
    else:
        print(f"  ✗ {total_overlap} record(s) overlap pulse regions — inspect the plot")


    # =========================================================
    # STEP 8 – Visualisation
    # =========================================================
    # Three panels:
    #   A. Full stream overview (downsampled) with annotated pulse positions
    #      and level shift.
    #   B. Same stream with selected baseline windows highlighted in green.
    #   C. One clean baseline record vs. one event record side by side.
    print("\n--- Step 8: Plotting ---")

    ds = 200    # downsample factor for full-stream plots (speeds up rendering)
    t_stream = np.arange(0, STREAM_LENGTH, ds) / SAMPLE_FREQ   # seconds

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle("NPS Pipeline: Mock Stream Analysis", fontsize=13, y=1.01)

    # ── Panel A: raw stream ───────────────────────────────────
    amp_colors = {1.0: 'red', 0.1: 'darkorange', 0.02: 'gold'}
    ax = axes[0]
    ax.plot(t_stream, stream[::ds], lw=0.25, color='steelblue', label='stream')
    labeled = set()
    for p, amp in pulses:
        color = amp_colors.get(amp, 'red')
        label = f'pulse A={amp:.2f} V' if amp not in labeled else ''
        labeled.add(amp)
        ax.axvspan(p / SAMPLE_FREQ, (p + 5 * PULSE_TAU) / SAMPLE_FREQ,
                   alpha=0.25, color=color, label=label)
    ax.axvline(LEVEL_SHIFT_I / SAMPLE_FREQ, color='purple',
               lw=1.5, ls='--', label=f'level shift (+{LEVEL_SHIFT_V} V)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('A.  Raw mock stream (downsampled ×200)')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(0, STREAM_LENGTH / SAMPLE_FREQ)

    # ── Panel B: selected baseline windows ───────────────────
    ax = axes[1]
    ax.plot(t_stream, stream[::ds], lw=0.25, color='steelblue', alpha=0.6)
    labeled = set()
    for p, amp in pulses:
        color = amp_colors.get(amp, 'red')
        label = f'pulse A={amp:.2f} V' if amp not in labeled else ''
        labeled.add(amp)
        ax.axvspan(p / SAMPLE_FREQ, (p + 5 * PULSE_TAU) / SAMPLE_FREQ,
                   alpha=0.15, color=color, label=label)
    first_bl = True
    for idx in clean_idx_full:
        label = f'baseline record ({len(clean_idx_full)} total)' if first_bl else ''
        ax.axvspan(idx / SAMPLE_FREQ, (idx + RECORD_LENGTH) / SAMPLE_FREQ,
                   alpha=0.35, color='limegreen', label=label)
        first_bl = False
    ax.axvline(LEVEL_SHIFT_I / SAMPLE_FREQ, color='purple', lw=1.5, ls='--')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('B.  Selected clean baseline windows (green)')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(0, STREAM_LENGTH / SAMPLE_FREQ)

    # ── Panel C: one baseline vs. one event record ────────────
    ax = axes[2]
    t_rec = np.arange(RECORD_LENGTH) / SAMPLE_FREQ * 1e3   # milliseconds

    if len(clean_idx_full) > 0:
        bl_rec = stream[clean_idx_full[0] : clean_idx_full[0] + RECORD_LENGTH]
        ax.plot(t_rec, bl_rec - bl_rec.mean(), lw=0.7, color='limegreen',
                label=f'baseline record (idx={clean_idx_full[0]:,})')

    # Build an event record centred on the first (largest) pulse peak
    p0, _ = pulses[0]
    ev_s  = p0 - RECORD_LENGTH // 4            # pre-trigger = ¼ of record
    if ev_s >= 0 and ev_s + RECORD_LENGTH < STREAM_LENGTH:
        ev_rec = stream[ev_s : ev_s + RECORD_LENGTH].copy()
        # Subtract mean of the first 100 pre-trigger samples as baseline
        ev_rec -= ev_rec[:100].mean()
        ax.plot(t_rec, ev_rec, lw=0.7, color='tomato',
                label=f'event record (pulse at {p0:,})')

    ax.axvline(RECORD_LENGTH / 4 / SAMPLE_FREQ * 1e3, color='grey',
               ls=':', lw=1.0, label='expected pulse onset (¼ window)')
    ax.set_xlabel('Time within record (ms)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('C.  Example records: clean baseline vs. triggered event')
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = 'test_nps_pipeline.png'
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"  Saved figure → {out_path}")
    print("\nDone.")
