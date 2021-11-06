**************
Magic Numbers
**************

There are several places in the Cait code, where so-called "magic" number appear, e.g. a hard-coded integer as index
to an array, with no or only short explanation why this number is taken here. Most of the time, this is due to the
numbering of labels, main parameters or fit parameters. These numbers are written as attributes in the HDF5 files,
however we also provide lists here.

Labels
=========

| 0 ... unlabeled
| 1 ... Event Pulse

A usual pulse, originating in a particle recoil within the absorber crystal.

| 2 ... Test/Control Pulse

A testpulse, induced by a temperature pulse from the detector heater. These pulses are intentionally induced every 5 seconds, to measure the variations in the detector response. Their pulse form is different than that of event pulses, with typically a longer rise time.

| 3 ... Noise

An empty noise baseline, or a noise trigger.

| 4 ... Squid Jump

A jump in the offset level of the baseline, coming from a reset of the SQUID or the loss of a flux quant.

| 5 ... Spike

A thin positive or negative spike from the detector electronics.

| 6 ... Early or late Trigger

An absorber pulse with maximum time significantly different than zero (>/< 20 ms).

| 7 ... Pile Up

Two or more events in one record window.

| 8 ... Carrier Event

A thin, low energetic pulse, originating in a particle recoil within the carrier crystal of the detector.

| 9 ... Strongly Saturated Event Pulse

Saturated Event Pulse. Saturation happens, when the TES is driven out of its linear operation range by a high energetic particle recoil within the absorber crystal.

| 10 ... Strongly Saturated Test/Control Pulse

Same as Label 9, but for a Test Pulse.

| 11 ... Decaying Baseline

Trigger of noise or a sub-threshold event, that gets elevated above threshold by a decaying baseline.

| 12 ... Temperature Rise

A sudden and continuous rise of the detector temperature causes a strong positive slope in the noise baseline, that eventually surpasses the threshold.

| 13 ... Stick Event

A pulse with long rise- and decay time, caused by a particle recoil within the sticks that hold the crystal. The sticks themselves are made from e.g. CaWO4.

| 14 ... Square Waves

Small, discrete jumps of the baseline level. A upward jump, followed right away by a downward jump, is also called a jump event and might look like a pulse on first sight.

| 15 ... Human Disturbance

An oscillating signal, caused by vibration.

| 16 ... Large Sawtooth

Multiple squid resets, cause by fast rising temperature - e.g. due to warm up of the cryostat or very large heat deposition to a sensible TES.

| 17 ... Cosinus Tail

An event with especially long tail, typical for the several Cosinus modules.

| 18 ... Light only Event

A direct hit in the light detector, without corresponding phonon signal.

| 19 ... Ring & Light Event

A recoil in the ring od the detector, this component is e.g. included in Gode/beaker modules.

| 20 ... Sharp Light Event

A very fast rising light event, typically caused by a direct hit in the light detector, with a hit in the detector following after.

| 99 ... unknown/other

In some cases, we have not enough insight to explain an event. If this concerns only a single or very few event, these are together put in an unknown class, which is then excluded from the analysis.


Main Parameters
===================

| 0 ... pulse_height

The Maximum Value of the Event, after application of a 50 sample moving average, in Volt.

| 1 ... t_zero

This value is calculated backwards in time: Starting at the position of the sample with the maximum value,
the last sample that subceeds 20% of the pulse height.

| 2 ... t_rise

This value is calculated backwards in time: Starting at the position of the sample with the maximum value,
the last sample that subceeds 80% of the pulse height.

| 3 ... t_max

The index of the maximal sample.

| 4 ... t_decaystart

The first sample after t_max, that falls below 90% of the maximal height.

| 5 ... t_half

The first sample after t_max, that falls below 73% of the maximal height.

| 6 ... t_end

The first sample after t_max, that falls below 36% of the maximal height.

| 7 ... offset

The average of the first 500 samples of the event.

| 8 ... linear_drift

The difference between the average of the first and last 500 samples of the event, divided by the record length.

| 9 ... quadratic_drift

Usually this is set to zero! The quadratic component of the baseline.

Parametric Fit Parameters
==============================

The fit parameters of the parametric pulse shape fit.

| 0 ... t_0
| 1 ... A_n
| 2 ... A_t
| 3 ... tau_n
| 4 ... tau_in
| 5 ... tau_t

Standard Event Fit Parameters
==================================

The fit parameters of the standard event fit:

| 0 ... pulse_height
| 1 ... onset
| 2 ... constant_coefficient
| 3 ... linear_coefficient
| 4 ... quadratic_coefficient
| 5 ... cubic_coefficient

The array fit has the same paramters!


Additional Main Parameters
================================

| 0 ... array_max

The maximum of the array.

| 1 ... array_min

The minimum of the array.

| 2 ... var_first_eight

The variance of the first eight of the record window. This is typically the variance of the baseline noise.

| 3 ... mean_first_eight

The mean value of the first eight of the record window. The first eight of the record window typically shows only the noise baseline, i.e. this is the mean value of the noise.

| 4 ... var_last_eight

The variance of the last eight of the record window. If this value differs strongly from the baseline variance, this indicates a strongly saturated pulse, a Pile-Up Event or an early trigger.

| 5 ... mean_last_eight

The mean value of the last eight of the record window. A difference of this value from the mean of the first eight indicates a strong baseline tilt.

| 6 ... var

The variance of the whole array.

| 7 ... mean

The mean value of the whole array.

| 8 ... skewness

The skewness of the whole array.

| 9 ... max_derivative

The maximal value of the derivative of the array.

| 10 ... ind_max_derivative

The index of the maximal value of the derivative of the array.

| 11 ... min_derivative

The minimal value of the derivative of the array.

| 12 ... ind_min_derivative

The index of the minimal value of the derivative of the array.

| 13 ... max_filtered

The maximum of the array, after applying the optimum filter.

| 14 ... ind_max_filtered

The index of the maximum of the array, after applying the optimum filter.

| 15 ... skewness_filtered_peak

The skewness of the array around its peak, after applying the optimum filter. Typically this value is higher, when the array deviates from the standard event.