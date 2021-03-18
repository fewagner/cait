**************
Magic Numbers
**************

There are several places in the Cait code, where so-called "magic" number appear, e.g. a hard-coded integer as index
to an array, with no are only short explanation why this number is taken here. Most of the time, this is due to the
numbering of labels, main parameters or fit parameters. These numbers are written as attributes in the HDF5 files,
however we also provide lists here.

Labels
=========

| 0 ... unlabeled
| 1 ... Event Pulse
| 2 ... Test/Control Pulse
| 3 ... Noise
| 4 ... Squid Jump
| 5 ... Spike
| 6 ... Early or late Trigger
| 7 ... Pile Up
| 8 ... Carrier Event
| 9 ... Strongly Saturated Event Pulse
| 10 ... Strongly Saturated Test/Control Pulse
| 11 ... Decaying Baseline
| 12 ... Temperature Rise
| 13 ... Stick Event
| 14 ... Square Waves
| 15 ... Human Disturbance
| 16 ... Large Sawtooth
| 17 ... Cosinus Tail
| 18 ... Light only Event
| 19 ... Ring & Light Event
| 20 ... Sharp Light Event
| 99 ... unknown/other

Main Parameters
===================

| 0 ... pulse_height
| 1 ... t_zero
| 2 ... t_rise
| 3 ... t_max
| 4 ... t_decaystart
| 5 ... t_half
| 6 ... t_end
| 7 ... offset
| 8 ... linear_drift
| 9 ... quadratic_drift

Parametric Fit Parameters
==============================

| 0 ... t_0
| 1 ... A_n
| 2 ... A_t
| 3 ... tau_n
| 4 ... tau_in
| 5 ... tau_t

Standard Event Fit Parameters
==================================

| 0 ... pulse_height
| 1 ... onset
| 2 ... constant_coefficient
| 3 ... linear_coefficient
| 4 ... quadratic_coefficient
| 5 ... cubic_coefficient


Additional Main Parameters
================================

| 0 ... array_max
| 1 ... array_min
| 2 ... var_first_eight
| 3 ... mean_first_eight
| 4 ... var_last_eight
| 5 ... mean_last_eight
| 6 ... var
| 7 ... mean
| 8 ... skewness
| 9 ... max_derivative
| 10 ... ind_max_derivative
| 11 ... min_derivative
| 12 ... ind_min_derivative
| 13 ... max_filtered
| 14 ... ind_max_filtered
| 15 ... skewness_filtered_peak
