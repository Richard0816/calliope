"""Sub-package holding one folder per notebook tab.

This file is intentionally almost empty -- its only job is to mark
``tabs`` as a Python sub-package so that ``calliope.tabs.<name>``
imports resolve. Every tab lives in its own subfolder -- ``batch``
(Tab 0, the batch runner), ``preprocess``, ``qc``, ``suite2p``,
``lowpass``, ``event_detection``, ``clustering``, ``crosscorrelation``,
``spatial_propagation`` (Tabs 1-8) -- and exposes both a ``tab``
module (Tk widgets + threading) and a ``logic`` module. Most ``logic``
modules mainly re-export calculations from ``calliope.core``; some
(``batch``, ``clustering``, ``suite2p``) also hold real tab-level
logic of their own.
"""
