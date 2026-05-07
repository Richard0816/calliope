"""Sub-package holding one folder per notebook tab.

This file is intentionally almost empty -- its only job is to mark
``tabs`` as a Python sub-package so that ``calliope.tabs.<name>``
imports resolve. Every tab lives in its own subfolder
(``preprocess``, ``qc``, ``suite2p``, ``lowpass``, ``event_detection``,
``clustering``, ``crosscorrelation``, ``spatial_propagation``) and
exposes both a ``tab`` module (Tk widgets + threading) and a ``logic``
module (re-exports of the calculations from ``calliope.core``).
"""
