"""Tab 2 - QC Preview. See :mod:`calliope.tabs.qc.tab` for the GUI
class. This sub-package is small (just the QC GIF viewer); its
``logic`` module is a thin re-export shim over ``core.preprocessing``
(``PreprocessResult`` + ``load_existing_preprocess``), which the tab
imports as ``from . import logic as preprocessing``.

This file exists only so Python recognises the folder as a sub-package.
"""
