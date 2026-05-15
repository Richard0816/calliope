"""calliope - GUI package for the calcium imaging pipeline.

A note for readers
--------------------------------
Python packages are *folders containing an ``__init__.py`` file*. The
file you are reading is the ``__init__.py`` for the top-level
``calliope`` package, so anything you put here runs the first time
something does ``import calliope``. We keep it almost empty -- it just
sets a version string and acts as an overview of the package layout.
Roughly the equivalent of an R package's ``DESCRIPTION`` plus a short
introductory ``.Rd`` file rolled into one.

Layout
------
- ``calliope.pipeline_gui``  - top-level Tk application (the "main")
- ``calliope.gui_common``    - state object + dialog/toolbar helpers
                                shared by every tab
- ``calliope.tabs.<name>``   - one subfolder per notebook tab. Each
                                holds ``tab.py`` (the GUI widgets) and
                                ``logic.py`` (a thin re-export shim
                                pointing at calculations in ``core``)
- ``calliope.core``          - pure-Python algorithms shared across
                                tabs (preprocessing, dF/F, lowpass,
                                event detection, hierarchical
                                clustering palettes, cross-correlation,
                                cellpose-based detection, the
                                cell-filter PyTorch model)
- ``calliope.data``          - bundled resource files (Suite2p ``ops``
                                JSON, the pretrained cell-filter
                                checkpoint, etc.)

Run the GUI with::

    python -m calliope

That command tells Python to "find the package called calliope and run
its ``__main__.py`` file" -- see the sibling file for what happens
next.
"""

# Public version string. Convention: most Python packages expose a
# ``__version__`` here so callers can do ``import calliope;
# calliope.__version__``. The leading/trailing underscores are not
# magic -- they are just part of the name. (R's analog is the
# ``DESCRIPTION`` file's ``Version:`` field.)
__version__ = "0.1.0"
