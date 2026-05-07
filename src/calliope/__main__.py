"""Entrypoint for ``python -m calliope`` - launches the CalLIOPE GUI.

In Python, a folder becomes a "package" once it has an ``__init__.py``
file inside it. When you run a package with ``python -m <package>``,
Python looks for a special ``__main__.py`` file in that package and
executes it. Think of this file as the equivalent of an R script you
would run with ``Rscript run.R`` -- it's the script that fires when the
user types the launch command.

The actual application lives in ``pipeline_gui.py``; this file just
imports its ``main()`` function and calls it. We keep them separate so
that other Python code can still ``from calliope.pipeline_gui import
PipelineApp`` without immediately starting the GUI's event loop.
"""

# ``from .pipeline_gui import main`` reads as: from the file
# ``pipeline_gui.py`` sitting next to me in this same package, pull in
# the function called ``main`` so I can call it below.
from .pipeline_gui import main

# This guard is a Python convention. ``__name__`` is a special variable
# Python sets automatically: when a file is executed directly (the
# typical case for an entrypoint) it equals the string ``"__main__"``;
# when the file is *imported* by some other module, ``__name__`` is the
# module's dotted path instead. The check below ensures ``main()`` only
# runs when this file is the program being launched, never as a
# side-effect of being imported.
if __name__ == "__main__":
    main()
