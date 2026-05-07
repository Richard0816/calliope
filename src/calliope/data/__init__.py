"""Bundled resource files (Suite2p ``ops`` JSON, AAV metadata, etc.).

This sub-package isn't really code -- it's a folder full of static
data files that ship alongside the Python source. The ``__init__.py``
file you are reading exists only because Python requires it for the
folder to count as a sub-package; without it, the rest of CalLIOPE
couldn't refer to ``calliope.data`` at all.

How files in here are read
--------------------------
Other modules look up bundled resources with the idiom::

    from pathlib import Path
    DATA_DIR = Path(__file__).parent              # this folder
    ops_path = DATA_DIR / "suite2p_ops_v1.json"   # a sibling file

``__file__`` is a special variable Python sets to the path of the
current source file. Calling ``.parent`` on it gives the folder the
file lives in, so the snippet above works regardless of where CalLIOPE
is installed on disk. Roughly the same idea as R's
``system.file("extdata", "foo.csv", package = "mypkg")``.
"""
