"""Bundled resource files -- currently the pretrained cell-filter
checkpoint ``cellfilter_best.pt`` (Tab 3's default classifier). Base
Suite2p settings are not bundled as a file; they live in source in
``calliope.core.calliope_settings``.

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
    ckpt_path = DATA_DIR / "cellfilter_best.pt"    # a sibling file

``__file__`` is a special variable Python sets to the path of the
current source file. Calling ``.parent`` on it gives the folder the
file lives in, so the snippet above works regardless of where CalLIOPE
is installed on disk. Roughly the same idea as R's
``system.file("extdata", "foo.csv", package = "mypkg")``.
"""
