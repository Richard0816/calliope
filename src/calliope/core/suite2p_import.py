"""Start a CalLIOPE run from an existing Suite2p result (UX audit finding 14).

Accepts either a Suite2p ``plane0`` folder (``F.npy``, ``Fneu.npy``,
``spks.npy``, ``iscell.npy``, ``stat.npy``, ``ops.npy``) or a MATLAB
``Fall.mat``, writes them into CalLIOPE's
``<run>/detection/final/suite2p/plane0`` layout, then runs the same
dF/F -> cell filter -> filtered dF/F steps Tab 3 runs after detection.
No raw movie is needed; Tabs 1-2 stay empty for such a run.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

_PLANE0_FILES = ("F.npy", "Fneu.npy", "spks.npy", "iscell.npy", "stat.npy",
                 "ops.npy")


def _from_fall_mat(fall: Path, plane0: Path) -> None:
    import scipy.io as sio
    m = sio.loadmat(str(fall), squeeze_me=True, struct_as_record=False)
    for k in ("F", "Fneu", "spks", "iscell"):
        np.save(plane0 / f"{k}.npy", np.asarray(m[k]))

    def to_dict(s):
        return {f: getattr(s, f) for f in s._fieldnames}

    stat = np.array([to_dict(s) for s in np.atleast_1d(m["stat"])],
                    dtype=object)
    for s in stat:
        for k in ("ypix", "xpix", "lam"):
            s[k] = np.atleast_1d(np.asarray(s[k]))
        s["ypix"] = s["ypix"].astype(int)
        s["xpix"] = s["xpix"].astype(int)
    np.save(plane0 / "stat.npy", stat, allow_pickle=True)
    ops = {k: (v.item() if isinstance(v, np.ndarray) and v.size == 1 else v)
           for k, v in to_dict(m["ops"]).items()}
    np.save(plane0 / "ops.npy", ops, allow_pickle=True)


def import_suite2p(src: str, run_folder: str, ckpt_path: str = "",
                   progress_q=None) -> str:
    """Copy/convert ``src`` (plane0 folder or Fall.mat) into
    ``run_folder`` and compute dF/F + cell filter. Returns the run
    folder. Picklable :mod:`calliope.core.offload` target."""
    from . import detection_run
    from .offload import make_progress_cb
    cb = make_progress_cb(progress_q) or print
    src_p, run = Path(src), Path(run_folder)
    plane0 = run.joinpath("detection", "final", "suite2p", "plane0")
    plane0.mkdir(parents=True, exist_ok=True)
    if src_p.is_dir():
        missing = [f for f in _PLANE0_FILES if not (src_p / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{src_p} is not a Suite2p plane0 folder (missing "
                f"{', '.join(missing)}).")
        for f in _PLANE0_FILES:
            shutil.copy2(src_p / f, plane0 / f)
    elif src_p.suffix.lower() == ".mat":
        _from_fall_mat(src_p, plane0)
    else:
        raise ValueError(f"Expected a plane0 folder or a Fall.mat, got {src_p}")
    n = len(np.load(plane0 / "stat.npy", allow_pickle=True))
    fs = np.load(plane0 / "ops.npy", allow_pickle=True).item().get("fs")
    cb(f"[import] {n} ROIs, fs={fs} Hz (from Suite2p ops) -> {plane0}")

    detection_run.compute_dff_memmaps(plane0, {}, baseline_mode="first_n",
                                      baseline_min=2.0, progress_cb=cb)
    if ckpt_path and Path(ckpt_path).is_file():
        cb("[import] running the cell filter...")
        detection_run.predict_cell_filter(plane0, ckpt_path, run.name)
    detection_run.compute_filtered_dff(plane0)
    cb("[import] done")
    return str(run)
