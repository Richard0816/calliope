"""Bridge a Suite2p Fall.mat into a CalLIOPE run folder (no raw movie)."""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

fall, run = Path(sys.argv[1]), Path(sys.argv[2])
plane0 = run / "detection" / "final" / "suite2p" / "plane0"
plane0.mkdir(parents=True, exist_ok=True)

m = sio.loadmat(fall, squeeze_me=True, struct_as_record=False)
for k in ("F", "Fneu", "spks", "iscell"):
    np.save(plane0 / f"{k}.npy", np.asarray(m[k]))


def to_dict(s):
    return {f: getattr(s, f) for f in s._fieldnames}


stat = np.array([to_dict(s) for s in np.atleast_1d(m["stat"])], dtype=object)
for s in stat:
    for k in ("ypix", "xpix", "lam"):
        s[k] = np.atleast_1d(np.asarray(s[k]))
    s["ypix"] = s["ypix"].astype(int)
    s["xpix"] = s["xpix"].astype(int)
np.save(plane0 / "stat.npy", stat, allow_pickle=True)
ops = to_dict(m["ops"])
ops = {k: (v.item() if isinstance(v, np.ndarray) and v.size == 1 else v)
       for k, v in ops.items()}
np.save(plane0 / "ops.npy", ops, allow_pickle=True)
np.save(run / "mean.npy", np.asarray(ops["meanImg"], np.float32))
print("plane0 written:", plane0, "N =", len(stat), "T =", m["F"].shape[1])

from calliope.core import detection_run  # noqa: E402
import calliope; DEFAULT_CKPT = Path(calliope.__file__).parent / "data" / "cellfilter_best.pt"

t = time.time()
detection_run.compute_dff_memmaps(plane0, {"use_gpu_dff": False},
                                  baseline_mode="first_n", baseline_min=2.0,
                                  progress_cb=print)
print(f"dff {time.time() - t:.1f}s"); t = time.time()
detection_run.predict_cell_filter(plane0, str(DEFAULT_CKPT), run.name)
print(f"cellfilter {time.time() - t:.1f}s")
print("kept:", int(np.load(plane0 / "predicted_cell_mask.npy").sum()),
      "of", len(stat), "| suite2p iscell:", int(m["iscell"][:, 0].sum()))
detection_run.compute_filtered_dff(plane0)
print(sorted(p.name for p in plane0.iterdir()))
