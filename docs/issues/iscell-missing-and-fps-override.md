# Open issues: "iscell.npy missing" warning and FPS override ignored downstream

Reported by a user after loading a recording. Two separate problems.
Investigation only so far; no code has been changed.

---

## Issue 1: "iscell.npy missing" warning in the ROI curation popout

### Where it comes from

The only source of this warning is the Tab 3 (Suite2p) curation popout:

- `src/calliope/tabs/suite2p/curation_popout.py` `_set_label` (~line 614) and
  `_on_promote` (~line 671) warn when `self._iscell is None`.
- `self._iscell` is set **once**, in `__init__`, via `_load_iscell`
  (~line 896).

`_load_iscell` returns `None` in more cases than "file absent", but the
warning always says "missing".

### Possible causes (ranked by likelihood)

1. **ROI count mismatch.** `_load_iscell` returns `None` when
   `arr.shape[0] != self._n_total` (~lines 909-913), i.e. `iscell.npy` and
   `stat.npy` have different numbers of ROIs. This can happen after a
   re-detect, or when files from different runs are mixed in one plane0.
   The file exists, but the message says it is missing.
2. **Unreadable file.** `_load_iscell` has a bare `except Exception:
   return None` (~line 915). Possible causes: a partly written or corrupt
   file, a cloud-sync placeholder (OneDrive/Dropbox not yet downloaded), or
   a Windows file lock.
3. **Old result kept by a reused popout.** `suite2p/tab.py` (~line 1527)
   reuses the existing popout as long as its `_plane0` matches. If it was
   opened while iscell was absent or mismatched, `_iscell` stays `None`
   even after the file is fixed, until the popout is closed. The user's
   workaround of re-running Tab 0 likely helped because it recreated
   things.
4. **The loaded folder never had an iscell.npy.** "Load existing panels"
   only checks `stat.npy` + `ops.npy` (`suite2p/logic.py`
   `plane0_has_outputs`, ~line 55). A copied folder, or a plain suite2p
   output with no classifier, passes that check.
5. **The popout points at a deleted scratch folder** after batch copies
   results from scratch to the HDD. `_repoint_after_copy` in
   `suite2p/tab.py` (~line 1543) and
   `BatchTab._repoint_pipeline_after_copy` (`batch/tab.py` ~line 2985)
   are supposed to prevent this. Worth checking if the reported path is a
   scratch path.

### Diagnosis tip

The warning prints the plane0 path. Check whether `iscell.npy` exists at
that path:

- If it exists, the cause is 1, 2 or 3.
- If the path is a scratch folder, the cause is 5.
- Otherwise, the cause is 4.

### Suggested fixes

- In `_load_iscell`, keep the reason the load failed (absent, row count
  mismatch with both counts, or the load exception) and show it in the
  warning instead of always saying "missing".
- In `_set_label` / `_on_promote`, if `self._iscell is None`, try
  `self._load_iscell()` again before warning. This fixes cause 3.
  Alternatively, reload iscell in `show_roi` when the popout is reused.
- Optionally, warn (without blocking) in `_on_load_existing` when
  `iscell.npy` is absent, so the problem shows up at load time rather
  than at the first curation keypress.
- Add tests for each `_load_iscell` failure path.

---

## Issue 2: FPS override (Tab 3) is ignored by Tab 4 and later stages

### Symptom

Detection was run with FPS override = 15.08. The Tab 4 (lowpass) subtitle
shows `fps=15.07`.

### Root cause

`fps_override` is only a Tab 3 parameter and is only used inside detection
and dF/F (`suite2p/tab.py` ~lines 969, 1048; `core/detection_run.py`
~lines 504, 873, 947). Every later stage gets its frame rate from
`utils.get_fps_from_notes(plane0)`. When `notes_root` is not given (the
default), that function just returns `DEFAULT_FPS = 15.07`
(`core/utils.py` ~lines 710, 859-866).

**This is not just a display bug.** Tab 4 stores the value in
`self._fps` (`lowpass/tab.py` ~line 536) and uses it for:

- the FFT axis
- the low-pass/derivative compute (`_on_compute`, ~line 738)
- figure export

Other callers that ignore the override:

- `tabs/lowpass/tab.py:536`
- `tabs/clustering/tab.py:870`, `:1456`
- `tabs/crosscorrelation/tab.py:1106`
- `tabs/suite2p/tab.py:1796`
- `tabs/suite2p/curation_popout.py:120` (`_fps_for`)
- `core/event_detection_run.py:350`
- `core/clustering.py:675`
- `core/nwb_export.py:189`
- `core/lowpass_run.py:284` (honours `params["fps_override"]` only if the
  caller passes it; check `core/batch_pipeline.py:234`)

Impact for 15.07 vs 15.08 is small: about a 0.07% shift in cutoff and
timing, roughly 0.4 s by the end of a 10-minute recording. A larger
override would produce wrong results with nothing to flag it.

### Suggested fix

The rate chosen at detection is already saved on disk.
`core/sparse_plus_cellpose.py` (~line 376) writes `final_ops['fs']` from
the settings, which take `fs_override` from `fps_override` (~line 643).
First confirm that `plane0/ops.npy["fs"]` really equals the override for
an override run.

1. Add a helper to `core/utils.py`, e.g.
   `get_fps_for_plane0(plane0, default_fps=DEFAULT_FPS) -> float`, that:
   - reads `fs` from `plane0/ops.npy` (or `load_plane_view`)
   - returns it if it is finite and > 0
   - otherwise falls back to `get_fps_from_notes(plane0)`
2. Switch all the callers listed above to the helper. Keep explicit
   `fps_override` parameters taking precedence where they exist.
3. Optionally show where the value came from in the Tab 4 subtitle, e.g.
   `fps=15.08 (ops)` vs `fps=15.07 (default)`.
4. Add a test: write an `ops.npy` with `fs=15.08` into a temp plane0 and
   assert the helper returns 15.08. Also test the fallback when `fs` is
   missing.

Existing data analysed before the fix: Tab 4 low-pass/derivative memmaps
and downstream results were computed at 15.07 Hz. Re-run Tab 4 onward
after the fix if exact timing matters.
