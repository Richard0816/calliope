"""Logic + file-I/O helpers for the batch runner tab.

This module hosts every non-GUI building block ``tab.py`` needs:

- **Constants** for the JSON / CSV sidecar filenames and the TIFF glob
  patterns.
- **Scratch-dir hygiene** -- ``prune_scratch_tree`` and
  ``is_scratch_mirror_skippable`` enumerate what gets dropped from each
  recording's scratch tree at the row-end finalize and what the
  continuous mirror skips in-flight.
- **Throttled / dedup'ing copy primitives** -- ``throttled_copy2`` (the
  single-drive rate limiter), ``copy_one_dedup`` (with hardlink reuse
  via an ``inode -> dst`` map), and the two sweep functions
  (``synchronous_final_sweep`` for the safety net before bulk-rmtree,
  ``mirror_sync_pass`` for the background mirror loop).
- **Param-spec assembly** -- ``build_batch_param_spec`` walks every
  per-tab ``PARAM_SPEC`` and rewrites the group labels with numbered
  stage prefixes so the Advanced dialog reads top-to-bottom in
  pipeline order.

Splitting this out of ``tab.py`` keeps the BatchTab class focused on
widget layout + worker orchestration; the I/O primitives can be unit-
tested or reused outside the GUI without dragging in customtkinter.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Sidecar filenames + TIFF globs
# ---------------------------------------------------------------------------

BATCH_JSON_NAME = "calliope_batch.json"
BATCH_REPORT_NAME = "batch_report.csv"
TIFF_GLOBS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")


# ---------------------------------------------------------------------------
# Scratch-dir hygiene constants
# ---------------------------------------------------------------------------

# Patterns dropped from each recording's scratch tree *before* the
# bulk SSD->HDD copy. Everything else is preserved verbatim so the
# GUI's "load existing" paths and a manual suite2p re-run with the
# cached registration both still work. See the 2026-05-11 lab note
# for the full audit of what stays vs goes.
#
# Saved bytes per recording are dominated by ``data_raw.bin`` (the
# pre-registration movie suite2p caches when ``keep_movie_raw=True``;
# ~8 GB on a 10-minute recording, fully redundant once the registered
# ``data.bin`` is on disk).
SCRATCH_PRUNE_FILE_PATTERNS: tuple[str, ...] = (
    "**/data_raw.bin",          # suite2p pre-registration cache
    "**/data_raw_chan2.bin",    # ditto for two-channel runs
    "**/*.tmp",                 # partial writes
    "**/*.lock",
)

# Subdirectories of each recording's scratch folder that are
# detection intermediates. ``sparse_plus_cellpose.run`` writes them
# directly under the recording folder (NOT under a ``detection/``
# subfolder) and ``merge_and_extract`` consumes their stat/F/Fneu/
# spks arrays into ``final/`` -- nothing in the pipeline reads them
# after that point. The hardlinked ``data.bin`` is deduped at the
# destination via the mirror's per-inode map, but per-pass
# ``F.npy`` / ``Fneu.npy`` / ``spks.npy`` / ``stat.npy`` add
# ~300-500 MB per recording of redundant arrays.
SCRATCH_RECORDING_INTERMEDIATE_DIRS: tuple[str, ...] = (
    "sparsery_pass",
    "cellpose_pass",
)

# Defensive: same drop policy applied under a ``<rec>/detection/``
# layer too, in case a future code path uses that nesting (Tab 3's
# README documents this layout historically). No-op when the
# subfolder doesn't exist.
SCRATCH_DETECTION_KEEP_DIRS: frozenset = frozenset({
    "_shared_reg",
    "final",
})

# Single-drive finalize throttle. When scratch and output share a
# drive, the row-end finalize would otherwise hammer the HDD with
# writes while the next row's preprocess is trying to read source
# TIFFs from the same drive -- net throughput collapses for both.
# Capping the finalize write rate at 10 MB/s leaves most of the
# HDD's bandwidth for the active stage's reads at the cost of a
# slower (but unattended) background transfer. 10 GB takes ~17 min
# at 10 MB/s vs ~67 s unthrottled; the trade-off is "GUI feels
# responsive" vs "transfer wraps up sooner". Two-drive mode skips
# the throttle entirely (no contention to manage).
SINGLE_DRIVE_FINALIZE_RATE_BYTES_PER_SEC: int = 10 * 1024 * 1024  # 10 MB/s


# ---------------------------------------------------------------------------
# Scratch prune + sweep helpers
# ---------------------------------------------------------------------------


def prune_scratch_tree(
    scratch: Path,
    *,
    keep_registration: bool = True,
    keep_shifted: bool = True,
    final_rec: Optional[Path] = None,
) -> tuple[int, int]:
    """Delete known-redundant files + directories under ``scratch``.

    Returns ``(n_paths, bytes_freed)`` for logging. Errors on
    individual paths are swallowed so a single weird filesystem
    state doesn't abort the entire finalize.

    The decision tree (see audit in the lab note) keeps everything
    the GUI's reload paths read (shifted TIFF for Tab 1, suite2p
    plane0 npys, dF/F memmaps, calliope_summary.xlsx, cluster
    exports, figures) plus ``_shared_reg/data.bin`` so a manual
    suite2p re-run can skip registration. It only drops:

    * suite2p's pre-registration movie cache (``data_raw*.bin``) --
      superseded once the registered ``data.bin`` is on disk.
    * ``sparsery_pass/`` + ``cellpose_pass/`` -- intermediate
      detection outputs that ``merge_and_extract`` already folded
      into ``final/``. The hardlinked ``data.bin`` survives via
      ``_shared_reg/`` + ``final/``; the per-pass F/Fneu/spks/stat
      arrays are pure clutter (~300-500 MB per recording).
    * Defensively, any ``<rec>/detection/<X>/`` subdirectory whose
      name isn't ``_shared_reg`` or ``final`` -- no current code
      path uses that nesting, but earlier docstrings reference it.
    * ``.tmp`` / ``.lock`` partial-write artefacts.

    The Tab 0 "What to save" toggles add two opt-out drops on top of
    that always-on baseline (defaults preserve current behavior):

    * ``keep_registration=False`` -- drop the registered movie
      ``data.bin``. It is hardlinked between ``_shared_reg/`` and
      ``final/.../plane0/``; BOTH names must be unlinked to free the
      bytes (the inode survives while either link remains). The ROI
      arrays (F/Fneu/spks/stat/iscell/ops) are never touched -- only
      detection re-runs need ``data.bin`` and they'd just
      re-register. ``final_rec`` is the surviving HDD copy: pass it
      so the final-output twin is removed too, not just scratch.
    * ``keep_shifted=False`` -- drop the motion-corrected
      ``shifted_*.tif`` / ``NNN_shifted_*.tif``. Tab 1's "reload
      existing" reads it, so dropping it means a Tab 1 reload would
      have to regenerate it from a compressed raw (if archived) or
      re-preprocess. Removed from both ``scratch`` and ``final_rec``.
    """
    # Extra glob patterns + the data.bin special-case (both hardlink
    # twins) are applied to scratch and -- because the continuous
    # mirror may already have copied them -- to the surviving final
    # output dir as well.
    extra_patterns: list[str] = []
    if not keep_shifted:
        extra_patterns += ["**/shifted_*.tif", "**/shifted_*.tiff",
                           "**/*_shifted_*.tif", "**/*_shifted_*.tiff"]
    if not keep_registration:
        # Match the registered movie wherever it sits. It is hardlinked
        # between ``_shared_reg/data.bin`` and ``final/.../plane0/data.bin``
        # (same inode -- obs 663); BOTH twins must be unlinked or the
        # inode survives via the remaining link and zero bytes are freed.
        # ``**/data.bin`` catches both. NOT ``data_raw*.bin`` -- that's a
        # different name, already in SCRATCH_PRUNE_FILE_PATTERNS.
        extra_patterns += ["**/data.bin"]
    def _dir_size(p: Path) -> int:
        total = 0
        try:
            for dp, _dn, fn in os.walk(p):
                for f in fn:
                    try:
                        total += (Path(dp) / f).stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    n_paths = 0
    bytes_freed = 0
    if not scratch.exists():
        return 0, 0

    def _prune_patterns(root: Path, patterns) -> None:
        nonlocal n_paths, bytes_freed
        for pattern in patterns:
            for f in root.glob(pattern):
                try:
                    if f.is_file() or f.is_symlink():
                        sz = 0
                        try:
                            sz = f.stat().st_size
                        except OSError:
                            pass
                        f.unlink()
                        n_paths += 1
                        bytes_freed += sz
                except OSError:
                    continue

    # File-level prune: glob each pattern from the recording root, plus
    # any user opt-out extras (dropped shifted TIFF / registration).
    _prune_patterns(scratch, SCRATCH_PRUNE_FILE_PATTERNS)
    if extra_patterns:
        _prune_patterns(scratch, extra_patterns)
        # The continuous mirror may have already copied data.bin /
        # shifted TIFFs to the HDD output before this finalize runs. To
        # actually free the bytes (and, for data.bin, drop the second
        # hardlink) the surviving final-output copies must go too.
        if final_rec is not None and Path(final_rec).exists():
            _prune_patterns(Path(final_rec), extra_patterns)
    # Directory-level prune: ``sparsery_pass/`` + ``cellpose_pass/``
    # live directly under the recording folder (NOT under a
    # ``detection/`` layer); this is the path layout
    # ``sparse_plus_cellpose.run`` actually writes.
    for sub in SCRATCH_RECORDING_INTERMEDIATE_DIRS:
        p = scratch / sub
        if not p.is_dir():
            continue
        sz = _dir_size(p)
        try:
            shutil.rmtree(p, ignore_errors=True)
            n_paths += 1
            bytes_freed += sz
        except OSError:
            continue
    # Defensive: same drop policy applied if anything ever lands at
    # ``<rec>/detection/<X>/`` for ``X not in {_shared_reg, final}``.
    det = scratch / "detection"
    if det.is_dir():
        for child in det.iterdir():
            if not child.is_dir():
                continue
            if child.name in SCRATCH_DETECTION_KEEP_DIRS:
                continue
            sz = _dir_size(child)
            try:
                shutil.rmtree(child, ignore_errors=True)
                n_paths += 1
                bytes_freed += sz
            except OSError:
                continue
    return n_paths, bytes_freed


def throttled_copy2(src: str, dst: str, *,
                    rate_bytes_per_sec: int,
                    chunk_bytes: int = 1024 * 1024) -> None:
    """``shutil.copy2`` equivalent that paces its writes to roughly
    ``rate_bytes_per_sec``. Implements the rate cap by sleeping
    between fixed-size chunks: ``expected_elapsed = bytes_written /
    rate``, ``actual_elapsed = time.time() - t0``, sleep the diff
    when positive. Drift correction comes for free because the next
    chunk's expected_elapsed is computed from cumulative bytes.

    ``shutil.copystat`` at the end mirrors what ``copy2`` does (mode,
    mtime, atime, flags). Fast-path for tiny files: nothing to throttle
    when the whole file fits in one chunk and lands well under the
    per-second budget.
    """
    t0 = time.time()
    bytes_written = 0
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            chunk = fsrc.read(chunk_bytes)
            if not chunk:
                break
            fdst.write(chunk)
            bytes_written += len(chunk)
            expected = bytes_written / rate_bytes_per_sec
            elapsed = time.time() - t0
            sleep_for = expected - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    shutil.copystat(src, dst)


def is_scratch_mirror_skippable(
    rel_path: Path,
    scratch_rec: Path,
    *,
    keep_registration: bool = True,
    keep_shifted: bool = True,
) -> bool:
    """Return True iff this scratch-relative path is something the
    continuous mirror should NOT copy to HDD.

    Mirrors the row-end ``prune_scratch_tree`` logic so the mirror
    doesn't waste bandwidth on artefacts the row-end finalize would
    have dropped anyway:
    - ``data_raw.bin`` / ``data_raw_chan2.bin`` -- suite2p's
      pre-registration cache, redundant once ``data.bin`` is on disk
    - ``*.tmp`` / ``*.lock`` -- partial writes
    - Anything under ``sparsery_pass/`` or ``cellpose_pass/`` --
      detection-pass intermediates folded into ``final/`` already
    - Anything under ``<rec>/detection/<X>/`` for X not in
      {``_shared_reg``, ``final``} -- defensive (legacy layout)

    The Tab 0 "What to save" opt-outs extend the skip set (defaults
    keep, matching current behavior): ``keep_registration=False``
    skips the registered movie ``data.bin``; ``keep_shifted=False``
    skips the motion-corrected ``shifted_*.tif`` so neither is ever
    pre-copied to the HDD output the user asked to shrink.
    """
    parts = rel_path.parts
    if not parts:
        return False
    name = parts[-1]
    if name in ("data_raw.bin", "data_raw_chan2.bin"):
        return True
    if name.endswith(".tmp") or name.endswith(".lock"):
        return True
    if parts[0] in ("sparsery_pass", "cellpose_pass"):
        return True
    if (len(parts) >= 2 and parts[0] == "detection"
            and parts[1] not in SCRATCH_DETECTION_KEEP_DIRS):
        return True
    if not keep_registration and name == "data.bin":
        return True
    if not keep_shifted and name.lower().endswith((".tif", ".tiff")) \
            and "shifted_" in name:
        return True
    return False


def measure_tree(root: Path) -> tuple[int, int]:
    """Return (n_files, total_bytes) for everything under ``root``.

    Used by the finalize + batch-end paths to report leftover scratch
    contents when the bulk-rmtree didn't clear everything (typically a
    Windows file lock survived two rmtree attempts).
    """
    n = 0
    nb = 0
    if not root.exists():
        return 0, 0
    for dirpath, _dn, files in os.walk(str(root)):
        for fname in files:
            try:
                nb += (Path(dirpath) / fname).stat().st_size
                n += 1
            except OSError:
                continue
    return n, nb


def synchronous_final_sweep(scratch: Path, final: Path,
                            inode_to_dst: dict,
                            *,
                            keep_registration: bool = True,
                            keep_shifted: bool = True) -> tuple[int, int]:
    """Walk scratch and copy every non-skip file that isn't already at
    HDD (or is shorter at HDD than at scratch). Synchronous; no daemon
    thread; no time bound.

    Belt-and-braces safety net before bulk-rmtree. The continuous
    mirror's final pass races with the finalize's ``rmtree`` when the
    HDD is contending with the next row's preprocess writes -- the
    mirror exits via timeout but its in-flight copies are mid-write
    when the source dir gets nuked. Running this sweep on the
    finalize thread (which we control) guarantees that every output
    file is at HDD before we delete the scratch copy.

    Comparison is by ``st.st_size``: if HDD has the same size, assume
    up-to-date (the mirror already wrote it). Otherwise copy. We can't
    rely on mtime alone because ``shutil.copy2`` preserves mtime, so
    a partial copy + stat would falsely report "up to date".

    Hardlinks are preserved via the shared ``inode_to_dst`` map so the
    `_shared_reg/data.bin` <-> `final/data.bin` pair stays hardlinked
    at the destination.
    """
    if not scratch.exists():
        return 0, 0
    n_copied = 0
    bytes_copied = 0
    for dirpath, _dn, files in os.walk(str(scratch)):
        for fname in files:
            sp = Path(dirpath) / fname
            try:
                rel = sp.relative_to(scratch)
            except ValueError:
                continue
            if is_scratch_mirror_skippable(
                    rel, scratch,
                    keep_registration=keep_registration,
                    keep_shifted=keep_shifted):
                continue
            try:
                src_st = sp.stat()
            except OSError:
                continue
            dp = final / rel
            # Size check: if HDD already has it at full size, skip.
            # Otherwise (missing or short) re-copy.
            if dp.exists():
                try:
                    if dp.stat().st_size == src_st.st_size:
                        continue
                except OSError:
                    pass
            try:
                dp.parent.mkdir(parents=True, exist_ok=True)
                # Remove the short/stale HDD copy before re-copying;
                # copy_one_dedup doesn't overwrite.
                if dp.exists():
                    try:
                        dp.unlink()
                    except OSError:
                        continue
                copy_one_dedup(sp, dp, inode_to_dst)
                n_copied += 1
                bytes_copied += int(src_st.st_size)
            except OSError:
                continue
    return n_copied, bytes_copied


def mirror_sync_pass(scratch: Path, final: Path,
                     seen: dict, inode_to_dst: dict,
                     *, stability_window_s: float = 1.0,
                     throttle_bytes_per_sec: Optional[int] = None,
                     keep_registration: bool = True,
                     keep_shifted: bool = True
                     ) -> bool:
    """Walk ``scratch`` once and copy every file that's new or has
    changed since the previous pass to its mirror under ``final``.

    Returns True iff at least one file was copied this pass (the
    worker uses this to back off polling when caught up).

    ``seen`` maps ``Path -> (size, mtime)``. ``inode_to_dst`` is the
    same hardlink-dedup map ``copy_one_dedup`` uses elsewhere -- a
    file whose inode was already mirrored gets ``os.link``-ed at the
    destination instead of re-copied, so ``_shared_reg/data.bin`` <->
    ``final/data.bin`` stay hardlinked at the HDD twin.

    Files whose mtime is younger than ``stability_window_s`` are
    skipped -- they're likely mid-write (suite2p flushing ``data.bin``
    incrementally, e.g.). The next pass will catch them once writes
    settle.

    When ``throttle_bytes_per_sec`` is set, each copy is rate-limited
    via ``throttled_copy2`` -- used in single-drive mode so the
    mirror doesn't starve the active stage's reads on the same
    physical disk.
    """
    if not scratch.exists():
        return False
    now = time.time()
    changed = False
    for dirpath, _dn, files in os.walk(str(scratch)):
        for fname in files:
            sp = Path(dirpath) / fname
            try:
                rel = sp.relative_to(scratch)
            except ValueError:
                continue
            if is_scratch_mirror_skippable(
                    rel, scratch,
                    keep_registration=keep_registration,
                    keep_shifted=keep_shifted):
                continue
            try:
                st = sp.stat()
            except OSError:
                continue
            sig = (st.st_size, st.st_mtime)
            if seen.get(sp) == sig:
                continue
            # Stability check: if the file was modified within the
            # last ``stability_window_s`` it's probably still being
            # written. Skip and try next pass.
            if now - st.st_mtime < stability_window_s:
                continue
            dp = final / rel
            try:
                dp.parent.mkdir(parents=True, exist_ok=True)
                # Remove an out-of-date HDD copy before re-copying;
                # ``copy_one_dedup`` doesn't overwrite.
                if dp.exists():
                    try:
                        dp.unlink()
                    except OSError:
                        continue
                copy_one_dedup(
                    sp, dp, inode_to_dst,
                    throttle_bytes_per_sec=throttle_bytes_per_sec)
                seen[sp] = sig
                changed = True
            except OSError:
                continue
    return changed


def copy_one_dedup(sp: Path, dp: Path,
                   inode_to_dst: dict,
                   *,
                   throttle_bytes_per_sec: Optional[int] = None) -> None:
    """Copy or hardlink a single file from scratch to HDD, recording
    the (st_dev, st_ino) -> destination map so subsequent hits for
    the same inode become hardlinks at the destination instead of
    independent byte-for-byte copies. Raises on copy failure.

    When ``throttle_bytes_per_sec`` is set, the copy is rate-limited
    via :func:`throttled_copy2`. Hardlinks bypass the throttle --
    they're a metadata op, not bandwidth-bound.
    """
    try:
        st = sp.stat()
        key = (st.st_dev, st.st_ino) if st.st_ino else None
    except OSError:
        key = None
    existing = inode_to_dst.get(key) if key else None
    if existing is not None:
        try:
            os.link(existing, str(dp))
            return
        except (OSError, NotImplementedError):
            pass
    if throttle_bytes_per_sec is not None:
        throttled_copy2(str(sp), str(dp),
                        rate_bytes_per_sec=throttle_bytes_per_sec)
    else:
        shutil.copy2(str(sp), str(dp))
    if key is not None:
        inode_to_dst[key] = str(dp)


# ---------------------------------------------------------------------------
# Union PARAM_SPEC assembly
# ---------------------------------------------------------------------------


def build_batch_param_spec() -> list:
    """Assemble the union PARAM_SPEC.

    Entries are emitted in pipeline order (preprocess -> detection ->
    low-pass -> events -> clustering -> xcorr -> pipeline-wide) and
    each source group's label is rewritten with a numbered stage
    prefix so the Advanced dialog reads top-to-bottom in operation
    order. Tab 4's "Low-pass" cutoff knob (which Tab 4 binds to a
    slider, not a PARAM_SPEC entry) is inserted alongside the
    rest of the low-pass section.
    """
    # Original-group -> renamed-group. Anything not in the map keeps
    # its native label.
    rename = {
        # 1. Preprocess (Tab 1)
        "QC gif": "1. Preprocess - QC GIF",
        # 2. Detection (Tab 3)
        "Sparsery": "2. Detection - Sparsery",
        "Cellpose": "2. Detection - Cellpose",
        "Merge": "2. Detection - Merge",
        "dF/F": "2. Detection - dF/F",
        "Default low-pass": "2. Detection - Default low-pass",
        "Pixel scale": "2. Detection - Pixel scale",
        "GPU": "2. Detection - GPU",
        # 3. Low-pass + derivative (Tab 4)
        "Low-pass": "3. Low-pass - Cutoff",
        "Low-pass filter": "3. Low-pass - Butterworth",
        "Derivative": "3. Low-pass - SG derivative",
        "Slider bounds": "3. Low-pass - Slider bounds",
        # 4. Event detection (Tab 5)
        "Per-ROI hysteresis": "4. Events - Per-ROI hysteresis",
        "Display": "4. Events - Display",
        "Population events - density":
            "4. Events - Population density",
        "Population events - peaks":
            "4. Events - Population peaks",
        "Population events - baseline":
            "4. Events - Population baseline",
        "Population events - boundaries":
            "4. Events - Population boundaries",
        "Population events - gaussian fit":
            "4. Events - Population Gaussian fit",
        # 5. Clustering / 6. Xcorr / 7. Pipeline-wide added below
    }

    spec: list = []
    seen: set[str] = set()

    def _extend(items):
        for entry in items:
            name = entry.get("name")
            if not name or name in seen:
                continue
            new = dict(entry)
            grp = new.get("group", "Parameters")
            new["group"] = rename.get(grp, grp)
            spec.append(new)
            seen.add(name)

    # 1. Preprocess
    try:
        from ..preprocess.tab import PreprocessTab
        _extend(PreprocessTab.PARAM_SPEC)
    except Exception:
        pass

    # 2. Detection
    try:
        from ..suite2p.tab import Suite2pTab
        _extend(Suite2pTab.PARAM_SPEC)
        # GCaMP variant lives as a Tk var on Tab 3, not in its PARAM_SPEC;
        # surface it here so the per-row Edit params dialog can override
        # the tau passed to suite2p deconvolution. The label strings are
        # the single source of truth in Suite2pTab.GCAMP_OPTIONS, so we
        # import them rather than re-listing.
        _gcamp_labels = [lbl for lbl, _tau in Suite2pTab.GCAMP_OPTIONS]
        spec.append({
            "name": "gcamp_variant",
            "label": "GCaMP variant (sets suite2p tau)",
            "type": "choice",
            "choices": _gcamp_labels,
            "default": _gcamp_labels[0],
            "group": "2. Detection - dF/F",
            "help": "drives the calcium-decay tau passed to suite2p "
                    "deconvolution; pick the GECI you injected",
        })
        seen.add("gcamp_variant")
        # Optional custom tau override mirroring Tab 3's free-text field:
        # a positive value wins over the variant dropdown (for an indicator
        # not in the preset list or a rig-calibrated tau); 0 means "use the
        # variant dropdown". Applied onto Tab 3's ``custom_tau_var`` in
        # ``_apply_row_params_to_tabs`` so the existing ``_on_run`` override
        # path handles it.
        spec.append({
            "name": "gcamp_tau_custom",
            "label": "Custom tau (s) -- overrides variant",
            "type": "float",
            "default": 0.0,
            "group": "2. Detection - dF/F",
            "help": "positive value overrides the GCaMP variant tau passed "
                    "to suite2p deconvolution; 0 = use the variant dropdown",
        })
        seen.add("gcamp_tau_custom")
    except Exception:
        pass

    # 3. Low-pass: cutoff first (binds to the slider value), then the
    # Tab 4 PARAM_SPEC entries for filter / derivative / slider bounds.
    spec.append({
        "name": "cutoff_hz", "label": "Low-pass cutoff (Hz)",
        "type": "float", "default": 1.0,
        "group": "3. Low-pass - Cutoff",
        "help": "cutoff applied to every kept ROI when batch runs Tab 4",
    })
    seen.add("cutoff_hz")
    try:
        from ..lowpass.tab import LowpassTab
        _extend(LowpassTab.PARAM_SPEC)
    except Exception:
        pass

    # 4. Event detection
    try:
        from ..event_detection.tab import EventDetectionTab
        _extend(EventDetectionTab.PARAM_SPEC)
    except Exception:
        pass

    # Helper for the remaining synthesized-entry blocks: append() each
    # entry through the same dedup gate ``_extend`` uses so we don't
    # silently duplicate a name that an earlier PARAM_SPEC already
    # contributed. (Background: pre-2026-05-16 this block used raw
    # ``spec.extend([...])``, which bypassed ``seen`` and produced two
    # ``baseline_mode`` entries -- Tab 5's choice [rolling, global] and
    # the synthesized Tab 3 choice [first_n, rolling]. AdvancedDialog
    # built two form rows that both wrote to the same key, and the
    # last-seen default ('first_n') leaked into Tab 5's row where it
    # wasn't a valid choice. The fix here renames the Tab 3 synthesized
    # entries to dff_baseline_mode / dff_baseline_min and routes all
    # new appends through the dedup gate so the bug can't recur.)
    def _add_synthesized(entries):
        for entry in entries:
            name = entry.get("name")
            if not name or name in seen:
                continue
            spec.append(entry)
            seen.add(name)

    # 5. Clustering (no source PARAM_SPEC)
    _add_synthesized([
        {"name": "prefix", "label": "dF/F prefix",
         "type": "str", "default": "r0p7_filtered_",
         "group": "5. Clustering",
         "help": "memmap prefix; filtered_ uses the cell-filter mask"},
        {"name": "threshold",
         "label": "Cluster threshold (0=auto)",
         "type": "float", "default": 0.0, "group": "5. Clustering",
         "help": "linkage cut height; 0 -> auto via auto_choose_threshold"},
        {"name": "palette", "label": "Cluster palette",
         "type": "str", "default": "tab10", "group": "5. Clustering"},
    ])

    # 6. Cross-correlation (no source PARAM_SPEC)
    _add_synthesized([
        {"name": "max_lag_seconds", "label": "xcorr max lag (s)",
         "type": "float", "default": 2.0,
         "group": "6. Cross-correlation"},
        {"name": "zero_lag", "label": "Compute zero-lag correlation",
         "type": "bool", "default": True,
         "group": "6. Cross-correlation"},
        {"name": "use_gpu", "label": "Use GPU for xcorr",
         "type": "bool", "default": True,
         "group": "6. Cross-correlation"},
    ])

    # 7. Pipeline-wide -- Tab 3's dF/F baseline (suite2p detection).
    # Distinct name from Tab 5's Population events ``baseline_mode``
    # (choices ["rolling", "global"]) so the two never collide in the
    # row.params dict or the AdvancedDialog form.
    _add_synthesized([
        {"name": "dff_baseline_mode", "label": "dF/F baseline mode",
         "type": "choice", "choices": ["first_n", "rolling"],
         "default": "first_n", "group": "7. Pipeline-wide",
         "help": "suite2p dF/F baseline: first_n = first-N-minute mean; "
                 "rolling = 45 s window, 10th percentile"},
        {"name": "dff_baseline_min", "label": "Baseline duration (min)",
         "type": "float", "default": 2.0, "group": "7. Pipeline-wide",
         "help": "first-N-min baseline length when dff_baseline_mode=first_n"},
    ])

    return spec
