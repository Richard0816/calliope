# One `output/` folder per recording, with per-item save toggles

Status: spec + first slice (2026-10-15). Branch `output-folder-redesign`
(off `main` db3f570). Decisions below were made by Richard on 2026-10-15
in answer to coworker feedback that "getting the data from each tab is
confusing: everything is in different places and it isn't clear what
each button saves".

## Decisions (locked)

| Question | Decision |
|---|---|
| Where | `<recording>/output/`, next to `detection/`. Everything else in the recording folder is internal (plane0 is what CalLIOPE reads, not something users browse). |
| Layout | By kind: `output/calliope_summary.xlsx`, `output/README.txt`, `output/figures/<tab>/…`, `output/data/<tab>/…`, `output/<rec>.nwb`. |
| Granularity | One toggle per item. Each plot has two: its image (PNG+SVG) and its data. |
| Format | CSV by default; one global "compact format" switch writes `.npy` instead for the big T×N traces. |
| Trace layout | Time down, ROIs across: `time_s`, then `roi_<suite2p id>` columns. |
| Defaults | Small tables on; big traces and plot data off. (Images: on, matching today's batch "Save figures" default. Not explicitly decided; flagged for review.) |
| Where toggles live | Full checklist in Tab 0 settings + a per-tab "Data to save" dropdown listing that tab's items; both edit the same setting. |
| Persistence | Remembered across sessions (`~/.calliope/exports.json`); each run's manifest records what it wrote. |
| When written | Automatically whenever a stage finishes (interactive and batch). A newly ticked item appears on the next stage run; the existing buttons stay as "save now". |
| Old runs | Migrated on open: `plane0/calliope_summary.xlsx` and `calliope_figures/` are moved into `output/` once, with a log line. |
| NWB | Moves to `output/`. |
| `C*_rois.npy` | Stays internal (Tab 7 reads it); a cluster-membership CSV in `output/data/` is the user copy. |

## Catalogue (first version)

Key = setting name. "img" items write PNG+SVG to `output/figures/<tab>/`;
"data" items write to `output/data/<tab>/`. Default in brackets.

| Tab | Key | Item | Kind | Default |
|---|---|---|---|---|
| 1 | `qc_movie` | QC movie (`qc.gif`) | img | on |
| 3 | `rois_table` | ROI table: id, centroid, area, cell prob, kept | data | on |
| 3 | `roi_maps` | All / kept ROI maps | img | on |
| 3 | `trace_f` | Raw F, kept ROIs (T×N) | data, big | off |
| 3 | `trace_fneu` | Neuropil Fneu, kept ROIs | data, big | off |
| 3 | `trace_dff` | Filtered dF/F, kept ROIs | data, big | off |
| 3 | `trace_spks` | Deconvolved spks, kept ROIs | data, big | off |
| 3 | `zdrift` | z-drift PC1 trace + QC plot | data + img | on |
| 4 | `lowpass_plots` | FFT / raw / low-pass plots | img | on |
| 4 | `lowpass_plot_data` | Data behind those three plots | data | off |
| 4 | `trace_lowpass` | Low-pass dF/F, kept ROIs | data, big | off |
| 4 | `trace_dt` | Derivative of low-pass dF/F | data, big | off |
| 5 | `event_windows` | Event windows (start, end, duration, n active, class) | data | on |
| 5 | `event_onsets` | Per-ROI onsets (event id, roi, onset, relative) | data | on |
| 5 | `event_monotonicity` | Per-event propagation direction stats | data | on |
| 5 | `event_plots` | Heatmap / raster / density plots | img | on |
| 5 | `event_density_data` | Density, baseline, threshold curves | data | off |
| 5 | `event_heatmap_data` | Heatmap matrix (time × ROI) | data, big | off |
| 5 | `event_raster_data` | Onset raster (time × ROI, 0/1) | data, big | off |
| 6 | `cluster_membership` | ROI → cluster id (visual order) | data | on |
| 6 | `cluster_plots` | Dendrogram / spatial cluster map | img | on |
| 6 | `cluster_mean_traces` | Mean dF/F per cluster (T×K) | data | off |
| 7 | `xcorr_pairs` | Pair tables (lag, r, p, q) per cluster pair | data | on |
| 7 | `xcorr_plots` | Violin plots | img | on |
| 7 | `xcorr_violin_data` | Data behind the violins | data | off |
| 8 | `propagation_maps` | Per-event order maps | img | on |
| 8 | `activation_times` | Per-ROI activation time per event (N×E) | data | off |
| – | `nwb` | NWB file | other | off |

`calliope_summary.xlsx` stays the one-stop workbook (always written)
and moves to `output/`.

## Plan

1. **Core (this slice):** `core/output_layout.py` — path helpers
   (`output_dir`, `figures_dir(tab)`, `data_dir(tab)`), the catalogue,
   load/save of the toggle set, `write_trace` (CSV or .npy, time-down),
   `write_table`, `README.txt` writer, and `migrate_run(save_folder)`.
   Unit tests.
2. **Workbook:** `summary_writer.summary_path` → `output/`; readers
   (`nwb_export`, Tab 7 EventWindows glob, `run_loader`) look in
   `output/` first, then the old location.
3. **Figures:** every `save_folder / "calliope_figures"` literal →
   `output_layout.figures_dir`; manifest → `output/manifest.json`
   (readers fall back to the old path).
4. **Data writers:** one `write_stage_outputs(stage, plane0, data)` per
   tab, called at the end of each stage (interactive + batch), honouring
   the toggles. Tab 3's "Also write filtered dF/F as CSV" becomes the
   `trace_dff` toggle.
5. **UI:** Tab 0 "Data to save" section (grouped checklist + compact
   switch + presets "Small tables" / "Everything" / "Nothing extra");
   per-tab dropdown (a `CTkOptionMenu`-style checklist) bound to the
   same store; status line names the files written.
6. **Migration on open:** `_on_open_run` calls `migrate_run` before
   loading; log what moved.
7. **Docs + scripts:** READMEs, walkthrough, `Calliope Full.md`, and the
   three `scripts/` that read `calliope_figures/`.

Overlap with `ux-audit-fixes` (unmerged): it changes
`summary_writer.write_sheets` (file lock) and Tab 5's summary write
(background process). Merge that branch first, or expect conflicts in
those two places.
