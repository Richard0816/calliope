# Tab 8 — Spatial propagation

Per-event activation-order ("coactivation") spatial maps + frame-to-frame
propagation vectors. For every population event detected by **Tab 5**,
ROIs that fired earliest are coloured cyan and those that fired latest
are red. Click through events with the spinner or Prev / Next buttons.

Three figures, laid out as a top row of two side-by-side activation-order
maps with a three-subplot directional-monotonicity analysis underneath:

1. **Top-left — activation order (plain).** The cyan→red spatial map
   painted from per-ROI first-onset rank, with no overlays — useful as
   a clean reference image for figures.
2. **Top-right — activation order + frame-to-frame arrows.** Same map,
   with white arrows overlaid that connect the centroid of the ROIs
   activating in each frame to the centroid for the next active frame.
3. **Bottom — directional monotonicity (Spearman ρ vs projection angle).**
   Tests whether the event's activation has a coherent direction of
   propagation, and how strong / significant that direction is. Three
   subplots:
   - **(a)** Scatter of active ROI positions coloured by activation
     frame; black arrow along `u(θ*) = (cos θ*, sin θ*)` with length
     scaled by `ρ_obs` (no arrow when `ρ_obs ≤ 0`).
   - **(b)** `ρ(θ)` curve across θ ∈ [0°, 360°) with the maximum
     marked.
   - **(c)** Permutation null histogram of `ρ_max` from shuffled
     activation times, with `ρ_obs` overlaid as a vertical red line
     and the empirical p-value in the subplot title.

   The maths (helper `core.spatial.directional_monotonicity_spearman`):

   ```
   for θ in linspace(0, 2π, 360):
       s_i(θ) = x_i cos θ + y_i sin θ           # 1-D projection
       ρ(θ)   = SpearmanCorr(s_i(θ), t_i)        # rank-only correlation
   θ* = argmax_θ ρ(θ);   ρ_obs = ρ(θ*)

   # permutation null (corrects for the multiple-θ search):
   for k in range(N_shuffles):
       t_perm    = permute(t)
       ρ_null[k] = max_θ SpearmanCorr(s_i(θ), t_perm)
   p = mean(ρ_null >= ρ_obs)
   ```

   ρ_obs near 1 = perfectly monotone propagation along `θ*`; near 0
   = no directional structure. The null is computed by sweeping θ
   on every shuffle and recording its maximum, so the p-value
   accounts for the same multiple-direction search as the observed
   statistic (otherwise random data with `n` cells and 360 θ tests
   would look spuriously significant).

   Replaces the pre-2026-05-10 "pairwise distance vs Δframe" violin,
   which described spread but never told you whether the event had
   a *direction* at all.

   The same scalars are stamped into the workbook's
   `EventMonotonicity` sheet (one row per event:
   `event_id, n_active, theta_star_deg, rho_obs, p_value,
   p_value_fdr, n_shuffles, u_x, u_y`) by `core.event_detection_run`
   whenever Tab 5 (or the headless runner / Tab 0 batch) finishes,
   so cross-recording analyses can stack monotonicity stats next to
   the existing `EventWindows` / `EventOnsets` tables.
   `p_value_fdr` is the Benjamini-Hochberg-adjusted q-value across
   the per-event tests in this recording — use this column (not the
   raw `p_value`) when calling significance across the recording's
   event ensemble.

## Where the data comes from

This tab is a **pure consumer of Tab 5's last render**. It does not
re-detect onsets or population events. Tab 5 publishes the results of
each Render via `AppState.set_event_results({...})`; this tab subscribes
in `__init__` and re-paints whenever a new payload arrives.

The published payload contains:

- `plane0` — `Path` to the Suite2p plane0 folder.
- `event_windows` — `(n_events, 2)` array of `(t0, t1)` seconds.
- `A` — `(N_iter, n_events)` boolean active-in-event matrix.
- `first_time` — `(N_iter, n_events)` first-onset time per ROI per event
  (NaN if the ROI didn't participate).
- `kept_idx` — Suite2p ROI ids that Tab 5 actually iterated over (the
  cell-filter keep mask, intersected with the manual ROI subset if the
  user enabled it on Tab 5).
- `fps`, `T`, `onsets_by_roi`.

The matrices' first dimension matches `kept_idx` exactly, so Tab 8 can
translate ROI rows back to Suite2p ids without any extra bookkeeping.

## Advanced settings

**Tab 8 has no Advanced dialog and no `PARAM_SPEC`.** It is a pure
*consumer* of Tab 5's last `event_results` publish — it runs no
detection of its own. Every knob that shapes what you see here lives on
**Tab 5 (Event detection)**: the dF/F filtering, the hysteresis
on/off thresholds, the population-event windowing, and the manual ROI
subset. Whatever Tab 5 detected with those parameters is exactly the
`A` / `first_time` / `event_windows` matrices Tab 8 paints. To change
the events, the participating ROIs, or the onset times, re-render on
Tab 5; Tab 8 re-paints automatically. See
[`tabs/event_detection/README.md`](../event_detection/README.md) for
that tab's full parameter reference.

The directional-monotonicity test's internals (`n_angles=360`,
`n_shuffles=10000`, `seed=0`, passed to
`spatial_helpers.directional_monotonicity_spearman` at
[`tab.py:945`](tab.py)) are **hardcoded**, not user-tunable — they are
fixed so the permutation p-value is reproducible across recordings.
(`spatial_helpers` is `logic.py`'s re-export shim for `core.spatial`,
so this is the same helper as the `core.spatial` one named above.)

The only user-tunable control on Tab 8 itself is one inline toggle:

| Setting (`key`) | Default | What it does | What it means to you |
| --- | --- | --- | --- |
| Use Suite2p spks (`_use_spks_var`) | `False` (unchecked) | **What it does.** A `BooleanVar` checkbox ([`tab.py:184`](tab.py)). Unchecked, the per-ROI first-activation time is Tab 5's hysteresis-onset estimate (the published `first_time` / `A`). Checked, `_apply_spks_override` ([`tab.py:535`](tab.py)) loads `spks.npy` via `core.utils.s2p_load_spks` (auto-orients to `(T, N_kept)`, subset to `kept_idx`), and `_first_time_from_spks` ([`tab.py:600`](tab.py)) recomputes activation per event window `[t0, t1]`: it slices frames `[floor(t0·fps), ceil(t1·fps))`, finds the **first frame with `spks > 0`** per ROI (`argmax` over a boolean mask), and writes that as `first_time`; ROIs with no deconvolved spike in the window are marked inactive (`A=False`, `first_time=NaN`). It mutates `self._data["A"]` / `["first_time"]` in place; unchecking restores the stashed Tab 5 matrices from `_last_results`. Requires a valid `fps` and an on-disk `spks.npy`, or it silently reverts. | Flip this when you trust Suite2p's OASIS deconvolution more than the hysteresis-onset detector — e.g. when dF/F is noisy and the Schmitt-trigger onset times look jittery, or to sanity-check that the propagation direction is robust to the activation-time definition. Spks onsets are typically **sparser and crisper** (one frame per spike), so expect fewer participating ROIs per event and a cleaner cyan→red gradient; the monotonicity ρ and p-value will shift accordingly. Gotchas: it only redefines *timing within already-detected windows* — it does **not** re-detect events, so the event count and `[t0, t1]` bounds are unchanged; it needs `spks.npy` in the plane0 folder and a known `fps` (the toggle auto-unchecks with a status-bar message if either is missing); and the override is what gets written when you **Export figures**, recorded as `use_spks_override` in the export manifest. Parallels the `spks` signal-kind radio on Tab 5 (set both the same way if you want both tabs reading the deconvolved signal), but the activation rule differs: Tab 5 reruns MAD-z + hysteresis on `spks.npy`, whereas Tab 8 here just takes the first non-zero spks frame in the window — so the two won't produce identical onset times even with both set to spks.

Event navigation (Prev / Next buttons, the `ttk.Spinbox` bound to
`event_var`) is not a setting — it only selects which already-computed
event window to display.

## Plane0 metadata loaded locally

Tab 5 doesn't ship `stat`/`ops` (they're large and the consumer can
load them itself), so on each ingest Tab 8 reads:

- `stat.npy` — Suite2p ROI pixel masks for `paint_spatial`.
- `ops.npy` — `Ly`, `Lx`, and optional `pix_to_um`.

Both are cached per `plane0` to avoid re-reading on every Tab 5 publish.

## Per-event painting (one navigation click)

For the selected event index `k`:

- `order_rank = order_map_for_event(first_time[:, k], A[:, k])` →
  ranks 1..K for active ROIs by first-onset time, NaN for inactives.
- `paint_order_map(order_rank, stat_filtered, Ly, Lx)` paints those
  ranks normalised to `[0, 1]` onto an `(Ly, Lx)` canvas using
  `core.utils.paint_spatial`. Pixels outside any ROI are NaN so the
  colormap's `set_bad` colour shows them as grey.
- `event_frame_centroids(first_time[:, k], A[:, k], stat_filtered, fps)`
  groups the ROIs that participated by their first-onset frame
  (`round(first_time * fps)`), and for each frame group computes:
  - centroid `(cx, cy)` = mean of contributing ROI centroids
    (`median(stat[i]['xpix'])`, `median(stat[i]['ypix'])`),
  - per-axis std `sigma_x`, `sigma_y` (zero when `n == 1`),
  - 2D RMS spread `sigma_xy = sqrt(sigma_x^2 + sigma_y^2)`.
  Frames with no activations are skipped, so the returned list is
  contiguous in *active* frames, not absolute frame index.
- Top-left and top-right: identical `imshow` (with
  `cmap=CYAN_TO_RED`, `vmin=0`, `vmax=1`, `origin="upper"` plus a
  flipped y-extent so the FOV matches suite2p's GUI orientation, axes
  in µm if `ops["pix_to_um"]` is present, else pixels) — built by a single
  inner helper that takes a `with_arrows` flag. The right-hand variant
  adds white-with-black-edge centroid dots and white `arrowstyle="->"`
  arrows connecting successive frame groups.
- Bottom: blank axis matched to the same FOV extent, centred in a 1:2:1
  inner-grid so it sits horizontally aligned with the top pair instead
  of stretching to the full window width. Each frame group becomes a
  `Circle` patch coloured by its absolute frame index via the same
  `CYAN_TO_RED` cmap (linear in `frame - f_min`, so non-contiguous
  active frames stay positioned correctly on the colour scale). Black
  arrows connect successive groups. A vertical colorbar on the right
  reports the frame range as ticks (`f_min`, midpoint, `f_max`); no
  per-circle text labels. The circle radius is `0.20 * sigma_xy *
  scale`, clamped to `[0.4%, 4%]` of the largest FOV dimension —
  `sigma_xy = sqrt(var_x + var_y)` is roughly 1.41σ in 2D, and
  rendering it at full scale blots out the FOV for diffuse frames, so
  we shrink to ~28% of one std and cap it.

All helpers (`order_map_for_event`, `paint_order_map`,
`event_frame_centroids`, `CYAN_TO_RED`) live in
[`core/spatial.py`](../../core/spatial.py) so other tabs / scripts can
paint the same maps.

## Headless figure rendering

Tab 8 has no analysis to run headlessly — every panel is a pure render
of the data Tab 5 publishes. For batch mode, `core/spatial.py`
exposes `render_spatial_event_figures(plane0, event_data, *,
figures_dir)` which loops over every event in `event_data['event_windows']`
and saves the order-map-with-arrows panel (the most informative single
view) as `event_001.png`, `event_002.png`, … in `figures_dir`. The
`event_data` dict is exactly what `core.event_detection_run.run_event_detection`
returns (`event_windows`, `A`, `first_time`, `kept_idx`, `fps`), so the
batch orchestrator hands one stage's output straight to the next.

The interactive Tab 8 still owns the full four-panel view + directional
monotonicity analysis and listens to `state.event_results` — those panels stay in
`tabs/spatial_propagation/tab.py`.

## UI

- **Status label** — tells the user when Tab 5 events are available
  and how many.
- **Refresh from Tab 5** — re-pulls the most recent
  `state.event_results` (useful after manually loading a saved
  recording).
- **Prev / Next + Spinbox** — event navigation. `<Return>` and
  `<FocusOut>` on the spinbox commit typed values. The Prev / Next
  buttons disable at the ends.

When Tab 5 publishes a new payload, the tab keeps the current event
index if it's still valid (so re-rendering Tab 5 with new parameters
doesn't yank you back to event 1 unless the event count drops).

## Re-implementation checklist

1. Implement the cyan→blue→red colormap and the
   `order_map_for_event` / `paint_order_map` pair (see
   `core/spatial.py`). Painting just delegates to
   `core.utils.paint_spatial` with rank-normalised values; the only
   subtle bit is using a coverage map to mark non-ROI pixels NaN so
   the colormap renders them grey.
2. Implement `event_frame_centroids` to group active ROIs by their
   first-onset frame (round `first_time * fps`) and emit
   `(cx, cy, sigma_xy)` per frame.
3. Subscribe to `AppState.event_results` (the publish-side is in
   Tab 5's `_on_render_done`). Translate the published `kept_idx`
   into `stat_filtered` by indexing the locally-loaded `stat.npy`.
4. Paint two figures per event on demand (one for the order map +
   arrow overlay, one for vectors-only with std circles). Rebuild
   each figure on every render so colorbars don't stack.
5. Use a `Spinbox` + Prev / Next buttons for navigation. Honour
   `ops["pix_to_um"]` if present (axes in µm), else fall back to
   pixels — both panels share the same scale factor.

## On-disk outputs

None in v1. Everything is in-memory. Saving per-event PNG batches and
a `coactivation_summary.csv` (per-ROI timing per event) is on the
roadmap and will mirror the layout written by the legacy
`spatial_heatmap_updated.coactivation_maps`.


## UI affordances

Tab 8 inherits the global customtkinter dark theme from `pipeline_gui`.

- **Three stacked matplotlib panels** (Activation order map, Activation order with arrows, Monotonicity panels) — matplotlib figures keep their white facecolor; navigation toolbars below each are dark-skinned.
- **Event navigation row.** Prev / Next `CTkButton`s + a `ttk.Spinbox` (no CTk spinbox equivalent) to step through events.
- **No popouts.** The Refresh button re-reads the event list from Tab 5 / from the EventWindows sheet on disk.
