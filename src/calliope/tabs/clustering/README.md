# Tab 6 — Clustering

**Goal.** Group cells whose dF/F traces co-vary, render the result as a dendrogram + spatial map, and export each cluster's ROI list to disk for Tab 7.

---

## 1. Inputs

From `plane0/`:
- `r0p7_filtered_dff.memmap.float32` — `(T, N_kept)` filtered dF/F (default prefix `r0p7_filtered_`).
- `F.npy` — `(N_total, T)` (only used to read shape).
- `stat.npy` — list of per-ROI dicts (`xpix`, `ypix`, `lam`, …).
- `ops.npy` — provides `Lx`, `Ly` for the spatial map.
- `predicted_cell_mask.npy` (or `iscell.npy`) — keep mask, used to translate filtered-list indices to Suite2p ROI indices on export.

The user can change the prefix in the GUI to operate on `r0p7_dff.memmap.float32` (unfiltered) instead, but the default and the rest of the pipeline assume the filtered view.

---

## 2. Pipeline steps

> **Headless entry point.** `core/clustering.py:run_clustering(plane0,
> params, *, figures_dir=None, write_summary=True)` runs the linkage,
> picks a threshold (auto via `clustering.auto_choose_threshold` when
> `params['threshold']` is `None`), cuts to flat clusters, exports
> `C{i}_rois.npy` (Suite2p ids) + `linkage.npy` + `linkage_roi_ids.npy` +
> `threshold_used.npy` + `_indices_are_suite2p`, and (optionally) renders `dendrogram.png` +
> `spatial_clusters.png`. Filtered-position → Suite2p translation uses the
> same memmap-anchored keep mask `load_filtered_dff` sized the matrix with
> (not the live `predicted_cell_mask`/`iscell`), so the marker is always
> truthful. Tab 6 still owns the interactive recluster / palette / picker
> UI. Tab 0's GUI batch runner drives Tab 6 itself (`apply_batch_row` →
> Run → Export); `run_clustering` is what `core.batch_pipeline.run_recording`
> and scripts call.

### Step 1 — Distance matrix

`ward_linkage(dff, method='ward')` (single implementation in `core/clustering.py`, re-exported by `tabs/clustering/logic.py`):

```python
dff_z = (dff − mean(dff, axis=0)) / (std(dff, axis=0) + 1e-8)   # z-score each column
dist  = pdist(dff_z.T, metric='euclidean')                       # Euclidean on z-scored data
Z     = linkage(dist, method='ward')                             # Ward / min-variance
```

Z-scoring kills amplitude differences between cells, so two ROIs with very different baseline brightness but the same activity *shape* land close together. On z-scored data, `||x − y||² = 2T · (1 − Pearson r(x, y))` — squared Euclidean and `1 − r` are equivalent up to a constant. **Pair-ranking is identical to correlation distance**; we use Euclidean because scipy's Ward implementation requires it.

The output `dist` is a condensed `(N·(N−1)/2,)` array fed straight to `scipy.cluster.hierarchy.linkage`.

#### Why Ward linkage?

- **Single linkage** chains cells through any tenuous correlation; produces giant straggly clusters.
- **Complete linkage** is too strict; tends to break true ensembles into many tiny groups.
- **Average (UPGMA)** merges by mean pairwise distance. On unbalanced data (one big mode + several small ones) it tends toward "1 huge cluster + a few tiny outliers". The earlier per-event xcorr fragmentation guard was a workaround for exactly this failure mode.
- **Ward** minimises within-cluster sum-of-squares (Lance-Williams). Produces compact, balanced clusters — the "4–5 modes" shape lab members find useful. Z-scoring removes the only objection to Ward (its Euclidean requirement); previous reluctance was based on a misreading of the math.

(Switched from average + correlation → Ward + Euclidean on 2026-05-11 after user pointed out that z-scoring already kills amplitude differences. The function is now named `ward_linkage`; there is no `_correlation_linkage`. Because the merge heights are Ward distances, the dendrogram y-axis, slider and recluster entry are labelled "Ward linkage distance", not `1 − r`.)

### Step 2 — Auto-pick a cut

`clustering.auto_choose_threshold(Z, target_counts=(4, 5))` walks candidate thresholds top-down and returns the fraction of `max(Z[:, 2])` that yields a cluster count in `[4, 5]`. It's a cheap heuristic — always inspect the dendrogram and override with the slider if needed.

The slider operates in **absolute Ward distance** (sum-of-squares units). The auto fraction is converted at render time: `T_auto = auto_frac · zmax`. The auto-pick counts via `fcluster` directly (post-2026-05-11) so the displayed cluster count always matches the spatial map.

### Step 3 — Render

Every render pass (slider drag, palette change, manual toggle) goes through `_render_all`:

```python
T          = current_threshold()
labels     = fcluster(Z, t=T, criterion='distance')   # raw label per leaf
n_clusters = unique(labels).size
```

Then a sequence of bookkeeping that **bypasses scipy's built-in palette cycling** so cluster numbering and colours stay consistent across renders:

1. `_compute_visual_cluster_map(T)`: walk leaves in dendrogram order; map "first label seen" → C1, "second" → C2, … This gives a stable visual ordering regardless of `fcluster`'s internal label numbering.
2. `_build_label_color_map(n_clusters)`: assign palette slot `i` (or the user's custom hex) to visual cluster `i+1`. Returns `{raw_label: hex_color}`.
3. `_make_link_color_func(T, labels, label_to_color)`: a closure that, for any internal node of `Z`, returns the colour of its descendant leaf's cluster (or `gray` if above the cut). Passed to `dendrogram(link_color_func=...)`.

#### Dendrogram

```python
dendrogram(Z, ax=ax, link_color_func=link_color_func,
           above_threshold_color='gray', no_labels=True)
ax.axhline(T, '--', linewidth=1.5, color='black')
```

#### Spatial map

For each leaf `i`, look up `hex = label_to_color[labels[i]]`, convert to RGB. Then `paint_spatial` (`utils.paint_spatial`) blends per-ROI colours onto the FOV using each ROI's `lam` weights:

```python
img[ypix, xpix] += v · lam              # weighted accumulation
w[ypix, xpix]   += lam
img[w>0] /= w[w>0]                      # normalize
```

Done independently for R, G, B channels. Background (no ROI coverage) is `NaN` so matplotlib draws it transparent over the figure background.

**Axes in µm + scale bar.** When the run carries a pixel calibration (`<plane0>/calliope_calibration.npy` → `utils.load_pix_to_um`, resolved by `_resolve_pix_to_um`), the live spatial map (and the `ReclusterWindow` sub-map) sets an `imshow` extent of `[0, Lx·pix, Ly·pix, 0]`, labels the axes `x (µm)` / `y (µm)`, and adds a black scale bar. Without a calibration it keeps the prior `x (px)` / `y (px)` labels and no bar. Both render paths go through the shared `logic.render_spatial_map(ax, img, Lx, Ly, pix_to_um, title)` helper so they behave identically; this matches the already-µm batch export (`core.clustering.plot_spatial`) and the Tab 8 maps.

A parallel `(Ly, Lx)` int label image (`logic.build_label_image`) encoding `filtered_idx + 1` per painted pixel is also rebuilt every render so the spatial-click handler can resolve a cursor position back to an ROI without a per-pixel scan over `stat`. Because the axes are in µm when calibrated, the click handler divides the cursor `(xdata, ydata)` by `pix_to_um` before indexing the (pixel-space) label image. Click any coloured ROI to open the cluster popout (see §"Cluster popout").

### Step 4 — Per-cluster colours

The dropdown lists `clustering.AVAILABLE_PALETTES` (categorical: `tab10`, `tab20`, `Set1`, `Set2`, `Set3`, `Paired`, `Accent`, …; continuous: `viridis`, `plasma`, `magma`, `inferno`, `cividis`, `Spectral`, …). `resolve_palette(name, n_colors)` returns `n_colors` hex strings.

The per-cluster custom dialog (`CustomColorDialog`) overrides the dropdown. Custom hex strings persist until the user clicks "Reset palette."

### Step 4b — Picked-cluster ROI list (copy to clipboard)

A small read-only entry under the recluster row shows the **Suite2p ROI ids** of every cluster currently ticked in the picker, formatted compactly via `gui_common.format_roi_indices` (`[1,2,3,7,10,11] → "1-3,7,10-11"`). A "Copy" button copies the string to the system clipboard so you can paste it into:

- Tab 4's "Manual ROI(s)" entry,
- Tab 5's "Restrict to manual ROI subset" entry,
- or any external script.

Computation (`_picked_cluster_roi_ids`):

```python
T            = current_threshold()
labels       = fcluster(Z, t=T, criterion='distance')
target_lbls  = [_visual_to_label[v] for v in picked_visual_ids]   # raw labels
local_idx    = where(isin(labels, target_lbls))[0]                # filtered-list positions

if 'filtered' in prefix.split('_'):
    translator = filtered_to_suite2p_indices(plane0, prefix)   # filtered-pos -> Suite2p id
    if translator is not None and translator.size == len(labels):
        return sorted(translator[local_idx].tolist())
return sorted(local_idx.tolist())
```

> **Keep-mask anchoring.** The filtered-position → Suite2p translator comes from
> `filtered_to_suite2p_indices(plane0, prefix)`, which resolves the keep mask via
> `utils.resolve_filtered_mask` **anchored to the `<prefix>dff.memmap.float32` file
> size** — the same resolver `load_filtered_dff` and `stat_for_prefix` use. The old
> `_load_filter_mask` read the *live* `predicted_cell_mask`/`iscell`; once curation
> drifted those after the memmap was written, its `sum()` no longer matched the
> linkage's leaf count, so the spatial map failed the `len(stat) != N` guard with
> **"Spatial unavailable (stat / dF/F mismatch)"** and the export/ROI-id paths
> silently fell back to emitting filtered-list positions instead of Suite2p ids.
> Anchoring to file size keeps `stat`, the dF/F columns, and the translator the
> exact same length. The headless `core.clustering.run_clustering` and its
> `plot_spatial` (via `stat_for_prefix`) use the same anchored mask; the
> live-mask readers `_load_filter_mask` / `load_filter_mask` were removed.

Refresh hooks:
- `_update_cluster_pick_label` (fires on every menu toggle) → calls `_refresh_roi_list_var`.
- `_render_all` (fires on threshold drag, palette change, manual toggle) → calls `_refresh_roi_list_var` after the dendrogram + spatial map are redrawn.

The Copy button is disabled whenever no clusters are picked (or analysis hasn't run); enabled otherwise. Status bar prints `Copied <N> ROI ids to clipboard.` after a successful copy.

The clipboard call is `self.clipboard_clear() + self.clipboard_append(text) + self.update()`. The trailing `update()` is required on Windows to make the copy survive after the Tk app exits.

### Step 5 — Reclustering a branch

The "Pick clusters…" menu lists C1..Cn with each entry painted in its current colour (and a luminance-aware foreground via `_readable_fg`). Pick one or more, set a new threshold, click "Recluster (new window)":

```python
target_labels = [visual_to_label[v] for v in visual_picked]
mask          = isin(labels, target_labels)
global_idx    = where(mask)[0]
dff_subset    = self._dff[:, global_idx]
Z_sub         = ward_linkage(dff_subset)
```

A separate `ReclusterWindow` opens with its own dendrogram + slider + spatial map. The parent state is untouched. Recluster windows pick a different palette (cycling through `CATEGORICAL_PALETTES`) so their colours visibly differ from the parent.

### Step 5b — Reload existing clusters

A "Reload clusters" button next to "Run analysis" restores a previous Export run without recomputing the linkage (which is the expensive part — `pdist` + Ward linkage is O(N²·T) and can be 30 s to a few minutes per recording).

The button is enabled only when both files exist for the active prefix:

```
<plane0>/{prefix}cluster_results/linkage.npy
<plane0>/{prefix}cluster_results/threshold_used.npy
```

`logic.reload_clustering_offload(plane0, prefix, cluster_dir)` (run in a child process via `core.offload.run_offloaded`, like Run analysis's `compute_clustering_offload`):

1. `Z = np.load(linkage.npy)` — the saved linkage matrix.
2. `thr = float(np.load(threshold_used.npy)[0])` — the cut at which the user clicked Export.
3. `_, T, N, _ = load_filtered_dff(...)` — only to read the current column count.
4. **Validation:** `Z.shape[0] + 1 == N`, and the ordered Suite2p ids the linkage was built on (`linkage_roi_ids.npy`; for older exports, the sorted union of the `C*_rois.npy` files when the `_indices_are_suite2p` marker is present) must equal the current kept ids (`core.clustering.linkage_roi_ids`). A promotion can swap kept ROIs at the same count (`[0,1,3]` → `[0,2,3]`), so the count alone is not enough. On a mismatch the reload aborts with a clear error rather than silently misaligning ROIs to dendrogram leaves.
5. `Lx`, `Ly` from `utils.load_plane_view`.
6. `zmax = max(Z[:, 2])` for slider calibration.

Only picklable data crosses the process boundary; `_on_reloaded` re-opens the dF/F memmap (`load_filtered_dff`) and `stat` (`stat_for_prefix`) by path on the main thread.

`_on_reloaded(data)`:

- Sets `_Z`, `_stat`, `_dff`, `_Lx`, `_Ly`, `_zmax` from the payload.
- Snaps `_manual_T = saved_threshold` (clamped to `[0, zmax]` to handle stale exports).
- Forces `manual_var = True` so the slider tracks the saved cut, not the auto threshold.
- Calibrates the slider, enables Export *_rois.npy / Export figures / Save summary / Recluster / cluster menu, seeds the recluster threshold to `_manual_T / 2`.
- Calls `_render_all()` (which redraws dendrogram + spatial map and refreshes the picked-cluster ROI list).
- Status: `Reloaded from <cluster_dir>. N_rois=N  T=T  saved cut=...  (n_clusters clusters).`

### Step 6 — Export

`_export_clusters()` writes, per cluster:

```
<plane0>/r0p7_filtered_cluster_results/
├── C1_rois.npy           ← Suite2p ROI indices (NOT filtered-list positions)
├── C2_rois.npy
├── ...
├── linkage.npy           ← the full Z matrix
├── linkage_roi_ids.npy   ← ordered Suite2p ids of the linkage leaves
├── threshold_used.npy    ← the threshold at which the cut was made
└── _indices_are_suite2p  ← marker file: "1"
```

Cluster numbering matches the dendrogram's left-to-right visual order (C1, C2, … = leftmost branch, second branch, …). The `_indices_are_suite2p` marker tells Tab 7 that the indices in `C*_rois.npy` are positions in `stat.npy` / `F.npy` (Suite2p indices), not positions in the kept-only memmap.

The export translates leaf positions to Suite2p ids with the ordered ids recorded when the linkage was computed (or reloaded). Export, the picked-cluster ROI list and the summary sheet first check that the current kept ids (`core.clustering.linkage_roi_ids`, which uses the file-size-anchored `filtered_to_suite2p_indices` — see Step 4's keep-mask note) are exactly those ids in the same order; if not, Export refuses with "Re-run clustering", the ROI list stays empty and the summary write is skipped. When Tab 3 regenerates the filtered dF/F after a promotion (`AppState.release_memmaps`), Tab 6 drops its in-memory linkage and disables Export / Recluster / Save summary until you re-run.

Stale `C*_rois.npy` files are deleted before writing, so re-export with fewer clusters doesn't leave ghost files behind.

---

## 3. Outputs (on disk)

`<plane0>/r0p7_filtered_cluster_results/` as listed above.

Plus the summary workbook's `Clusters` sheet via `summary_writer.write_clusters_sheet`, with `roi (Suite2p), cluster_id, cluster_color, threshold_used, method, metric` columns.

---

## 4. UI knobs (actions)

Buttons that *do* something but carry no tunable value:

- **Run analysis** — fresh linkage compute from the dF/F memmap.
- **Reload clusters** — restore a previously exported run from `linkage.npy` + `threshold_used.npy`. Skips the linkage compute; lands in manual-threshold mode at the saved cut.
- **Per-cluster colors…** — custom hex per cluster (overrides the palette dropdown until reset).
- **Reset palette** — drop custom colours, fall back to the dropdown palette.
- **Recluster (new window)** — drill into the ticked branch(es) at a finer cut in a separate window (see the `recluster threshold` setting below).
- **ROIs in picked clusters (Suite2p ids)** — read-only entry that auto-fills with the Suite2p ROI ids of every cluster ticked in the picker, in compact range form. **Copy** button puts the string on the system clipboard (paste into Tab 4 / Tab 5 manual entries or external scripts).
- **Export *_rois.npy** — write per-cluster ROI lists to disk.
- **Export figures** — save the current dendrogram + spatial map.
- **Save summary** — write/refresh the Clusters sheet.

---

## Advanced settings (inline knobs)

Tab 6 has **no PARAM_SPEC / Advanced dialog** — every tunable is an inline Tk-var on the tab itself, defined in `tabs/clustering/tab.py`. Defaults are the literal `tk.*Var(value=…)` defaults. The headless equivalents (the `params` dict consumed by [`core/clustering.py`](../../core/clustering.py) `run_clustering`) are noted per row so a script run reproduces the GUI. Tab 0's batch runner sets `prefix` / `palette` / `threshold` through `ClusteringTab.apply_batch_row`: a `threshold` > 0 is an absolute Ward cut that the next Run applies in manual mode (clamped to `[0, zmax]`) and the batch export uses; `0` means auto. The batch pushes `threshold` only at the clustering stage, and an interactive Run (no batch active) discards any cut a batch left armed, so a skipped or aborted batch stage can't force its cut on your next Run.

| Setting (`var`) | GUI label | Default | What it does | What it means to you |
| --- | --- | --- | --- | --- |
| `prefix_var` ([tab.py:425](tab.py)) | `prefix:` | `r0p7_filtered_` | Names the dF/F memmap to cluster: `<plane0>/<prefix>dff.memmap.float32`. Drives `load_filtered_dff` (loads the `(T, N_kept)` columns), `stat_for_prefix` (matching ROI stats), and `filtered_to_suite2p_indices` (export id translation). When the prefix contains `filtered`, the keep mask is resolved via `utils.resolve_filtered_mask` **anchored to the memmap file size** so leaf count, `stat`, and the export translator stay the same length. Headless: `params['prefix']`. | Leave it at `r0p7_filtered_` for the normal pipeline — that's the curated, cell-filtered, baseline-corrected dF/F. Switch to `r0p7_` (unfiltered) only to cluster *every* detected source including rejected ROIs; the export then writes filtered-list positions, not Suite2p ids, and Tab 7 alignment is your problem. A prefix with no matching memmap disables **Run analysis** (status bar says "Missing dF/F memmaps"). |
| `manual_var` ([tab.py:406](tab.py)) | `Manual threshold` (checkbox) | `False` (auto) | Switches `_current_threshold()` between **auto** (`auto_frac × zmax`, from `auto_choose_threshold` targeting 4–5 clusters) and **manual** (the slider's absolute value). Turning it ON enables the slider and snaps it to the current auto value as a starting point; OFF disables the slider and reverts to auto. Headless: pass `threshold=None` (or any value <= 0) for auto, or a positive float for manual. Batch: `threshold` column, `0` = auto. | Keep it OFF (auto) for a sane first look — auto lands ~4–5 clusters, good for ~100–500 cells. Tick it ON the moment the auto cut looks wrong (one giant blob, or shattered into 30 specks) so you can drag the slider. **Reload clusters** also forces this ON at the saved cut. |
| `threshold_scale` ([tab.py:369](tab.py)) | `cut` (vertical slider) | range `zmax → 0`, resolution `zmax/1000` (~1000 steps) | The cut **height in absolute Ward distance** (sum-of-squares units, *not* a `1 − r` value). Feeds `fcluster(Z, t=T, criterion='distance')` on every settled drag; the dashed line on the dendrogram tracks it and the spatial map recolours. Renders are debounced (~120 ms) so a drag doesn't freeze the GUI. Only live when `manual_var` is ON. | This is your main shaping control. **Lower** the cut → finer split → **more, smaller** clusters; **raise** it → coarser → **fewer, larger** clusters. Drag until the spatial map shows ensembles that match your anatomy/physiology, then Export. The absolute scale is recording-specific (it depends on `zmax`), so a value that works on one recording won't transfer to another — re-judge by eye each time. |
| `palette_var` ([tab.py:412](tab.py)) | `palette:` (dropdown) | `tab10` | Picks the colour scheme from `AVAILABLE_PALETTES` (categorical `tab10/tab20/Set1/Set2/Set3/Paired/Accent/Pastel1/Pastel2/Dark2`; continuous `viridis/plasma/inferno/magma/cividis/coolwarm/RdBu/RdYlBu/Spectral/turbo/rainbow/hsv`). `resolve_palette(name, n_clusters)` returns one hex per cluster, spread across the full range so C1 and Cn are maximally distinct. Picking a palette **clears** any per-cluster custom colours. Headless: `params['palette']`. | Cosmetic — it has **zero** effect on which ROIs cluster together, only on the colours in the dendrogram, spatial map, and the `cluster_color` column of the Clusters sheet. Use a **categorical** palette (`tab10`) when clusters are nominal (the default, almost always right); use a **continuous** one (`viridis`) only if you want C1→Cn to read as an ordered ramp. If you have >10 clusters with a 10-colour categorical palette, colours cycle/repeat — switch to `tab20` or a continuous map for distinct colours. |
| `recluster_thr_var` ([tab.py:307](tab.py)) | `new threshold (Ward dist.):` (entry) | `""` (seeded to `manual_T / 2` after a run/reload) | The cut for the **Recluster (new window)** sub-tree. Parsed as a plain float and used as an **absolute Ward-distance** cut: the ticked branches' ROIs are sliced out, `ward_linkage` re-runs on just that subset, and the new window cuts its own dendrogram at this value. The parent clustering is untouched. | Use this to sub-divide one fat ensemble without disturbing the rest. It's pre-filled with half the main cut as a reasonable "go finer" starting point; **lower** it for a tighter split of the branch, **raise** it toward the parent cut for a coarser one. Tick at least one cluster in **Pick clusters…** first, and the selection must contain ≥2 ROIs or the recluster is refused. Sub-windows pick a *different* palette than the parent on purpose so the two don't visually collide. |

> **Units.** Since the 2026-05-11 switch to Ward + Euclidean the cut is in **absolute Ward (sum-of-squares) distance**, and the header, dendrogram y-axis and recluster entry now say so. The z-scoring makes pair *ranking* identical to `1 − r` (§2 Step 1), but the numeric value on the slider is a Ward distance, not a correlation. Calibrate by eye, not by an expected `1 − r` number.

---

## 4b. Cluster popout (click an ROI on the spatial map)

Module: `tabs/clustering/cluster_popout.py`. Single-instance per plane0 — clicking a different ROI swaps focus on the existing window.

Stacked panels:
1. **Heatmap** of filtered lowpass dF/F restricted to that ROI's cluster (`(n_in_cluster, T)`, per-row min-max normalised, `magma` colormap). Rows are labelled with Suite2p ROI ids (every row up to 40, else ~40 evenly spaced). The title names the cluster as Tab 6 does (C1, C2, ...) and the clicked ROI by Suite2p id; the cluster's colour tints the title background.
2. **Median dF/F** across the cluster's ROIs (not normalised), with the 25-75% range shaded in the cluster colour.
3. **Event raster** for the same rows. In the GUI this is the `|d/dt| ≥ k·MAD` derivative-threshold raster computed in-popout: the live `AppState.event_results` payload published by Tab 5 does **not** carry a precomputed `'raster'` (only the headless event-detection runner emits one), so the `event_results['raster']` branch is forward-compatible scaffolding that activates only if a future payload supplies a column-matched raster (and its row-alignment to the filtered stat list is established). Whether to build a real onset raster from `event_results['onsets_by_roi']` is an open owner decision.
4. **Status row** lists the cluster's members by **Suite2p ROI id** (capped at 30 + overflow count). Ids come from the ids the linkage was built on (`_linkage_ids_current`); if the kept ROIs changed since clustering ran, labels fall back to column positions marked `pos`.

All panels share the time axis (seconds, using the resolved frame rate — Tab 3's FPS override, else the default fallback); the heatmap and raster draw a yellow row marker for the clicked ROI.

The popout reuses Tab 6's already-loaded `_dff` and `_stat`, so no extra disk reads happen on click.

---

## 5. Auto-threshold heuristic (`auto_choose_threshold`)

```python
grid = arange(start=0.90, stop=0.05 - 1e-9, step=-0.01)   # descending cut fractions of zmax
# count_clusters (fcluster) is monotone non-decreasing down the grid, so:
i   = bisect grid for the first index whose count_clusters(Z, grid[i]) >= min(target_counts)
# then a short forward scan from i lands the first count actually in target_counts
for f in grid[i:]:
    n = count_clusters(Z, f)                # == unique(fcluster(Z, t=f*zmax, 'distance')).size
    if n in target_counts:
        return f
    if n > max(target_counts):
        break
return start                                  # nothing hit the band → fall back to 0.90
```

(See `auto_choose_threshold` in `core/clustering.py` for the exact implementation.) Defaults `target_counts=(4, 5)` aim for **4–5 clusters**, which is a good visual starting point for ~100–500 cells. Override via the slider for any other count.

---

## 6. Re-implementation checklist

1. **Distance + linkage:**
   - z-score each column of dF/F (so squared Euclidean ranks pairs identically to `1 − r`; see §2 Step 1).
   - `dist = scipy.spatial.distance.pdist(X.T, metric='euclidean')`.
   - `Z = scipy.cluster.hierarchy.linkage(dist, method='ward')`.
2. **Pick a threshold** (auto or slider). `labels = scipy.cluster.hierarchy.fcluster(Z, t=T, criterion='distance')`.
3. **Stable cluster numbering:** walk dendrogram leaves left-to-right, map first-seen label → C1, etc. Don't trust scipy's label ordering for visual cluster ids.
4. **Stable colours:** build a `{raw_label: hex_color}` dict from your palette in visual-id order, then pass a `link_color_func` to `dendrogram` that paints each link with its descendant's colour. Bypass scipy's palette cycling.
5. **Spatial paint:** `paint_spatial(values, stat, Ly, Lx)` accumulates `value × lam` weighted by `lam` per pixel and divides by accumulated weights. Run separately for R, G, B channels; `NaN` outside ROI coverage.
6. **Recluster:** for selected branches, slice dF/F to those ROIs, re-run `ward_linkage`, open a sub-window with its own slider and palette.
7. **Export format:**
   - One `C{i}_rois.npy` per cluster, indices = Suite2p ROI indices.
   - `linkage.npy` (full `Z`), `linkage_roi_ids.npy` (ordered Suite2p ids of the leaves), `threshold_used.npy` (single-element float).
   - Marker file `_indices_are_suite2p` to disambiguate from the legacy filtered-position layout.
8. **AVAILABLE_PALETTES, resolve_palette, auto_choose_threshold** — all in `core/clustering.py`. Reproduce signatures `resolve_palette(name, n_colors) -> list[str]` and `auto_choose_threshold(Z, target_counts=(4,5)) -> float`.
9. **Picked-cluster ROI list:** translate the picked visual ids → raw fcluster labels via `_visual_to_label`, mask the leaf array with `np.isin`, translate leaf positions to Suite2p ROI ids with the ids the linkage was built on (empty if the current kept ids differ), sort, format with `format_roi_indices` (compact `a-b` runs), and expose a Copy button that calls `clipboard_clear/append + update()` so the text survives Tk shutdown on Windows.
10. **Reload existing clusters:** `np.load` the saved `linkage.npy` + `threshold_used.npy`, validate `Z.shape[0]+1 == N_kept` and that `linkage_roi_ids.npy` equals the current kept ids, reload dF/F + stat + ops, snap manual mode + slider to the saved threshold, render. Skips the `ward_linkage` call entirely.


## UI affordances

Tab 6 inherits the global customtkinter dark theme from `pipeline_gui`.

- **Horizontal sash** splits `(Dendrogram + cut slider) | Spatial map`. Drag the sash left or right to redistribute width between the two canvases; the cut slider stays glued to the right edge of the dendrogram pane.
- **Popouts.**
  - **Per-cluster colors** (`CustomColorDialog`) — opens from "Per-cluster colors..." in the bottom controls; non-resizable color-picker dialog with one swatch button per cluster.
  - **Recluster (new window)** (`ReclusterWindow`) — opens from the recluster row; resizable Toplevel hosting a sub-dendrogram + sub-spatial map plus its own slider. Read-only.
  - **Cluster heatmap + raster** (`ClusterPopout`) — opens on click of an ROI in the spatial map; resizable Toplevel showing the heatmap, the cluster's median dF/F and the raster for that ROI's cluster, labelled by Suite2p ROI id.
- All matplotlib toolbars route through `gui_common.attach_fig_toolbar` and pick up the dark restyle automatically.
