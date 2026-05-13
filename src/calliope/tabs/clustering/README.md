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
> `params['threshold']` is `None` or 0), cuts to flat clusters, exports
> `C{i}_rois.npy` (Suite2p ids) + `linkage.npy` + `threshold_used.npy` +
> `_indices_are_suite2p`, and (optionally) renders `dendrogram.png` +
> `spatial_clusters.png`. Tab 6 still owns the interactive recluster /
> palette / picker UI; the headless module is what Tab 0's batch worker
> calls and what you'd reach for in a script.

### Step 1 — Distance matrix

`_ward_linkage(dff, method='ward')`:

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

(Switched from average + correlation → Ward + Euclidean on 2026-05-11 after user pointed out that z-scoring already kills amplitude differences. Backward-compat alias `_correlation_linkage` still exists to avoid breaking importers.)

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

A parallel `(Ly, Lx)` int label image (`_build_label_image`) encoding `filtered_idx + 1` per painted pixel is also rebuilt every render so the spatial-click handler can resolve a cursor position back to an ROI without a per-pixel scan over `stat`. Click any coloured ROI to open the cluster popout (see §"Cluster popout").

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
    keep_mask  = _load_filter_mask(plane0)
    translator = where(keep_mask)[0]            # filtered-pos -> Suite2p id
    return sorted(translator[local_idx].tolist())
return sorted(local_idx.tolist())
```

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
Z_sub         = _correlation_linkage(dff_subset)
```

A separate `ReclusterWindow` opens with its own dendrogram + slider + spatial map. The parent state is untouched. Recluster windows pick a different palette (cycling through `CATEGORICAL_PALETTES`) so their colours visibly differ from the parent.

### Step 5b — Reload existing clusters

A "Reload clusters" button next to "Run analysis" restores a previous Export run without recomputing the linkage (which is the expensive part — pdist of `1−r` is O(N²·T) and can be 30 s to a few minutes per recording).

The button is enabled only when both files exist for the active prefix:

```
<plane0>/{prefix}cluster_results/gui_recluster/linkage.npy
<plane0>/{prefix}cluster_results/gui_recluster/threshold_used.npy
```

`_reload_compute(plane0, prefix)`:

1. `Z = np.load(linkage.npy)` — the saved condensed linkage matrix.
2. `thr = float(np.load(threshold_used.npy)[0])` — the cut at which the user clicked Export.
3. `dff_mm, T, N = _load_filtered_dff(...)` — fresh memmap for spatial recolouring + reclustering.
4. **Validation:** `Z.shape[0] + 1 == N`. If the cell-filter mask has changed since export, leaf count and ROI count won't match — the reload aborts with a clear error rather than silently misaligning ROIs to dendrogram leaves.
5. `stat`, `ops`, `Lx`, `Ly` loaded as in `_compute`.
6. `zmax = max(Z[:, 2])` for slider calibration.

`_on_reloaded(data)`:

- Sets `_Z`, `_stat`, `_dff`, `_Lx`, `_Ly`, `_zmax` from the payload.
- Snaps `_manual_T = saved_threshold` (clamped to `[0, zmax]` to handle stale exports).
- Forces `manual_var = True` so the slider tracks the saved cut, not the auto threshold.
- Calibrates the slider, enables Export / Save summary / Recluster / cluster menu, seeds the recluster threshold to `_manual_T / 2`.
- Calls `_render_all()` (which redraws dendrogram + spatial map and refreshes the picked-cluster ROI list).
- Status: `Reloaded from <cluster_dir>. N_rois=N  T=T  saved cut=...  (n_clusters clusters).`

### Step 6 — Export

`_export_clusters()` writes, per cluster:

```
<plane0>/r0p7_filtered_cluster_results/gui_recluster/
├── C1_rois.npy           ← Suite2p ROI indices (NOT filtered-list positions)
├── C2_rois.npy
├── ...
├── linkage.npy           ← the full Z matrix
├── threshold_used.npy    ← the threshold at which the cut was made
└── _indices_are_suite2p  ← marker file: "1"
```

Cluster numbering matches the dendrogram's left-to-right visual order (C1, C2, … = leftmost branch, second branch, …). The `_indices_are_suite2p` marker tells Tab 7 that the indices in `C*_rois.npy` are positions in `stat.npy` / `F.npy` (Suite2p indices), not positions in the kept-only memmap.

For the filtered prefix, the export translates from filtered-position to Suite2p index using `np.where(keep_mask)[0]`. For an unfiltered prefix it's a no-op.

Stale `C*_rois.npy` files are deleted before writing, so re-export with fewer clusters doesn't leave ghost files behind.

---

## 3. Outputs (on disk)

`<plane0>/r0p7_filtered_cluster_results/gui_recluster/` as listed above.

Plus the summary workbook's `Clusters` sheet via `summary_writer.write_clusters_sheet`, with `cluster_id, roi (Suite2p), color_hex, threshold, method, metric` columns.

---

## 4. UI knobs

- **Run analysis** — fresh linkage compute from the dF/F memmap.
- **Reload clusters** — restore a previously exported run from `linkage.npy` + `threshold_used.npy`. Skips the linkage compute; lands in manual-threshold mode at the saved cut.
- **Manual threshold checkbox** — toggle to enable the slider; OFF returns to the auto threshold.
- **Vertical cut slider** — `from_=zmax, to=0`, resolution = `zmax/1000`. Drag to reshape clusters.
- **Palette dropdown** — choose any palette in `AVAILABLE_PALETTES`.
- **Per-cluster colors…** — custom hex per cluster (overrides palette).
- **Reset palette** — drop custom colours.
- **Recluster branch(es)** — drill into a subset of the current clustering at a finer threshold in a separate window.
- **ROIs in picked clusters (Suite2p ids)** — read-only entry that auto-fills with the Suite2p ROI ids of every cluster ticked in the picker, in compact range form. **Copy** button puts the string on the system clipboard (paste into Tab 4 / Tab 5 manual entries or external scripts).
- **prefix entry** — change which dF/F memmap to cluster on (default `r0p7_filtered_`).
- **Export *_rois.npy** — write to disk.
- **Save summary** — write/refresh the Clusters sheet.

---

## 4b. Cluster popout (click an ROI on the spatial map)

Module: `tabs/clustering/cluster_popout.py`. Single-instance per plane0 — clicking a different ROI swaps focus on the existing window.

Stacked panels:
1. **Heatmap** of filtered lowpass dF/F restricted to that ROI's cluster (`(n_in_cluster, T)`, per-row min-max normalised, `magma` colormap). The cluster's colour from the spatial map tints the title background so the popout reads at a glance.
2. **Event raster** for the same rows. Prefers `AppState.event_results['raster']` (Tab 5's per-ROI hysteresis onsets) when its column count matches the dF/F memmap so the popout shows the user's *real* event detections; otherwise falls back to a simple `|d/dt| ≥ k·MAD` derivative-threshold raster computed in-popout.

Both panels share the time axis (seconds when `notes.txt` provides fps, frame indices otherwise) and draw a yellow row marker for the clicked ROI's position within the cluster. The status bar lists the cluster's filtered-position roster (capped at 30 + overflow count).

The popout reuses Tab 6's already-loaded `_dff` and `_stat`, so no extra disk reads happen on click.

---

## 5. Auto-threshold heuristic (`auto_choose_threshold`)

```python
fractions   = linspace(0.05, 0.95, 91)         # candidate cuts as fraction of zmax
best_frac   = fractions[0]
for f in fractions:
    n = unique(fcluster(Z, t=f*zmax, criterion='distance')).size
    if target_counts[0] <= n <= target_counts[1]:
        best_frac = f
        break       # pick the smallest fraction (highest cut) that hits the band
return best_frac
```

(See `core/clustering.py` for the exact implementation.) Defaults aim for **4–5 clusters**, which is a good visual starting point for ~100–500 cells. Override via the slider for any other count.

---

## 6. Re-implementation checklist

1. **Distance + linkage:**
   - z-score each column of dF/F (defensive; `correlation` metric is mean/scale-invariant anyway).
   - `dist = scipy.spatial.distance.pdist(X.T, metric='euclidean')`.
   - `Z = scipy.cluster.hierarchy.linkage(dist, method='ward')`.
2. **Pick a threshold** (auto or slider). `labels = scipy.cluster.hierarchy.fcluster(Z, t=T, criterion='distance')`.
3. **Stable cluster numbering:** walk dendrogram leaves left-to-right, map first-seen label → C1, etc. Don't trust scipy's label ordering for visual cluster ids.
4. **Stable colours:** build a `{raw_label: hex_color}` dict from your palette in visual-id order, then pass a `link_color_func` to `dendrogram` that paints each link with its descendant's colour. Bypass scipy's palette cycling.
5. **Spatial paint:** `paint_spatial(values, stat, Ly, Lx)` accumulates `value × lam` weighted by `lam` per pixel and divides by accumulated weights. Run separately for R, G, B channels; `NaN` outside ROI coverage.
6. **Recluster:** for selected branches, slice dF/F to those ROIs, re-run `_correlation_linkage`, open a sub-window with its own slider and palette.
7. **Export format:**
   - One `C{i}_rois.npy` per cluster, indices = Suite2p ROI indices.
   - `linkage.npy` (full `Z`), `threshold_used.npy` (single-element float).
   - Marker file `_indices_are_suite2p` to disambiguate from the legacy filtered-position layout.
8. **AVAILABLE_PALETTES, resolve_palette, auto_choose_threshold** — all in `core/clustering.py`. Reproduce signatures `resolve_palette(name, n_colors) -> list[str]` and `auto_choose_threshold(Z, target_counts=(4,5)) -> float`.
9. **Picked-cluster ROI list:** translate the picked visual ids → raw fcluster labels via `_visual_to_label`, mask the leaf array with `np.isin`, translate filtered-list positions to Suite2p ROI ids using the keep mask if the prefix is filtered, sort, format with `format_roi_indices` (compact `a-b` runs), and expose a Copy button that calls `clipboard_clear/append + update()` so the text survives Tk shutdown on Windows.
10. **Reload existing clusters:** `np.load` the saved `linkage.npy` + `threshold_used.npy`, validate `Z.shape[0]+1 == N_kept`, reload dF/F + stat + ops, snap manual mode + slider to the saved threshold, render. Skips the `_correlation_linkage` call entirely.
