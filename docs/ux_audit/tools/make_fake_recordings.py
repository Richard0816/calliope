"""Synthetic two-photon calcium recordings for driving CalLIOPE.

Ground truth is saved next to each TIFF (``*_truth.npz``) so outputs can be
checked against known cells / events / propagation direction.
"""
import sys
from pathlib import Path

import numpy as np
import tifffile


def calcium_kernel(fs, tau_rise=0.05, tau_decay=0.4, dur=3.0):
    t = np.arange(int(dur * fs)) / fs
    k = (1 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)
    return k / k.max()


def make_recording(out, *, fs=15.08, T=2700, H=256, W=256, n_cells=70,
                   n_bursts=12, wave=True, seed=0, offset=-180,
                   bleach=0.12, jitter=1.5, noise_only=False, split=1,
                   ij_fps=None):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]

    # Neuropil: smooth blotchy background.
    from scipy.ndimage import gaussian_filter, shift as ndshift
    neuropil = gaussian_filter(rng.random((H, W)), 12)
    neuropil = 250 + 500 * (neuropil - neuropil.min()) / np.ptp(neuropil)

    # Somata with min spacing.
    cells = []
    while len(cells) < n_cells and not noise_only:
        c = rng.uniform(12, [H - 12, W - 12])
        if all(np.hypot(*(c - d)) > 16 for d in cells):
            cells.append(c)
    cells = np.array(cells) if cells else np.zeros((0, 2))
    radii = rng.uniform(4.5, 7.0, len(cells))
    base_bright = rng.uniform(300, 700, len(cells))
    footprints = []
    for (cy, cx), r in zip(cells, radii):
        d = np.hypot(yy - cy, xx - cx)
        fp = np.clip(1.2 - d / r, 0, 1) ** 0.7  # ring-ish soft disk
        fp[d < r * 0.35] *= 0.7  # darker nucleus, as in real GCaMP somata
        footprints.append(fp.astype(np.float32))

    # Spike trains.
    kern = calcium_kernel(fs)
    spikes = np.zeros((len(cells), T), np.float32)
    spikes += (rng.random((len(cells), T)) < 0.15 / fs) * rng.uniform(
        0.3, 0.8, (len(cells), T))  # ~0.15 Hz spontaneous
    burst_frames = np.sort(rng.choice(np.arange(int(8 * fs), T - int(5 * fs)),
                                      n_bursts, replace=False))
    # Enforce >= 6 s between bursts.
    burst_frames = burst_frames[np.r_[True, np.diff(burst_frames) > 6 * fs]]
    participants = []
    for bf in burst_frames:
        part = rng.random(len(cells)) < 0.8
        participants.append(part)
        for i in np.where(part)[0]:
            if wave:
                # wave from top-left to bottom-right, ~400 ms across FOV
                delay = (cells[i, 0] + cells[i, 1]) / (H + W) * 0.4
            else:
                delay = rng.uniform(0, 0.1)
            f = int(bf + delay * fs + rng.normal(0, 0.5))
            if 0 <= f < T:
                spikes[i, f] += rng.uniform(1.5, 2.5)
    dff = np.array([np.convolve(s, kern)[:T] for s in spikes], np.float32)

    # Assemble movie frame by frame.
    bleach_curve = 1 - bleach * (1 - np.exp(-np.arange(T) / (T / 2)))
    movie = np.empty((T, H, W), np.int16)
    fp_stack = np.stack(footprints) if footprints else np.zeros((0, H, W),
                                                                np.float32)
    shifts = np.cumsum(rng.normal(0, 0.15, (T, 2)), axis=0)
    shifts = np.clip(shifts, -jitter, jitter)
    for t in range(T):
        frame = neuropil * (1 + 0.15 * dff[:, t].mean() if len(cells) else 1)
        if len(cells):
            amp = base_bright * (1 + dff[:, t])
            frame = frame + np.tensordot(amp, fp_stack, axes=1)
        frame = frame * bleach_curve[t]
        if jitter:
            frame = ndshift(frame, shifts[t], order=1, mode="nearest")
        frame = rng.poisson(np.clip(frame, 0, None) / 4.0) * 4.0
        frame = frame + offset + rng.normal(0, 6, frame.shape)
        movie[t] = np.clip(frame, -32768, 32767).astype(np.int16)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = {"axes": "TYX"}
    if ij_fps:
        md["finterval"] = 1.0 / ij_fps
    parts = np.array_split(np.arange(T), split)
    paths = []
    for k, idx in enumerate(parts):
        p = out if split == 1 else out.with_name(
            f"{out.stem}_part{k + 1:03d}{out.suffix}")
        tifffile.imwrite(p, movie[idx], imagej=True, metadata=md)
        paths.append(p)
    np.savez(out.with_name(out.stem + "_truth.npz"), cells=cells, radii=radii,
             burst_frames=burst_frames, fs=fs, participants=np.array(
                 participants), dff=dff)
    print(f"{out.stem}: {len(cells)} cells, {len(burst_frames)} bursts at "
          f"{np.round(burst_frames / fs, 1).tolist()} s; files={[p.name for p in paths]}"
          f" range=[{movie.min()}, {movie.max()}]")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    make_recording(root / "raw" / "rec_wave.tif", ij_fps=15.08)
    make_recording(root / "raw" / "rec_split.tif", fs=30.0, T=3600,
                   n_cells=45, n_bursts=8, seed=1, split=2, ij_fps=30.0)
    make_recording(root / "raw_noise" / "rec_noise.tif", T=1200,
                   noise_only=True, seed=2, ij_fps=15.08)
