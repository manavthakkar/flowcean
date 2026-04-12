#!/usr/bin/env python3
"""
Delocalization onset analysis — all labeling strategies compared.

The key idea
------------
Three labeling strategies are evaluated side-by-side:

  1. Threshold-based  (lbl_err_*)
     label = True when position_error > T (a fixed error threshold).
     Simple and deterministic; requires ground-truth pose at inference.

  2. Predictive-window  (lbl_win_*)
     label = True when an AMCL reset (/initialpose) occurs within the
     next W seconds.  Trains the model to detect delocalization early
     from sensor features — ground truth is only needed at training time.

  3. Original (is_delocalized)
     label = True when position_error > 0.8 m (recording reset threshold).

Ground truth comes from /initialpose messages (the AMCL resets recorded
in the bag) and /momo/pose (GPS/ground-truth pose).

Panels
------
  [0,0] Error time series with threshold reference lines and reset markers
  [0,1] Pre-reset error profiles (how early degradation becomes visible)
  [1,0] Threshold-based label analysis
  [1,1] Predictive-window label analysis
  [2,0] Positive-rate & error-separation summary across ALL strategies
  [2,1] Commitment curve (point-of-no-return validation)

Usage
-----
Run from the robot_localization_failure directory:

    python -m ml_pipeline.debug.find_threshold \\
        recordings/test_data/training_data_rec_20260302_082153

    # Tune windows and thresholds:
    python -m ml_pipeline.debug.find_threshold <bag_dir> \\
        --candidate-windows 5 10 15 20 \\
        --thresh-labels 0.2 0.3 0.4 0.5 0.6 0.8 \\
        --margin 5.0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader


# ─── helpers ──────────────────────────────────────────────────────────────────

def _yaw(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      qw * qw + qx * qx - qy * qy - qz * qz)


def _angle_diff(a, b):
    return ((a - b + math.pi) % (2 * math.pi)) - math.pi


# ─── bag reading ───────────────────────────────────────────────────────────────

def read_bag(bag_path: Path):
    """
    Returns:
        amcl   : list[(time_ns, x, y, yaw)]
        gt     : list[(time_ns, x, y, yaw)]
        resets : sorted list[time_ns]          — /initialpose timestamps
    """
    amcl, gt, resets = [], [], []

    with AnyReader([bag_path]) as reader:
        topics = {"/amcl_pose", "/momo/pose", "/initialpose"}
        connections = [c for c in reader.connections if c.topic in topics]
        total = sum(c.msgcount for c in connections)
        done = 0

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, conn.msgtype)
            done += 1
            if done % 2000 == 0:
                print(f"  reading {done}/{total} …", end="\r")

            if conn.topic == "/amcl_pose":
                p = msg.pose.pose  # type: ignore[union-attr]
                amcl.append((
                    timestamp,
                    p.position.x, p.position.y,
                    _yaw(p.orientation.x, p.orientation.y,
                         p.orientation.z, p.orientation.w),
                ))
            elif conn.topic == "/momo/pose":
                p = msg.pose
                gt.append((
                    timestamp,
                    p.position.x, p.position.y,
                    _yaw(p.orientation.x, p.orientation.y,
                         p.orientation.z, p.orientation.w),
                ))
            elif conn.topic == "/initialpose":
                resets.append(timestamp)

    print(f"  reading done ({total} messages)          ")
    return amcl, gt, sorted(resets)


# ─── error computation ─────────────────────────────────────────────────────────

def compute_errors(amcl, gt):
    gt_t   = np.array([p[0] for p in gt], dtype=np.int64)
    gt_x   = np.array([p[1] for p in gt])
    gt_y   = np.array([p[2] for p in gt])
    gt_yaw = np.array([p[3] for p in gt])

    times, pos_err, head_err = [], [], []
    for t, ax, ay, ayaw in amcl:
        idx = int(np.searchsorted(gt_t, t))
        idx = min(max(idx, 0), len(gt_t) - 1)
        if idx > 0 and abs(gt_t[idx - 1] - t) < abs(gt_t[idx] - t):
            idx -= 1
        pe = math.sqrt((ax - gt_x[idx]) ** 2 + (ay - gt_y[idx]) ** 2)
        he = abs(_angle_diff(ayaw, gt_yaw[idx]))
        times.append(t)
        pos_err.append(pe)
        head_err.append(he)

    return (np.array(times, dtype=np.int64),
            np.array(pos_err),
            np.array(head_err))


# ─── noise floor estimation ────────────────────────────────────────────────────

def estimate_noise_floor(times_ns, pos_err, reset_times_ns,
                         window_s: float = 5.0, margin_s: float = 2.0):
    margin_ns = int(margin_s * 1e9)
    window_ns = int(window_s * 1e9)
    baseline_errors = []

    for r in reset_times_ns:
        mask = (times_ns >= r + margin_ns) & (times_ns < r + margin_ns + window_ns)
        baseline_errors.extend(pos_err[mask].tolist())

    if not baseline_errors:
        return 0.05, 0.1
    arr = np.array(baseline_errors)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


# ─── pre-reset error profiles ─────────────────────────────────────────────────

def pre_reset_error_profiles(times_ns, pos_err, reset_times_ns,
                             offsets_s: list[float]):
    profiles: dict[float, list[float]] = {o: [] for o in offsets_s}

    for r in reset_times_ns:
        for offset in offsets_s:
            target = r - int(offset * 1e9)
            if target < times_ns[0]:
                continue
            idx = int(np.searchsorted(times_ns, target))
            idx = min(max(idx, 0), len(times_ns) - 1)
            if abs(times_ns[idx] - target) < int(3e9):
                profiles[offset].append(float(pos_err[idx]))

    return profiles


# ─── threshold-based labeling ─────────────────────────────────────────────────

def threshold_labels(pos_err: np.ndarray,
                     candidates: list[float],
                     ) -> dict[float, np.ndarray]:
    """
    Returns a dict mapping each threshold T to a boolean array where
    label[i] = True when pos_err[i] > T.
    """
    return {T: pos_err > T for T in candidates}


# ─── predictive-window labeling ────────────────────────────────────────────────

def predictive_labels(times_ns, reset_times_ns,
                      window_s: float, margin_s: float = 2.0):
    resets = np.array(sorted(reset_times_ns), dtype=np.int64)
    margin_ns = int(margin_s * 1e9)
    window_ns = int(window_s * 1e9)

    # Vectorised: any reset r in (t, t+W]?
    lo = np.searchsorted(resets, times_ns, side="right")
    hi = np.searchsorted(resets, times_ns + window_ns, side="right")
    labels = lo < hi

    # Zero out post-reset margin
    for r in resets:
        mask = (times_ns >= r) & (times_ns < r + margin_ns)
        labels[mask] = False

    return labels


# ─── commitment curve ──────────────────────────────────────────────────────────

def build_episodes(times_ns, pos_err, reset_times_ns, margin_s: float = 2.0):
    margin_ns = int(margin_s * 1e9)
    t0 = times_ns[0]
    starts = [t0] + list(reset_times_ns)
    ends   = list(reset_times_ns) + [times_ns[-1] + 1]
    episodes = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        mask = (times_ns >= s + margin_ns) & (times_ns < e)
        if mask.sum() < 5:
            continue
        episodes.append({
            "t_s":            (times_ns[mask] - t0) / 1e9,
            "pos":            pos_err[mask],
            "ended_by_reset": i < len(reset_times_ns),
        })
    return episodes


def commitment_curve(episodes, thresholds):
    committed_frac = np.full(len(thresholds), np.nan)
    n_crossings    = np.zeros(len(thresholds), dtype=int)
    for i, T in enumerate(thresholds):
        committed, crossed = 0, 0
        for ep in episodes:
            if not ep["ended_by_reset"]:
                continue
            above = ep["pos"] >= T
            if not above.any():
                continue
            first = int(np.argmax(above))
            recovered = (ep["pos"][first:] < T * 0.5).any()
            crossed += 1
            if not recovered:
                committed += 1
        n_crossings[i] = crossed
        if crossed > 0:
            committed_frac[i] = committed / crossed
    return committed_frac, n_crossings


# ─── plotting ──────────────────────────────────────────────────────────────────

def make_plots(
    times_ns, pos_err, reset_times_ns,
    noise_p50, noise_p95,
    offsets_s, profiles,
    thresh_candidates, thresh_label_map,
    candidate_windows, window_labels, window_errors,
    thresholds, committed_frac,
    onset_thr, ponr_thr,
    out_path: Path,
):
    t0    = times_ns[0]
    t_s   = (times_ns - t0) / 1e9
    rst_s = [(r - t0) / 1e9 for r in reset_times_ns]

    tab10  = plt.get_cmap("tab10")
    oranges = plt.get_cmap("Oranges")

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle("Delocalization labeling strategy analysis", fontsize=13, y=1.01)

    # ── [0,0] Error time series ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_s, pos_err, lw=0.7, color="steelblue", label="position error [m]")
    for r in rst_s:
        ax.axvline(r, color="red", lw=0.9, alpha=0.5, linestyle="--")
    ax.axhline(noise_p95, color="green", lw=1.2, linestyle=":",
               label=f"noise floor p95: {noise_p95:.3f} m")
    # Threshold reference lines
    n_t = len(thresh_candidates)
    for i, T in enumerate(thresh_candidates):
        c = oranges(0.35 + 0.65 * i / max(n_t - 1, 1))
        frac = (pos_err > T).mean() * 100
        ax.axhline(T, color=c, lw=0.9, linestyle=":", alpha=0.8,
                   label=f"T={T:.2f} m ({frac:.0f}% True)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title(f"Error time series — {len(rst_s)} resets (red = /initialpose)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)

    # ── [0,1] Pre-reset error profiles ──────────────────────────────────────
    ax = axes[0, 1]
    sorted_offsets = sorted(offsets_s, reverse=True)
    data = [profiles[o] for o in sorted_offsets]
    labels_x = [f"T−{int(o)}s" for o in sorted_offsets] + ["T=reset"]
    data_at_reset = profiles.get(0, [])
    ax.boxplot(data + [data_at_reset],
               tick_labels=labels_x,
               patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.5),
               medianprops=dict(color="navy", lw=2))
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(noise_p95, color="green", lw=1.2, linestyle=":",
               label=f"noise floor p95: {noise_p95:.3f} m")
    if not math.isnan(onset_thr):
        ax.axhline(onset_thr, color="orange", lw=1.2, linestyle="--",
                   label=f"onset threshold: {onset_thr:.2f} m")
    ax.set_ylabel("position error [m]")
    ax.set_title("Pre-reset error profiles\n"
                 "(distribution at different time offsets before each reset)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2, axis="y")

    # ── [1,0] Threshold-based label distributions ───────────────────────────
    ax = axes[1, 0]
    x_bins = np.linspace(0, max(pos_err.max(), 1.5), 80)
    n_t = len(thresh_candidates)
    for i, T in enumerate(thresh_candidates):
        lbl = thresh_label_map[T]
        err_true = pos_err[lbl]
        frac_true = lbl.mean() * 100
        if len(err_true) == 0:
            continue
        c = oranges(0.35 + 0.65 * i / max(n_t - 1, 1))
        counts, _ = np.histogram(err_true, bins=x_bins)
        counts = counts / counts.sum() if counts.sum() > 0 else counts
        ax.plot(x_bins[:-1], counts, lw=1.5, color=c,
                label=f"T={T:.2f} m  ({frac_true:.0f}% True)")
    ax.axvline(noise_p95, color="green", lw=1.2, linestyle=":",
               label=f"noise floor p95: {noise_p95:.3f} m")
    if not math.isnan(onset_thr):
        ax.axvline(onset_thr, color="orange", lw=1.2, linestyle="--",
                   label=f"onset: {onset_thr:.2f} m")
    ax.set_xlabel("position error [m]  (at True-labeled samples)")
    ax.set_ylabel("fraction of True samples")
    ax.set_title("Threshold-based labels: error dist. of True samples\n"
                 "label = True when pos_error > T  (instant, reactive)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── [1,1] Predictive-window label distributions ──────────────────────────
    ax = axes[1, 1]
    for i, W in enumerate(sorted(candidate_windows)):
        lbl = window_labels[W]
        err_true = window_errors[W]["true"]
        frac_true = lbl.mean() * 100
        if len(err_true) == 0:
            continue
        counts, _ = np.histogram(err_true, bins=x_bins)
        counts = counts / counts.sum() if counts.sum() > 0 else counts
        ax.plot(x_bins[:-1], counts, lw=1.5, color=tab10(i),
                label=f"W={int(W)}s  ({frac_true:.0f}% True)")
    ax.axvline(noise_p95, color="green", lw=1.2, linestyle=":",
               label=f"noise floor p95: {noise_p95:.3f} m")
    if not math.isnan(onset_thr):
        ax.axvline(onset_thr, color="orange", lw=1.2, linestyle="--",
                   label=f"onset: {onset_thr:.2f} m")
    ax.set_xlabel("position error [m]  (at True-labeled samples)")
    ax.set_ylabel("fraction of True samples")
    ax.set_title("Predictive-window labels: error dist. of True samples\n"
                 "label = True if reset within next W s  (early, predictive)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── [2,0] Summary: positive rate + error separation ─────────────────────
    ax = axes[2, 0]
    ax2 = ax.twinx()

    bar_labels: list[str]   = []
    pos_rates:  list[float] = []
    med_true:   list[float] = []
    bar_colors: list        = []

    for i, T in enumerate(thresh_candidates):
        lbl = thresh_label_map[T]
        bar_labels.append(f"err\n>{T:.2f}")
        pos_rates.append(lbl.mean() * 100)
        med_true.append(float(np.median(pos_err[lbl])) if lbl.any() else float("nan"))
        bar_colors.append(oranges(0.35 + 0.65 * i / max(n_t - 1, 1)))

    for i, W in enumerate(sorted(candidate_windows)):
        lbl = window_labels[W]
        bar_labels.append(f"win\n{int(W)}s")
        pos_rates.append(lbl.mean() * 100)
        true_err = window_errors[W]["true"]
        med_true.append(float(np.median(true_err)) if len(true_err) > 0 else float("nan"))
        bar_colors.append(tab10(i))

    x_pos = np.arange(len(bar_labels))
    ax.bar(x_pos, pos_rates, color=bar_colors, alpha=0.65,
           edgecolor="white", width=0.6, label="positive rate [%]")
    ax2.plot(x_pos, med_true, "D--", color="black", ms=5, lw=1.2,
             label="median True error [m]")
    ax2.axhline(noise_p95, color="green", lw=1.0, linestyle=":",
                label=f"noise floor p95 {noise_p95:.3f} m")
    if not math.isnan(onset_thr):
        ax2.axhline(onset_thr, color="orange", lw=1.0, linestyle="--",
                    label=f"onset {onset_thr:.2f} m")

    # Separator between threshold and window groups
    sep = len(thresh_candidates) - 0.5
    ax.axvline(sep, color="gray", lw=1.2, linestyle=":")
    ax.text(sep / 2, ax.get_ylim()[1] * 0.95, "threshold-based",
            ha="center", va="top", fontsize=8, color="saddlebrown")
    ax.text(sep + (len(candidate_windows)) / 2, ax.get_ylim()[1] * 0.95,
            "predictive-window", ha="center", va="top", fontsize=8, color="navy")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel("positive rate [%]")
    ax2.set_ylabel("median True-sample error [m]")
    ax.set_title("All labeling strategies — positive rate & True-sample error\n"
                 "bars = % True  |  diamonds = median pos_error of True samples")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")

    # ── [2,1] Commitment curve ────────────────────────────────────────────────
    ax = axes[2, 1]
    ax.plot(thresholds, committed_frac * 100, marker="o", ms=3,
            color="steelblue", label="committed episodes [%]")
    ax.axhline(90, color="gray", lw=1.0, linestyle=":", label="90% target")
    ax.axhline(50, color="gray", lw=0.7, linestyle=":", alpha=0.5, label="50%")
    ax.axvline(noise_p95, color="green", lw=1.2, linestyle=":",
               label=f"noise floor p95 ({noise_p95:.3f} m)")
    if not math.isnan(onset_thr):
        ax.axvline(onset_thr, color="orange", lw=1.2, linestyle="--",
                   label=f"onset ({onset_thr:.2f} m)")
    if not math.isnan(ponr_thr):
        ax.axvline(ponr_thr, color="red", lw=1.2, linestyle="--",
                   label=f"no-return ({ponr_thr:.2f} m)")
    ax.set_xlabel("position error threshold [m]")
    ax.set_ylabel("committed [%]")
    ax.set_title("Commitment curve\n"
                 "once error crosses T, % of episodes that did NOT recover")
    ax.legend(fontsize=8)
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"✔  Plot saved → {out_path}")
    plt.close()


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag_path", type=Path,
                        help="Path to the rosbag2 directory")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--margin", type=float, default=5.0,
                        help="Seconds to exclude after each reset (default: 5.0)")
    parser.add_argument("--candidate-windows", type=float, nargs="+",
                        default=[5.0, 10.0, 15.0, 20.0],
                        help="Prediction horizons W to evaluate in seconds")
    parser.add_argument("--thresh-labels", type=float, nargs="+",
                        default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.80],
                        help="Error thresholds for threshold-based label evaluation")
    args = parser.parse_args()

    bag_path = args.bag_path.resolve()
    out_path = args.output or bag_path / "find_threshold.png"

    print(f"Bag: {bag_path}")
    amcl, gt, resets = read_bag(bag_path)
    print(f"  AMCL msgs   : {len(amcl)}")
    print(f"  GT msgs     : {len(gt)}")
    print(f"  Resets      : {len(resets)}")

    if not amcl or not gt:
        print("ERROR: /amcl_pose or /momo/pose missing.")
        return
    if not resets:
        print("ERROR: No /initialpose messages — cannot determine resets.")
        return

    print("Computing errors …")
    times_ns, pos_err, _ = compute_errors(amcl, gt)

    # ── noise floor ─────────────────────────────────────────────────────────
    noise_p50, noise_p95 = estimate_noise_floor(
        times_ns, pos_err, resets, margin_s=args.margin)
    print(f"  Noise floor (post-reset baseline): p50={noise_p50:.3f} m  p95={noise_p95:.3f} m")

    # ── pre-reset error profiles ─────────────────────────────────────────────
    offsets_s = [30.0, 20.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    profiles = pre_reset_error_profiles(times_ns, pos_err, resets, offsets_s)
    print("\nPre-reset position error (median over all episodes):")
    for o in sorted(offsets_s, reverse=True):
        vals = profiles[o]
        if vals:
            print(f"  T−{int(o):2d}s : median={np.median(vals):.3f} m  "
                  f"p75={np.percentile(vals, 75):.3f} m  "
                  f"n={len(vals)}")

    # ── threshold-based labels ────────────────────────────────────────────────
    thresh_candidates = sorted(args.thresh_labels)
    thresh_label_map = threshold_labels(pos_err, thresh_candidates)

    print("\nThreshold-based labeling (position error only):")
    for T in thresh_candidates:
        lbl = thresh_label_map[T]
        frac = lbl.mean() * 100
        med_t = float(np.median(pos_err[lbl]))  if lbl.any()  else float("nan")
        med_f = float(np.median(pos_err[~lbl])) if (~lbl).any() else float("nan")
        print(f"  T={T:.2f} m → {frac:.0f}% True  |  "
              f"median True={med_t:.3f} m  False={med_f:.3f} m")

    # ── predictive-window analysis ───────────────────────────────────────────
    print("\nPredictive-window labeling:")
    window_labels_map: dict[float, np.ndarray] = {}
    window_errors:     dict[float, dict]        = {}

    for W in sorted(args.candidate_windows):
        lbl = predictive_labels(times_ns, resets, W, margin_s=args.margin)
        window_labels_map[W] = lbl
        window_errors[W] = {
            "true":  pos_err[lbl],
            "false": pos_err[~lbl],
        }
        frac        = lbl.mean() * 100
        med_true    = float(np.median(pos_err[lbl]))  if lbl.any()  else float("nan")
        med_false   = float(np.median(pos_err[~lbl])) if (~lbl).any() else float("nan")
        print(f"  W={int(W):2d}s → {frac:.0f}% True  |  "
              f"median True={med_true:.3f} m  False={med_false:.3f} m")

    # ── commitment curve ─────────────────────────────────────────────────────
    episodes   = build_episodes(times_ns, pos_err, resets, args.margin)
    thresholds = np.linspace(0.02, 1.5, 75)
    committed_frac, _ = commitment_curve(episodes, thresholds)

    onset_mask = np.where(committed_frac >= 0.5)[0]
    onset_thr  = float(thresholds[onset_mask[0]]) if len(onset_mask) > 0 else float("nan")
    ponr_mask  = np.where(committed_frac >= 0.9)[0]
    ponr_thr   = float(thresholds[ponr_mask[0]])  if len(ponr_mask) > 0 else float("nan")

    # ── recommendation ───────────────────────────────────────────────────────
    best_W = None
    best_dist = float("inf")
    for W in args.candidate_windows:
        true_errs = window_errors[W]["true"]
        if len(true_errs) == 0:
            continue
        dist = abs(float(np.median(true_errs)) - onset_thr)
        if dist < best_dist:
            best_dist = dist
            best_W = W

    print()
    print("=" * 60)
    print("  DELOCALIZATION ONSET ANALYSIS RESULTS")
    print("=" * 60)
    print(f"  Noise floor (clearly localized) : ≤ {noise_p95:.3f} m")
    print(f"  Degradation onset threshold     : ≈ {onset_thr:.2f} m  (50% commitment)")
    print(f"  Point of no return              : ≈ {ponr_thr:.2f} m  (90% commitment)")
    print()
    print("  Threshold-based recommendation:")
    best_thresh = min(
        (T for T in thresh_candidates if (pos_err > T).mean() > 0.05),
        key=lambda T: abs(T - onset_thr),
        default=None,
    )
    if best_thresh is not None:
        print(f"    Closest threshold to onset : T={best_thresh:.2f} m")
    print()
    print("  Predictive-window recommendation:")
    if best_W is not None:
        print(f"    Prediction horizon W : {int(best_W)} s")
        med = float(np.median(window_errors[best_W]["true"])) if window_errors[best_W]["true"].size else float("nan")
        print(f"    (True samples median error ≈ {med:.3f} m  |  onset ≈ {onset_thr:.2f} m)")
    else:
        print("    (no candidate windows had data — try --candidate-windows)")
    print("=" * 60)

    print("\nMaking plots …")
    make_plots(
        times_ns, pos_err,
        reset_times_ns=resets,
        noise_p50=noise_p50, noise_p95=noise_p95,
        offsets_s=offsets_s, profiles=profiles,
        thresh_candidates=thresh_candidates,
        thresh_label_map=thresh_label_map,
        candidate_windows=sorted(args.candidate_windows),
        window_labels=window_labels_map,
        window_errors=window_errors,
        thresholds=thresholds, committed_frac=committed_frac,
        onset_thr=onset_thr, ponr_thr=ponr_thr,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
