from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl

from flowcean.ros import load_rosbag
from rosbags.highlevel import AnyReader

from custom_transforms.particle_cloud_statistics import ParticleCloudStatistics
from custom_transforms.scan_map_statistics import ScanMapStatistics

from .helpers import (
    get_topics,
    timeseries_to_df,
    yaw_from_quat
)


def process_single_bag(
    bag_path: str,
    topics: dict,
    message_paths: list,
    position_threshold: float,
    heading_threshold: float,
) -> pl.DataFrame:

    print(f"\n=== Processing bag: {bag_path} ===")

    raw_lf = load_rosbag(bag_path, topics, message_paths=message_paths, cache=False)

    # -----------------------------
    # Extract occupancy map
    # -----------------------------
    map_df = raw_lf.select("/map").collect()
    map_ts = map_df["/map"][0]
    map_value = map_ts[0]["value"]

    occupancy_map = {
        "data": map_value["data"],
        "info.width": map_value["info.width"],
        "info.height": map_value["info.height"],
        "info.resolution": map_value["info.resolution"],
        "info.origin.position.x": map_value["info.origin.position.x"],
        "info.origin.position.y": map_value["info.origin.position.y"],
        "info.origin.position.z": map_value["info.origin.position.z"],
        "info.origin.orientation.x": map_value["info.origin.orientation.x"],
        "info.origin.orientation.y": map_value["info.origin.orientation.y"],
        "info.origin.orientation.z": map_value["info.origin.orientation.z"],
        "info.origin.orientation.w": map_value["info.origin.orientation.w"],
    }

    # -----------------------------
    # ScanMapStatistics
    # -----------------------------
    sms = ScanMapStatistics(
        occupancy_map=occupancy_map,
        scan_topic="/scan",
        sensor_pose_topic="/amcl_pose",
    )
    with_sms = sms.apply(raw_lf)

    # -----------------------------
    # ParticleCloudStatistics
    # -----------------------------
    pcs = ParticleCloudStatistics(particle_cloud_feature_name="/particle_cloud")
    full_df = pcs.apply(with_sms).collect()

    # Drop large raw/intermediate columns immediately to free RAM.
    # /map (full occupancy grid), /particle_cloud (5000 particles × N msgs),
    # and scan_points* (all scan point arrays) are no longer needed.
    _drop_after_transforms = [
        "/map", "/particle_cloud", "scan_points", "scan_points_sensor"
    ]
    full_df = full_df.drop(
        [c for c in _drop_after_transforms if c in full_df.columns]
    )

    # -----------------------------
    # Build Base Index from scan time
    # -----------------------------
    scan_ts = full_df["/scan"][0]
    base = pl.DataFrame({"time": [e["time"] for e in scan_ts]}).sort("time")

    # -----------------------------
    # Scan–map features
    # -----------------------------
    scanmap_cols = [
        "point_distance", "point_fitting", "point_inlier", "point_quality",
        "ray_inlier", "ray_inlier_percent", "ray_matching_percent",
        "ray_outlier_percent", "ray_quality",
        "angle_inlier", "angle_quality",
        "line_angle", "line_distance", "line_fitting", "line_length",
    ]

    for col in scanmap_cols:
        ts = timeseries_to_df(full_df, col, col)
        base = base.join(ts, on="time", how="inner")

    # -----------------------------
    # Particle cloud features
    # -----------------------------
    pcs_cols = [
        "cog_max_distance", "cog_mean_dist", "cog_mean_absolute_deviation",
        "cog_median", "cog_median_absolute_deviation",
        "cog_min_distance", "cog_standard_deviation",
        "circle_radius", "circle_mean", "circle_mean_absolute_deviation",
        "circle_median", "circle_median_absolute_deviation",
        "circle_min_distance", "circle_standard_deviation",
        "num_clusters",
        "main_cluster_variance_x", "main_cluster_variance_y",
    ]

    for col in pcs_cols:
        ts = timeseries_to_df(full_df, col, col)
        if ts.height > 0:
            base = base.join_asof(ts.sort("time"), on="time", strategy="backward")

    # -----------------------------
    # AMCL pose + covariance
    # -----------------------------
    amcl_rows = []
    for e in full_df["/amcl_pose"][0]:
        t = e["time"]
        v = e["value"]
        row = {
            "time": t,
            "amcl_x": v["pose.pose.position.x"],
            "amcl_y": v["pose.pose.position.y"],
            "amcl_qx": v["pose.pose.orientation.x"],
            "amcl_qy": v["pose.pose.orientation.y"],
            "amcl_qz": v["pose.pose.orientation.z"],
            "amcl_qw": v["pose.pose.orientation.w"],
        }
        cov = v.get("pose.covariance")
        if cov is not None and len(cov) >= 36:
            row["amcl_cov_x"] = float(cov[0])
            row["amcl_cov_y"] = float(cov[7])
            row["amcl_cov_yaw"] = float(cov[35])
        amcl_rows.append(row)
    amcl_df = pl.DataFrame(amcl_rows).sort("time")
    base = base.join_asof(amcl_df, on="time", strategy="backward")

    # -----------------------------
    # Odometry velocity
    # -----------------------------
    if "/imperfect_odom" in full_df.columns:
        odom_rows = []
        for e in full_df["/imperfect_odom"][0]:
            t = e["time"]
            v = e["value"]
            odom_rows.append({
                "time": t,
                "odom_linear_x": float(v["twist.twist.linear.x"]),
                "odom_angular_z": float(v["twist.twist.angular.z"]),
            })
        if odom_rows:
            odom_df = pl.DataFrame(odom_rows).sort("time")
            base = base.join_asof(odom_df, on="time", strategy="backward")

    # -----------------------------
    # Commanded velocity
    # -----------------------------
    if "/cmd_vel" in full_df.columns:
        cmd_rows = []
        for e in full_df["/cmd_vel"][0]:
            t = e["time"]
            v = e["value"]
            cmd_rows.append({
                "time": t,
                "cmd_linear_x": float(v["linear.x"]),
                "cmd_angular_z": float(v["angular.z"]),
            })
        if cmd_rows:
            cmd_df = pl.DataFrame(cmd_rows).sort("time")
            base = base.join_asof(cmd_df, on="time", strategy="backward")

    # -----------------------------
    # GT pose
    # -----------------------------
    gt_rows = []
    for e in full_df["/momo/pose"][0]:
        t = e["time"]
        v = e["value"]
        gt_rows.append({
            "time": t,
            "gt_x": v["pose.position.x"],
            "gt_y": v["pose.position.y"],
            "gt_qx": v["pose.orientation.x"],
            "gt_qy": v["pose.orientation.y"],
            "gt_qz": v["pose.orientation.z"],
            "gt_qw": v["pose.orientation.w"],
        })
    gt_df = pl.DataFrame(gt_rows).sort("time")
    base = base.join_asof(gt_df, on="time", strategy="backward")

    base = base.drop_nulls(subset=["amcl_x", "gt_x"])

    # -----------------------------
    # Errors and labels
    # -----------------------------
    base = base.with_columns([
        pl.struct("amcl_qx", "amcl_qy", "amcl_qz", "amcl_qw")
        .map_elements(lambda s: yaw_from_quat(
            s["amcl_qx"], s["amcl_qy"], s["amcl_qz"], s["amcl_qw"]))
        .alias("amcl_yaw"),

        pl.struct("gt_qx", "gt_qy", "gt_qz", "gt_qw")
        .map_elements(lambda s: yaw_from_quat(
            s["gt_qx"], s["gt_qy"], s["gt_qz"], s["gt_qw"]))
        .alias("gt_yaw"),
    ])

    base = base.with_columns([
        (((pl.col("gt_x") - pl.col("amcl_x"))**2 +
          (pl.col("gt_y") - pl.col("amcl_y"))**2).sqrt())
        .alias("position_error"),

        (pl.col("gt_yaw") - pl.col("amcl_yaw")).alias("heading_error_raw")
    ])

    base = base.with_columns(
        ((pl.col("heading_error_raw") + np.pi) % (2*np.pi) - np.pi)
        .alias("heading_error")
    )

    base = base.with_columns(
        ((pl.col("position_error") > position_threshold) |
         (pl.col("heading_error").abs() > heading_threshold))
        .alias("is_delocalized")
    )

    base = base.with_columns(
        (pl.col("position_error") +
         0.5 * pl.col("heading_error").abs())
        .alias("combined_error")
    )

    # -----------------------------
    # Multi-label columns
    # -----------------------------

    # Threshold-based labels: True when error exceeds threshold
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        col_name = f"lbl_err_{int(round(thr * 100)):03d}"
        base = base.with_columns(
            ((pl.col("position_error") > thr) | (pl.col("heading_error").abs() > thr))
            .alias(col_name)
        )

    # Predictive-window labels: True if an AMCL reset occurs within next W seconds.
    # Post-reset margin (5 s) is treated as False — robot is re-localising, not failing.
    WINDOW_SECONDS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    MARGIN_S = 5.0

    reset_times_ns: np.ndarray = np.array([], dtype=np.int64)
    try:
        raw_resets: list[int] = []
        with AnyReader([Path(bag_path)]) as reader:
            conns = [c for c in reader.connections if c.topic == "/initialpose"]
            for _conn, ts, _raw in reader.messages(connections=conns):
                raw_resets.append(int(ts))
        if raw_resets:
            reset_times_ns = np.array(sorted(raw_resets), dtype=np.int64)
            print(f"  /initialpose resets found: {len(reset_times_ns)}")
        else:
            print("  ⚠ No /initialpose messages — predictive-window labels will be all False.")
    except Exception as exc:
        print(f"  ⚠ Could not read /initialpose: {exc}")

    sample_times = base["time"].to_numpy().astype(np.int64)
    margin_ns = int(MARGIN_S * 1e9)

    # Build post-reset margin mask (shared across all windows)
    in_margin = np.zeros(len(sample_times), dtype=bool)
    for r in reset_times_ns:
        in_margin |= (sample_times >= r) & (sample_times < r + margin_ns)

    win_series: list[pl.Series] = []
    for W in WINDOW_SECONDS:
        col_name = f"lbl_win_{W:02d}s"
        if len(reset_times_ns) == 0:
            win_series.append(pl.Series(col_name, np.zeros(len(sample_times), dtype=bool)))
            continue
        window_ns = int(W * 1e9)
        # Any reset r such that r > t AND r <= t + W?
        # searchsorted(..., side="right") gives first index where resets[i] > t
        lo = np.searchsorted(reset_times_ns, sample_times, side="right")
        hi = np.searchsorted(reset_times_ns, sample_times + window_ns, side="right")
        in_window = lo < hi
        in_window[in_margin] = False
        win_series.append(pl.Series(col_name, in_window))

    if win_series:
        base = base.with_columns(win_series)

    return base
