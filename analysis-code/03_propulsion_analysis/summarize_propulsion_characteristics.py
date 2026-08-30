#!/usr/bin/env python3
"""Summarize direction, geometry, timing, and speed of P2 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


IMAGE_WIDTH_PX = 3840.0
IMAGE_HEIGHT_PX = 2160.0
NEEDLE_ANALYSIS_PX = np.array([1333.0, -1183.0])
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260815
MAX_INTERPOLATION_FRAMES = 3
ROLLING_MEDIAN_FRAMES = 5
SAVGOL_WINDOW_FRAMES = 7

WRIST_COLUMNS = (
    "left_wrist_x",
    "left_wrist_y",
    "right_wrist_x",
    "right_wrist_y",
)
SHOULDER_COLUMNS = (
    "left_shoulder_x",
    "left_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_y",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive retained-cohort P2 propulsion characteristics."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--cycle-counts", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cycles", type=int, default=210)
    parser.add_argument("--expected-candidates", type=int, default=4916)
    parser.add_argument("--expected-gaps", type=int, default=4706)
    parser.add_argument("--reference-shoulder-width-m", type=float, default=0.40)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def true_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def savgol7_quadratic(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < SAVGOL_WINDOW_FRAMES:
        return values.copy()
    coefficients = np.array([-2, 3, 6, 7, 6, 3, -2], dtype=float) / 21.0
    smoothed = np.convolve(values, coefficients, mode="same")
    time = np.arange(SAVGOL_WINDOW_FRAMES, dtype=float)
    smoothed[:3] = np.polyval(np.polyfit(time, values[:7], 2), time[:3])
    smoothed[-3:] = np.polyval(np.polyfit(time, values[-7:], 2), time[-3:])
    return smoothed


def smooth_coordinate(values: np.ndarray) -> np.ndarray:
    source = pd.Series(np.asarray(values, dtype=float))
    interpolated = source.interpolate(
        limit=MAX_INTERPOLATION_FRAMES, limit_area="inside"
    ).to_numpy(dtype=float)
    output = np.full(len(interpolated), np.nan)
    for start, end in true_intervals(np.isfinite(interpolated)):
        run = interpolated[start : end + 1]
        median_smoothed = (
            pd.Series(run)
            .rolling(ROLLING_MEDIAN_FRAMES, center=True, min_periods=1)
            .median()
            .to_numpy(float)
        )
        output[start : end + 1] = savgol7_quadratic(median_smoothed)
    return output


def load_raw_trajectory(path: Path) -> pd.DataFrame:
    columns = ["frame", *SHOULDER_COLUMNS, *WRIST_COLUMNS]
    data = pd.read_csv(path, usecols=columns)
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values("frame").drop_duplicates("frame", keep="first")
    data["frame"] = data.frame.astype(int)
    return data.set_index("frame", drop=False)


def add_direction_smoothing(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    for column in WRIST_COLUMNS:
        data[f"{column}_dir"] = data[column].rolling(
            3, center=True, min_periods=2
        ).median()
    return data


def add_metric_smoothing(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    for column in WRIST_COLUMNS:
        data[f"{column}_metric"] = smooth_coordinate(data[column].to_numpy())
    return data


def shoulder_midpoint(
    raw: pd.DataFrame, start: int, end_exclusive: int
) -> np.ndarray:
    segment = raw.loc[(raw.index >= start) & (raw.index < end_exclusive)]
    segment = segment.dropna(subset=list(SHOULDER_COLUMNS))
    if segment.empty:
        return np.array([np.nan, np.nan])
    points = np.column_stack(
        [
            0.5 * (segment.left_shoulder_x + segment.right_shoulder_x)
            * IMAGE_WIDTH_PX,
            -0.5 * (segment.left_shoulder_y + segment.right_shoulder_y)
            * IMAGE_HEIGHT_PX,
        ]
    )
    return np.nanmedian(points, axis=0)


def wrist_midpoints(
    raw: pd.DataFrame, start: int, end: int, suffix: str
) -> tuple[np.ndarray, np.ndarray]:
    segment = raw.loc[(raw.index >= start) & (raw.index <= end)]
    columns = [f"{column}_{suffix}" for column in WRIST_COLUMNS]
    segment = segment.dropna(subset=columns)
    if segment.empty:
        return np.empty(0, dtype=int), np.empty((0, 2))
    points = np.column_stack(
        [
            0.5
            * (segment[f"left_wrist_x_{suffix}"] + segment[f"right_wrist_x_{suffix}"])
            * IMAGE_WIDTH_PX,
            -0.5
            * (segment[f"left_wrist_y_{suffix}"] + segment[f"right_wrist_y_{suffix}"])
            * IMAGE_HEIGHT_PX,
        ]
    )
    return segment.frame.to_numpy(int), points


def pca_axis(points: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 3:
        raise ValueError("Cycle direction requires at least three paired samples")
    centered = points - points.mean(axis=0, keepdims=True)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    values = np.maximum(values, 0)
    if values.sum() <= np.finfo(float).eps:
        raise ValueError("Degenerate cycle-level PCA")
    axis = vectors[:, -1].astype(float)
    if np.dot(axis, reference) < 0:
        axis *= -1
    axis /= np.linalg.norm(axis)
    return axis, float(values[-1] / values.sum())


def signed_angle_deg(reference: np.ndarray, direction: np.ndarray) -> float:
    cross = reference[0] * direction[1] - reference[1] * direction[0]
    return float(np.degrees(np.arctan2(cross, np.dot(reference, direction))))


def wrap_degrees(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values + 180.0) % 360.0 - 180.0


def circular_summary(degrees: Iterable[float]) -> dict:
    degrees = np.asarray(list(degrees), dtype=float)
    radians = np.radians(degrees)
    c = float(np.mean(np.cos(radians)))
    s = float(np.mean(np.sin(radians)))
    resultant = float(np.hypot(c, s))
    mean = float(np.degrees(np.arctan2(s, c)))
    circular_sd = (
        float(np.degrees(np.sqrt(-2 * np.log(resultant))))
        if resultant > 0
        else float("inf")
    )
    absolute = np.abs(wrap_degrees(degrees))
    return {
        "n": int(len(degrees)),
        "signed_circular_mean_deg": mean,
        "circular_sd_deg": circular_sd,
        "mean_resultant_length_R": resultant,
        "median_absolute_deviation_deg": float(np.median(absolute)),
        "q1_absolute_deviation_deg": float(np.quantile(absolute, 0.25)),
        "q3_absolute_deviation_deg": float(np.quantile(absolute, 0.75)),
    }


def video_cluster_bootstrap(cycles: pd.DataFrame) -> dict:
    videos = np.asarray(sorted(cycles.video_id.unique()), dtype=int)
    by_video = {
        int(video): cycles.loc[cycles.video_id.eq(video), "angle_deg"].to_numpy()
        for video in videos
    }
    point_mean = circular_summary(cycles.angle_deg)["signed_circular_mean_deg"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(videos, size=len(videos), replace=True)
        angles = np.concatenate([by_video[int(video)] for video in sampled])
        means[index] = circular_summary(angles)["signed_circular_mean_deg"]
    offsets = wrap_degrees(means - point_mean)
    interval = wrap_degrees(point_mean + np.quantile(offsets, [0.025, 0.975]))
    return {
        "cluster_unit": "video_id",
        "video_clusters": int(len(videos)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "signed_circular_mean_95_percent_interval_deg": [
            float(interval[0]),
            float(interval[1]),
        ],
    }


def derive_cycle_axes(
    candidates: pd.DataFrame,
    cycles: pd.DataFrame,
    raw_by_video: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    references: dict[int, np.ndarray] = {}
    for video_id, rows in cycles.groupby("video_id", sort=True):
        raw = raw_by_video[int(video_id)]
        midpoints = np.vstack(
            [
                shoulder_midpoint(
                    raw, int(row.p2_start_frame), int(row.p2_end_frame_exclusive)
                )
                for row in rows.itertuples(index=False)
            ]
        )
        video_midpoint = np.nanmedian(midpoints, axis=0)
        reference = NEEDLE_ANALYSIS_PX - video_midpoint
        reference /= np.linalg.norm(reference)
        references[int(video_id)] = reference

    rows: list[dict] = []
    for cycle in cycles.itertuples(index=False):
        cycle_candidates = candidates[candidates.cycle_key.eq(str(cycle.cycle_key))]
        raw = add_direction_smoothing(raw_by_video[int(cycle.video_id)])
        frame_parts: list[np.ndarray] = []
        point_parts: list[np.ndarray] = []
        for candidate in cycle_candidates.itertuples(index=False):
            frames, points = wrist_midpoints(
                raw, int(candidate.start_frame), int(candidate.end_frame), "dir"
            )
            frame_parts.append(frames)
            point_parts.append(points)
        if not point_parts:
            raise ValueError(f"No candidate frames for cycle {cycle.cycle_key}")
        frames = np.concatenate(frame_parts)
        points = np.vstack(point_parts)
        _, unique_indices = np.unique(frames, return_index=True)
        points = points[np.sort(unique_indices)]
        reference = references[int(cycle.video_id)]
        axis, variance_fraction = pca_axis(points, reference)
        angle = signed_angle_deg(reference, axis)
        rows.append(
            {
                "video_id": int(cycle.video_id),
                "work_cycle_id": int(cycle.work_cycle_id),
                "cycle_key": str(cycle.cycle_key),
                "paired_candidate_frames": int(len(points)),
                "task_outward_x": float(reference[0]),
                "task_outward_y": float(reference[1]),
                "propulsion_axis_x": float(axis[0]),
                "propulsion_axis_y": float(axis[1]),
                "angle_deg": angle,
                "absolute_angle_deg": abs(angle),
                "pc1_variance_fraction": variance_fraction,
            }
        )
    return pd.DataFrame(rows)


def event_metric(
    candidate: pd.Series,
    raw: pd.DataFrame,
    cycle: pd.Series,
    axis_row: pd.Series,
) -> dict:
    start = int(candidate.start_frame)
    end = int(candidate.end_frame)
    segment = raw.loc[(raw.index >= start) & (raw.index <= end)]
    columns = [f"{column}_metric" for column in WRIST_COLUMNS]
    segment = segment.dropna(subset=columns)
    if len(segment) < 2:
        raise ValueError(f"Too few paired samples for {candidate.cycle_key}")

    left = np.column_stack(
        [
            segment.left_wrist_x_metric.to_numpy() * IMAGE_WIDTH_PX,
            -segment.left_wrist_y_metric.to_numpy() * IMAGE_HEIGHT_PX,
        ]
    )
    right = np.column_stack(
        [
            segment.right_wrist_x_metric.to_numpy() * IMAGE_WIDTH_PX,
            -segment.right_wrist_y_metric.to_numpy() * IMAGE_HEIGHT_PX,
        ]
    )
    midpoint = 0.5 * (left + right)
    propulsion = np.array(
        [axis_row.propulsion_axis_x, axis_row.propulsion_axis_y], dtype=float
    )
    propulsion /= np.linalg.norm(propulsion)
    lateral = np.array([-propulsion[1], propulsion[0]])
    shoulder_width = float(cycle.median_shoulder_width_px)
    axial = midpoint @ propulsion
    lateral_position = midpoint @ lateral
    transverse_separation = np.abs((right - left) @ lateral)
    frames = segment.frame.to_numpy(int)
    increments = np.linalg.norm(np.diff(midpoint, axis=0), axis=1)
    adjacent = np.diff(frames) == 1
    path_length_sw = float(increments[adjacent].sum() / shoulder_width)
    duration = float(candidate.duration_s)
    return {
        "video_id": int(candidate.video_id),
        "work_cycle_id": int(candidate.work_cycle_id),
        "cycle_key": str(candidate.cycle_key),
        "candidate_index": int(candidate.candidate_index),
        "start_frame": start,
        "end_frame": end,
        "start_time_s": float(candidate.start_time_s),
        "end_time_s": float(candidate.end_time_s),
        "duration_s": duration,
        "valid_bilateral_samples": int(len(segment)),
        "axial_displacement_sw": abs(float(axial[-1] - axial[0])) / shoulder_width,
        "signed_axial_displacement_sw": float(axial[-1] - axial[0]) / shoulder_width,
        "lateral_excursion_sw": float(lateral_position.max() - lateral_position.min())
        / shoulder_width,
        "transverse_wrist_separation_sw": float(np.median(transverse_separation))
        / shoulder_width,
        "wrist_midpoint_path_length_sw": path_length_sw,
        "wrist_midpoint_path_speed_sw_per_s": (
            path_length_sw / duration if duration > 0 else np.nan
        ),
    }


def distribution(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n": int(len(values)),
        "median": float(values.median()),
        "q1": float(values.quantile(0.25)),
        "q3": float(values.quantile(0.75)),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates)
    cycles = pd.read_csv(args.cycle_counts)
    require_columns(
        candidates,
        {
            "video_id",
            "work_cycle_id",
            "cycle_key",
            "candidate_index",
            "start_frame",
            "end_frame",
            "start_time_s",
            "end_time_s",
            "duration_s",
        },
        "candidate table",
    )
    require_columns(
        cycles,
        {
            "video_id",
            "work_cycle_id",
            "cycle_key",
            "p2_start_frame",
            "p2_end_frame_exclusive",
            "fps",
            "median_shoulder_width_px",
            "automatic_kinematic_candidate_count",
        },
        "cycle-count table",
    )
    candidates["cycle_key"] = candidates.cycle_key.astype(str)
    cycles["cycle_key"] = cycles.cycle_key.astype(str)
    if len(cycles) != args.expected_cycles or len(candidates) != args.expected_candidates:
        raise ValueError(
            f"Expected {args.expected_cycles} cycles/{args.expected_candidates} "
            f"candidates; found {len(cycles)}/{len(candidates)}"
        )

    raw_by_video = {
        int(video): load_raw_trajectory(
            args.trajectory_dir / f"{int(video)}_trajectory.csv"
        )
        for video in sorted(cycles.video_id.unique())
    }
    axes = derive_cycle_axes(candidates, cycles, raw_by_video)
    if len(axes) != args.expected_cycles:
        raise RuntimeError("Every retained cycle must yield one direction estimate")

    cycle_lookup = cycles.set_index("cycle_key", drop=False)
    axis_lookup = axes.set_index("cycle_key", drop=False)
    metric_raw = {
        video: add_metric_smoothing(raw) for video, raw in raw_by_video.items()
    }
    rows = []
    for candidate in candidates.sort_values(
        ["video_id", "work_cycle_id", "candidate_index"]
    ).itertuples(index=False):
        key = str(candidate.cycle_key)
        rows.append(
            event_metric(
                pd.Series(candidate._asdict()),
                metric_raw[int(candidate.video_id)],
                cycle_lookup.loc[key],
                axis_lookup.loc[key],
            )
        )
    metrics = pd.DataFrame(rows)

    gap_rows: list[dict] = []
    for key, group in metrics.groupby("cycle_key", sort=False):
        ordered = group.sort_values("start_frame").reset_index(drop=True)
        for index in range(1, len(ordered)):
            previous = ordered.iloc[index - 1]
            current = ordered.iloc[index]
            gap_rows.append(
                {
                    "video_id": int(current.video_id),
                    "work_cycle_id": int(current.work_cycle_id),
                    "cycle_key": str(key),
                    "preceding_candidate_index": int(previous.candidate_index),
                    "current_candidate_index": int(current.candidate_index),
                    "inter_action_gap_s": float(
                        current.start_time_s - previous.end_time_s
                    ),
                }
            )
    gaps = pd.DataFrame(gap_rows)
    if len(gaps) != args.expected_gaps:
        raise RuntimeError(
            f"Expected {args.expected_gaps} within-cycle adjacent pairs; found {len(gaps)}"
        )

    cycle_medians = (
        metrics.groupby("cycle_key", as_index=False)
        .agg(
            cycle_median_axial_displacement_sw=("axial_displacement_sw", "median"),
            cycle_median_lateral_excursion_sw=("lateral_excursion_sw", "median"),
        )
        .merge(cycles[["cycle_key", "video_id", "work_cycle_id"]], on="cycle_key")
    )
    d_p90 = float(cycle_medians.cycle_median_axial_displacement_sw.quantile(0.90))
    w_p90 = float(cycle_medians.cycle_median_lateral_excursion_sw.quantile(0.90))
    count_distribution = distribution(cycles.automatic_kinematic_candidate_count)
    direction = circular_summary(axes.angle_deg)
    bootstrap = video_cluster_bootstrap(axes)
    separation = distribution(metrics.transverse_wrist_separation_sw)
    duration = distribution(metrics.duration_s)
    gap = distribution(gaps.inter_action_gap_s)
    speed = distribution(metrics.wrist_midpoint_path_speed_sw_per_s)
    reference_width = float(args.reference_shoulder_width_m)
    summary = {
        "cohort": {
            "retained_cycles": int(len(cycles)),
            "automatic_kinematic_candidates": int(len(candidates)),
            "within_cycle_adjacent_candidate_pairs": int(len(gaps)),
        },
        "candidate_count_per_cycle": {
            "aggregation": "one count per retained work cycle",
            **count_distribution,
        },
        "propulsion_direction": {
            "aggregation": "one PCA direction estimate per retained work cycle",
            **direction,
            **bootstrap,
        },
        "axial_displacement_design_basis": {
            "aggregation": "P90 across 210 cycle-level medians",
            "p90_sw": d_p90,
            "reference_scaled_m": d_p90 * reference_width,
        },
        "lateral_excursion_design_basis": {
            "aggregation": "P90 across 210 cycle-level medians",
            "p90_sw": w_p90,
            "reference_scaled_m": w_p90 * reference_width,
        },
        "eventwise_transverse_wrist_separation": {
            "aggregation": "pooled candidate-level distribution",
            **separation,
            "reference_scaled_median_m": separation["median"] * reference_width,
        },
        "candidate_duration": {
            "aggregation": "pooled candidate-level distribution",
            **duration,
        },
        "inter_action_gap": {
            "definition": "next candidate onset minus preceding candidate end within cycle",
            "aggregation": "pooled within-cycle adjacent-candidate pairs",
            **gap,
        },
        "wrist_midpoint_path_speed": {
            "aggregation": "pooled candidate-level distribution",
            **speed,
            "reference_scaled_median_m_per_s": speed["median"] * reference_width,
        },
        "reference_scaling": {
            "reference_shoulder_width_m": reference_width,
            "purpose": (
                "Conversion of shoulder-normalized summaries to nominal design "
                "values; not metric recovery from the videos."
            ),
        },
        "interpretation_boundary": (
            "All 4,916 intervals are automatic kinematic candidates. The summary "
            "does not claim semantic confirmation of every interval as a sewing action."
        ),
    }

    if (gaps.inter_action_gap_s < 0).any():
        raise RuntimeError("Candidate intervals overlap; inter-action gaps are negative")
    if not np.isfinite(
        metrics[
            [
                "duration_s",
                "axial_displacement_sw",
                "lateral_excursion_sw",
                "transverse_wrist_separation_sw",
                "wrist_midpoint_path_speed_sw_per_s",
            ]
        ]
    ).all().all():
        raise RuntimeError("At least one primary candidate metric is non-finite")

    axes.to_csv(output / "cycle_direction_estimates.csv", index=False)
    metrics.to_csv(output / "candidate_level_metrics.csv", index=False)
    gaps.to_csv(output / "within_cycle_inter_action_gaps.csv", index=False)
    cycle_medians.to_csv(output / "cycle_level_geometry_medians.csv", index=False)
    (output / "propulsion_characteristics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
