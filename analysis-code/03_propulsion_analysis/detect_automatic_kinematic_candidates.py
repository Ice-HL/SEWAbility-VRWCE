#!/usr/bin/env python3
"""Detect bilateral-forward automatic kinematic candidates in retained P2 intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_WIDTH_PX = 3840.0
IMAGE_HEIGHT_PX = 2160.0
NEEDLE_X_NORM = 0.34713542
NEEDLE_Y_NORM = 0.54768519
NEEDLE_SCREEN_PX = np.array(
    [NEEDLE_X_NORM * IMAGE_WIDTH_PX, NEEDLE_Y_NORM * IMAGE_HEIGHT_PX]
)

ROLLING_MEDIAN_FRAMES = 5
SAVGOL_WINDOW_FRAMES = 7
MAX_INTERPOLATION_FRAMES = 3
FORWARD_VELOCITY_THRESHOLD_SW_S = 0.010
SYNCHRONOUS_GAP_BRIDGE_S = 0.15
MIN_SYNCHRONOUS_COVERAGE_S = 0.30
MIN_EACH_WRIST_NET_FORWARD_DISPLACEMENT_SW = 0.005
NO_RETURN_MERGE_MAX_GAP_S = 0.50
NO_RETURN_TOLERANCE_SW = 0.010


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect automatic bilateral-forward P2 kinematic candidates."
    )
    parser.add_argument(
        "--phase-intervals",
        type=Path,
        required=True,
        help="CSV with exactly one retained P2 interval per work cycle.",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        required=True,
        help="Directory containing {video_id}_trajectory.csv files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cycles", type=int, default=210)
    parser.add_argument("--expected-candidates", type=int, default=4916)
    return parser.parse_args()


def true_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def close_short_gaps(mask: np.ndarray, maximum_gap_frames: int) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    index = 0
    while index < len(output):
        if output[index]:
            index += 1
            continue
        end = index
        while end < len(output) and not output[end]:
            end += 1
        if index > 0 and end < len(output) and end - index <= maximum_gap_frames:
            output[index:end] = True
        index = end
    return output


def savgol7_quadratic(values: np.ndarray) -> np.ndarray:
    """Seven-point quadratic Savitzky–Golay smoother without SciPy."""
    values = np.asarray(values, dtype=float)
    if len(values) < SAVGOL_WINDOW_FRAMES:
        return values.copy()
    coefficients = np.array([-2, 3, 6, 7, 6, 3, -2], dtype=float) / 21.0
    smoothed = np.convolve(values, coefficients, mode="same")
    time = np.arange(SAVGOL_WINDOW_FRAMES, dtype=float)
    smoothed[:3] = np.polyval(np.polyfit(time, values[:7], 2), time[:3])
    smoothed[-3:] = np.polyval(np.polyfit(time, values[-7:], 2), time[-3:])
    return smoothed


def smooth_projected_position(
    values: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    source = pd.Series(np.asarray(values, dtype=float))
    interpolated = source.interpolate(
        limit=MAX_INTERPOLATION_FRAMES, limit_area="inside"
    ).to_numpy(dtype=float)
    smoothed = np.full(len(interpolated), np.nan)
    velocity = np.full(len(interpolated), np.nan)
    for start, end in true_intervals(np.isfinite(interpolated)):
        run = interpolated[start : end + 1]
        median_smoothed = (
            pd.Series(run)
            .rolling(ROLLING_MEDIAN_FRAMES, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )
        run_smoothed = savgol7_quadratic(median_smoothed)
        smoothed[start : end + 1] = run_smoothed
        if len(run_smoothed) >= 2:
            velocity[start : end + 1] = np.gradient(run_smoothed) * fps
    return smoothed, velocity


def standardize_intervals(path: Path, expected_cycles: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    aliases = {
        "Video ID": "video_id",
        "Work cycle ID": "work_cycle_id",
        "Phase": "phase",
        "Start frame": "start_frame",
        "End frame (exclusive)": "end_frame_exclusive",
        "FPS": "fps",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    required = {
        "video_id",
        "work_cycle_id",
        "phase",
        "start_frame",
        "end_frame_exclusive",
        "fps",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"phase interval table is missing columns: {missing}")
    frame = frame[frame.phase.astype(str).eq("P2")].copy()
    if len(frame) != expected_cycles:
        raise ValueError(
            f"Expected {expected_cycles} retained P2 intervals; found {len(frame)}"
        )
    frame["video_id"] = pd.to_numeric(frame.video_id, errors="raise").astype(int)
    frame["work_cycle_id"] = pd.to_numeric(
        frame.work_cycle_id, errors="raise"
    ).astype(int)
    frame["start_frame"] = pd.to_numeric(frame.start_frame, errors="raise").astype(int)
    frame["end_frame_exclusive"] = pd.to_numeric(
        frame.end_frame_exclusive, errors="raise"
    ).astype(int)
    frame["fps"] = pd.to_numeric(frame.fps, errors="raise")
    frame["cycle_key"] = (
        frame.video_id.astype(str) + ":" + frame.work_cycle_id.astype(str)
    )
    if frame.cycle_key.duplicated().any():
        raise ValueError("P2 interval table must have exactly one row per cycle")
    return frame.sort_values(["video_id", "work_cycle_id"])


def load_trajectory(path: Path) -> pd.DataFrame:
    required = [
        "frame",
        "left_shoulder_x",
        "left_shoulder_y",
        "right_shoulder_x",
        "right_shoulder_y",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
    ]
    frame = pd.read_csv(path, usecols=required)
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("frame").drop_duplicates("frame", keep="first")
    frame["frame"] = frame.frame.astype(int)
    return frame


def prepare_cycle(
    trajectory: pd.DataFrame, interval: pd.Series
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    start = int(interval.start_frame)
    end_exclusive = int(interval.end_frame_exclusive)
    fps = float(interval.fps)
    segment = trajectory[
        trajectory.frame.ge(start) & trajectory.frame.lt(end_exclusive)
    ].copy()
    segment = segment.sort_values("frame").reset_index(drop=True)
    if len(segment) < 2:
        raise ValueError(f"Too few rows in P2 interval {interval.cycle_key}")

    shoulder_width = np.hypot(
        (segment.left_shoulder_x - segment.right_shoulder_x) * IMAGE_WIDTH_PX,
        (segment.left_shoulder_y - segment.right_shoulder_y) * IMAGE_HEIGHT_PX,
    )
    median_shoulder_width_px = float(np.nanmedian(shoulder_width))
    if not np.isfinite(median_shoulder_width_px) or median_shoulder_width_px <= 0:
        raise ValueError(f"Invalid shoulder-width scale for {interval.cycle_key}")

    shoulder_midpoint = np.column_stack(
        [
            0.5 * (segment.left_shoulder_x + segment.right_shoulder_x)
            * IMAGE_WIDTH_PX,
            0.5 * (segment.left_shoulder_y + segment.right_shoulder_y)
            * IMAGE_HEIGHT_PX,
        ]
    )
    median_shoulder_midpoint = np.nanmedian(shoulder_midpoint, axis=0)
    outward_axis = NEEDLE_SCREEN_PX - median_shoulder_midpoint
    outward_axis /= np.linalg.norm(outward_axis)

    positions: dict[str, np.ndarray] = {}
    velocities: dict[str, np.ndarray] = {}
    for hand in ("left", "right"):
        wrist = np.column_stack(
            [
                segment[f"{hand}_wrist_x"].to_numpy(float) * IMAGE_WIDTH_PX,
                segment[f"{hand}_wrist_y"].to_numpy(float) * IMAGE_HEIGHT_PX,
            ]
        )
        projected = wrist @ outward_axis / median_shoulder_width_px
        positions[hand], velocities[hand] = smooth_projected_position(projected, fps)

    metadata = {
        "fps": fps,
        "p2_start_frame": start,
        "p2_end_frame_exclusive": end_exclusive,
        "median_shoulder_width_px": median_shoulder_width_px,
        "outward_axis_screen_x": float(outward_axis[0]),
        "outward_axis_screen_y": float(outward_axis[1]),
    }
    return segment, positions, velocities, metadata


def detect_raw_candidates(
    segment: pd.DataFrame,
    positions: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray],
    fps: float,
) -> list[dict]:
    synchronous = (
        (velocities["left"] > FORWARD_VELOCITY_THRESHOLD_SW_S)
        & (velocities["right"] > FORWARD_VELOCITY_THRESHOLD_SW_S)
    )
    synchronous = close_short_gaps(
        synchronous, int(round(SYNCHRONOUS_GAP_BRIDGE_S * fps))
    )
    minimum_frames = int(round(MIN_SYNCHRONOUS_COVERAGE_S * fps))
    frame_numbers = segment.frame.to_numpy(int)
    rows: list[dict] = []
    for start_index, end_index in true_intervals(synchronous):
        coverage_frames = end_index - start_index + 1
        left_displacement = float(
            positions["left"][end_index] - positions["left"][start_index]
        )
        right_displacement = float(
            positions["right"][end_index] - positions["right"][start_index]
        )
        if coverage_frames < minimum_frames:
            continue
        if not np.isfinite(left_displacement + right_displacement):
            continue
        if min(left_displacement, right_displacement) < MIN_EACH_WRIST_NET_FORWARD_DISPLACEMENT_SW:
            continue
        rows.append(
            {
                "start_index_in_p2": start_index,
                "end_index_in_p2": end_index,
                "start_frame": int(frame_numbers[start_index]),
                "end_frame": int(frame_numbers[end_index]),
                "coverage_frames": coverage_frames,
                "left_net_forward_displacement_sw": left_displacement,
                "right_net_forward_displacement_sw": right_displacement,
            }
        )
    return rows


def merge_without_return(
    raw_candidates: list[dict], positions: dict[str, np.ndarray], fps: float
) -> list[dict]:
    max_gap_frames = int(round(NO_RETURN_MERGE_MAX_GAP_S * fps))
    groups: list[dict] = []
    for raw_index, candidate in enumerate(raw_candidates, start=1):
        if not groups:
            merge = False
        else:
            prior_end = int(groups[-1]["end_index_in_p2"])
            current_start = int(candidate["start_index_in_p2"])
            gap_frames = current_start - prior_end - 1
            merge = gap_frames <= max_gap_frames
            for hand in ("left", "right"):
                bridge = positions[hand][prior_end : current_start + 1]
                if len(bridge) == 0 or not np.isfinite(bridge).all():
                    merge = False
                    break
                relative = bridge - positions[hand][prior_end]
                if (
                    positions[hand][current_start] - positions[hand][prior_end]
                    < -NO_RETURN_TOLERANCE_SW
                    or np.min(relative) < -NO_RETURN_TOLERANCE_SW
                ):
                    merge = False
                    break
        if merge:
            groups[-1]["end_index_in_p2"] = candidate["end_index_in_p2"]
            groups[-1]["end_frame"] = candidate["end_frame"]
            groups[-1]["raw_indices"].append(raw_index)
        else:
            groups.append(
                {
                    "start_index_in_p2": candidate["start_index_in_p2"],
                    "end_index_in_p2": candidate["end_index_in_p2"],
                    "start_frame": candidate["start_frame"],
                    "end_frame": candidate["end_frame"],
                    "raw_indices": [raw_index],
                }
            )

    merged: list[dict] = []
    for event_index, group in enumerate(groups, start=1):
        start_index = int(group["start_index_in_p2"])
        end_index = int(group["end_index_in_p2"])
        start_frame = int(group["start_frame"])
        end_frame = int(group["end_frame"])
        merged.append(
            {
                "candidate_index": event_index,
                "source_raw_candidate_indices": ";".join(map(str, group["raw_indices"])),
                "raw_core_count": len(group["raw_indices"]),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": start_frame / fps,
                "end_time_s": end_frame / fps,
                "duration_s": (end_frame - start_frame) / fps,
                "left_net_forward_displacement_sw": float(
                    positions["left"][end_index] - positions["left"][start_index]
                ),
                "right_net_forward_displacement_sw": float(
                    positions["right"][end_index] - positions["right"][start_index]
                ),
                "candidate_type": "automatic_kinematic_candidate",
            }
        )
    return merged


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    intervals = standardize_intervals(args.phase_intervals, args.expected_cycles)
    trajectory_cache: dict[int, pd.DataFrame] = {}
    candidate_rows: list[dict] = []
    cycle_rows: list[dict] = []
    for interval in intervals.itertuples(index=False):
        video_id = int(interval.video_id)
        if video_id not in trajectory_cache:
            path = args.trajectory_dir / f"{video_id}_trajectory.csv"
            trajectory_cache[video_id] = load_trajectory(path)
        segment, positions, velocities, metadata = prepare_cycle(
            trajectory_cache[video_id], pd.Series(interval._asdict())
        )
        raw = detect_raw_candidates(
            segment, positions, velocities, float(metadata["fps"])
        )
        merged = merge_without_return(raw, positions, float(metadata["fps"]))
        common = {
            "video_id": video_id,
            "work_cycle_id": int(interval.work_cycle_id),
            "cycle_key": str(interval.cycle_key),
        }
        candidate_rows.extend({**common, **row} for row in merged)
        cycle_rows.append(
            {
                **common,
                **metadata,
                "raw_candidate_count": len(raw),
                "automatic_kinematic_candidate_count": len(merged),
            }
        )

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["video_id", "work_cycle_id", "candidate_index"]
    )
    cycles = pd.DataFrame(cycle_rows).sort_values(["video_id", "work_cycle_id"])
    if len(cycles) != args.expected_cycles or cycles.cycle_key.duplicated().any():
        raise RuntimeError("Cycle-level detector output failed cohort checks")
    if len(candidates) != args.expected_candidates:
        raise RuntimeError(
            f"Expected {args.expected_candidates} automatic candidates; "
            f"detected {len(candidates)}"
        )
    if int(cycles.automatic_kinematic_candidate_count.sum()) != len(candidates):
        raise RuntimeError("Cycle counts do not reconcile with candidate rows")

    candidates.to_csv(output / "automatic_kinematic_candidates.csv", index=False)
    cycles.to_csv(output / "candidate_counts_by_cycle.csv", index=False)
    manifest = {
        "retained_cycles": int(len(cycles)),
        "automatic_kinematic_candidates": int(len(candidates)),
        "candidate_count_median": float(
            cycles.automatic_kinematic_candidate_count.median()
        ),
        "candidate_count_q1": float(
            cycles.automatic_kinematic_candidate_count.quantile(0.25)
        ),
        "candidate_count_q3": float(
            cycles.automatic_kinematic_candidate_count.quantile(0.75)
        ),
        "interpretation_boundary": (
            "Candidates satisfy the prespecified bilateral-forward kinematic "
            "rule. They were not independently adjudicated as semantic sewing "
            "actions and are therefore reported as automatic kinematic candidates."
        ),
    }
    (output / "candidate_detection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
