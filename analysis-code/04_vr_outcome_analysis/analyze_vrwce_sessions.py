#!/usr/bin/env python3
"""Derive VR-WCE outcomes from four pseudonymized technical sessions.

The script intentionally separates four analysis populations:

1. All runtime-recorded P2 actions contribute to task progression and workload;
   bilateral participation is reported separately.
2. Bilateral-classified actions contribute to action-duration summaries. An
   inter-action gap is retained only when the two originally adjacent recorded
   actions are both bilateral-classified; gaps are not bridged across excluded
   single-hand actions.
3. Bilateral-classified actions with at least three synchronized samples and
   non-zero paths contribute to basic bilateral trajectory outcomes.
4. Bilateral hand trajectories with at least ten synchronized samples
   contribute to unsmoothed higher-order movement-dynamics outcomes.

No inferential comparison is performed. Action-level observations are nested
within four individual technical sessions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


OUT = Path("results")
FIGURE_STEM = "Fig5_VRWCE_technical_sessions"
SUPPLEMENTARY_STEM = "VRWCE_higher_order_stability"

# Populated from repeated ``--session LABEL=PATH`` arguments at runtime.
# Labels are pseudonyms and need not match the source-directory names.
SESSION_SPECS: list[tuple[str, Path, str]] = []

DEFAULT_PROFILES = {
    "S1": "Program-generated reference",
    "S2": "Program-generated oscillatory",
    "U1": "Human-operated",
    "U2": "Human-operated",
}


def session_source(folder: str | Path) -> Path:
    """Return an explicitly supplied session directory."""
    return Path(folder).expanduser().resolve()

BASIC_MIN_SAMPLES = 3
HIGHER_MIN_SAMPLES = 10


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if "." in text:
        prefix, fraction = text.split(".", 1)
        text = f"{prefix}.{fraction[:6].ljust(6, '0')}"
    return datetime.fromisoformat(text)


def elapsed(start: str | None, end: str | None) -> float | None:
    a, b = parse_time(start), parse_time(end)
    if a is None or b is None:
        return None
    return (b - a).total_seconds()


def pos(value: Any) -> np.ndarray | None:
    if not isinstance(value, dict):
        return None
    try:
        arr = np.asarray([float(value["x"]), float(value["y"]), float(value["z"])])
    except (KeyError, TypeError, ValueError):
        return None
    return arr if np.all(np.isfinite(arr)) else None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def q(values: Iterable[float], p: float) -> float | None:
    vals = np.asarray([float(x) for x in values if x is not None and np.isfinite(x)])
    return float(np.quantile(vals, p)) if vals.size else None


def mean(values: Iterable[float]) -> float | None:
    vals = np.asarray([float(x) for x in values if x is not None and np.isfinite(x)])
    return float(np.mean(vals)) if vals.size else None


def sample_sd(values: Iterable[float]) -> float | None:
    vals = np.asarray([float(x) for x in values if x is not None and np.isfinite(x)])
    return float(np.std(vals, ddof=1)) if vals.size >= 2 else None


def cv(values: Iterable[float]) -> float | None:
    vals = [float(x) for x in values if x is not None and np.isfinite(x)]
    m = mean(vals)
    sd = sample_sd(vals)
    if m is None or sd is None or math.isclose(m, 0.0):
        return None
    return sd / m


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or math.isclose(b, 0.0):
        return None
    return a / b


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = [float(x) for x in values if x is not None and np.isfinite(x)]
    return {
        "n": len(vals),
        "mean": mean(vals),
        "sd": sample_sd(vals),
        "cv": cv(vals),
        "q1": q(vals, 0.25),
        "median": q(vals, 0.50),
        "q3": q(vals, 0.75),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def trajectory_from_samples(
    samples: list[dict[str, Any]],
    position_key: str,
    availability_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[datetime, np.ndarray]] = []
    for sample in samples:
        if availability_key and not sample.get(availability_key):
            continue
        timestamp = parse_time(sample.get("time"))
        point = pos(sample.get(position_key))
        if timestamp is not None and point is not None:
            rows.append((timestamp, point))
    rows.sort(key=lambda item: item[0])
    deduplicated: list[tuple[datetime, np.ndarray]] = []
    for row in rows:
        if deduplicated and row[0] <= deduplicated[-1][0]:
            continue
        deduplicated.append(row)
    if not deduplicated:
        return np.empty((0,), dtype=float), np.empty((0, 3), dtype=float)
    t0 = deduplicated[0][0]
    times = np.asarray([(t - t0).total_seconds() for t, _ in deduplicated], dtype=float)
    points = np.vstack([p for _, p in deduplicated])
    return times, points


def basic_trajectory_metrics(times: np.ndarray, points: np.ndarray) -> dict[str, Any]:
    n = len(points)
    duration = float(times[-1] - times[0]) if n >= 2 else None
    if n < 2:
        return {
            "n": n,
            "duration_s": duration,
            "path_m": None,
            "net_m": None,
            "path_efficiency": None,
            "mean_speed_m_s": None,
            "basic_eligible": False,
        }
    segments = np.diff(points, axis=0)
    path = float(np.sum(np.linalg.norm(segments, axis=1)))
    net = float(np.linalg.norm(points[-1] - points[0]))
    eligible = bool(n >= BASIC_MIN_SAMPLES and duration and duration > 0 and path > 0)
    return {
        "n": n,
        "duration_s": duration,
        "path_m": path,
        "net_m": net,
        "path_efficiency": net / path if eligible else None,
        "mean_speed_m_s": path / duration if duration and duration > 0 else None,
        "basic_eligible": eligible,
    }


def higher_order_metrics(times: np.ndarray, points: np.ndarray) -> dict[str, Any]:
    n = len(points)
    duration = float(times[-1] - times[0]) if n >= 2 else None
    eligible = bool(n >= HIGHER_MIN_SAMPLES)
    empty = {
        "higher_eligible": eligible,
        "speed_variability_m_s": None,
        "acceleration_rms_m_s2": None,
        "jerk_rms_m_s3": None,
        "direction_change_rate_deg_s": None,
    }
    if not eligible:
        return empty
    dt = np.diff(times)
    if np.any(dt <= 0):
        return {**empty, "higher_eligible": False}
    displacement = np.diff(points, axis=0)
    velocity = displacement / dt[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    speed_variability = float(np.std(speed, ddof=1)) if len(speed) >= 2 else None

    velocity_times = (times[:-1] + times[1:]) / 2.0
    dvt = np.diff(velocity_times)
    acceleration = np.diff(velocity, axis=0) / dvt[:, None]
    acceleration_rms = (
        float(np.sqrt(np.mean(np.sum(acceleration**2, axis=1))))
        if len(acceleration)
        else None
    )

    acceleration_times = (velocity_times[:-1] + velocity_times[1:]) / 2.0
    dat = np.diff(acceleration_times)
    jerk = np.diff(acceleration, axis=0) / dat[:, None]
    jerk_rms = (
        float(np.sqrt(np.mean(np.sum(jerk**2, axis=1)))) if len(jerk) else None
    )

    norms = np.linalg.norm(displacement, axis=1)
    valid_angles: list[float] = []
    for idx in range(1, len(displacement)):
        denom = norms[idx - 1] * norms[idx]
        if denom <= 0:
            continue
        cosine_value = float(
            np.clip(np.dot(displacement[idx - 1], displacement[idx]) / denom, -1.0, 1.0)
        )
        valid_angles.append(math.degrees(math.acos(cosine_value)))
    direction_change_rate = sum(valid_angles) / duration if duration and valid_angles else 0.0
    return {
        "higher_eligible": True,
        "speed_variability_m_s": speed_variability,
        "acceleration_rms_m_s2": acceleration_rms,
        "jerk_rms_m_s3": jerk_rms,
        "direction_change_rate_deg_s": direction_change_rate,
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return None
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def episode_span(episodes: list[dict[str, Any]]) -> float | None:
    starts = [parse_time(item.get("startTime")) for item in episodes]
    ends = [parse_time(item.get("endTime")) for item in episodes]
    starts = [item for item in starts if item is not None]
    ends = [item for item in ends if item is not None]
    return (max(ends) - min(starts)).total_seconds() if starts and ends else None


def distinct_attempt_count(episodes: list[dict[str, Any]]) -> int:
    """Count temporally distinct grasp attempts, merging overlapping hand records."""
    intervals: list[tuple[datetime, datetime]] = []
    for item in episodes:
        start = parse_time(item.get("startTime"))
        end = parse_time(item.get("endTime"))
        if start is not None and end is not None and end >= start:
            intervals.append((start, end))
    if not intervals:
        return 0
    intervals.sort(key=lambda interval: interval[0])
    count = 1
    _, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            count += 1
            current_end = end
    return count


def action_span(actions: list[dict[str, Any]]) -> float | None:
    return episode_span(actions)


def session_geometry(p2: dict[str, Any]) -> dict[str, float | None]:
    left = p2.get("pushLeftZone") or {}
    right = p2.get("pushRightZone") or {}
    lc = pos(left.get("tableLocalCenter"))
    rc = pos(right.get("tableLocalCenter"))
    return {
        "cue_width_m": float(left.get("width")) if left.get("width") is not None else None,
        "cue_length_m": float(left.get("length")) if left.get("length") is not None else None,
        "cue_center_separation_m": float(np.linalg.norm(rc - lc))
        if lc is not None and rc is not None
        else None,
    }


def session_task_axes(p2: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return validated unit forward and lateral axes from the logged cue geometry."""
    left = p2.get("pushLeftZone") or {}
    right = p2.get("pushRightZone") or {}
    forward_left = pos(left.get("tableLocalForward"))
    forward_right = pos(right.get("tableLocalForward"))
    lateral_left = pos(left.get("tableLocalRight"))
    lateral_right = pos(right.get("tableLocalRight"))
    vectors = (forward_left, forward_right, lateral_left, lateral_right)
    if any(vector is None or np.linalg.norm(vector) <= 0 for vector in vectors):
        raise ValueError("Missing or zero-length task axis in P2 guidance-zone metadata")
    forward_left = forward_left / np.linalg.norm(forward_left)
    forward_right = forward_right / np.linalg.norm(forward_right)
    lateral_left = lateral_left / np.linalg.norm(lateral_left)
    lateral_right = lateral_right / np.linalg.norm(lateral_right)
    if not np.allclose(forward_left, forward_right, atol=1e-6):
        raise ValueError("Left and right guidance zones use different forward axes")
    if not np.allclose(lateral_left, lateral_right, atol=1e-6):
        raise ValueError("Left and right guidance zones use different lateral axes")
    if not math.isclose(float(np.dot(forward_left, lateral_left)), 0.0, abs_tol=1e-6):
        raise ValueError("Logged forward and lateral task axes are not orthogonal")
    return forward_left, lateral_left


def time_normalized_lateral_profiles(
    n_grid: int = 101,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Build bilateral-midpoint lateral profiles for visualization only.

    Each eligible bilateral action is linearly normalized to 0–100% action
    time. Lateral displacement is measured along the recorded table-local
    right axis and expressed relative to the action's first paired sample.
    The operation uses every action eligible for the basic bilateral analysis;
    no representative action is selected.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    profiles: dict[str, dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []

    for session, folder, _ in SESSION_SPECS:
        p2 = read_json(session_source(folder) / "phase2_push.json")
        zone = p2.get("pushLeftZone") or {}
        lateral_axis = pos(zone.get("tableLocalRight"))
        if lateral_axis is None or np.linalg.norm(lateral_axis) <= 0:
            raise ValueError(f"Missing table-local lateral axis for {session}")
        lateral_axis = lateral_axis / np.linalg.norm(lateral_axis)

        traces: list[np.ndarray] = []
        action_indices: list[int] = []
        raw_peak_to_peak_mm: list[float] = []
        for action_index, action in enumerate(p2.get("pushCycles") or [], start=1):
            if (
                bool(action.get("isSingleHand"))
                or not bool(action.get("usedLeft"))
                or not bool(action.get("usedRight"))
            ):
                continue

            paired: list[tuple[datetime, np.ndarray, np.ndarray]] = []
            for sample in action.get("samples") or []:
                if not (sample.get("hasLeft") and sample.get("hasRight")):
                    continue
                timestamp = parse_time(sample.get("time"))
                left = pos(sample.get("leftTableLocalPosition"))
                right = pos(sample.get("rightTableLocalPosition"))
                if timestamp is not None and left is not None and right is not None:
                    paired.append((timestamp, left, right))

            if len(paired) < BASIC_MIN_SAMPLES:
                continue
            paired.sort(key=lambda item: item[0])
            times = np.asarray(
                [(item[0] - paired[0][0]).total_seconds() for item in paired],
                dtype=float,
            )
            left_points = np.vstack([item[1] for item in paired])
            right_points = np.vstack([item[2] for item in paired])
            unique = np.r_[True, np.diff(times) > 0]
            times = times[unique]
            left_points = left_points[unique]
            right_points = right_points[unique]
            if len(times) < BASIC_MIN_SAMPLES or times[-1] <= 0:
                continue
            left_path = float(np.linalg.norm(np.diff(left_points, axis=0), axis=1).sum())
            right_path = float(np.linalg.norm(np.diff(right_points, axis=0), axis=1).sum())
            if left_path <= 0 or right_path <= 0:
                continue

            midpoint = (left_points + right_points) / 2.0
            lateral = midpoint @ lateral_axis
            normalized_time = times / times[-1]
            raw_lateral_mm = (lateral - lateral[0]) * 1000.0
            trace_mm = np.interp(grid, normalized_time, raw_lateral_mm)
            traces.append(trace_mm)
            action_indices.append(action_index)
            raw_peak_to_peak_mm.append(float(np.ptp(raw_lateral_mm)))
            for normalized, displacement in zip(grid, trace_mm):
                source_rows.append(
                    {
                        "session": session,
                        "action_index": action_index,
                        "normalized_action_time_percent": 100.0 * normalized,
                        "bilateral_midpoint_lateral_displacement_mm": displacement,
                    }
                )

        array = np.vstack(traces)
        profiles[session] = {
            "grid": grid,
            "traces": array,
            "action_indices": action_indices,
            "raw_peak_to_peak_mm": np.asarray(raw_peak_to_peak_mm, dtype=float),
            "median": np.median(array, axis=0),
            "q1": np.quantile(array, 0.25, axis=0),
            "q3": np.quantile(array, 0.75, axis=0),
        }

    return profiles, source_rows


def analyze_transfer_phase(
    phase: dict[str, Any], session: str, phase_label: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episodes = phase.get("grabSessions") or []
    rows: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes, start=1):
        times, points = trajectory_from_samples(
            episode.get("samples") or [], "tableLocalPosition"
        )
        metrics = basic_trajectory_metrics(times, points)
        rows.append(
            {
                "session": session,
                "phase": phase_label,
                "episode_index": index,
                "hand": episode.get("hand"),
                "episode_duration_s": elapsed(episode.get("startTime"), episode.get("endTime")),
                **metrics,
            }
        )
    efficiencies = [row["path_efficiency"] for row in rows if row["path_efficiency"] is not None]
    summary = {
        f"{phase_label}_recorded_activity_span_s": episode_span(episodes),
        f"{phase_label}_recorded_grasp_episodes": len(episodes),
        f"{phase_label}_distinct_transfer_attempts": distinct_attempt_count(episodes),
        f"{phase_label}_eligible_grasp_episodes": len(efficiencies),
        f"{phase_label}_path_efficiency_mean": mean(efficiencies),
        f"{phase_label}_path_efficiency_median": q(efficiencies, 0.5),
    }
    return summary, rows


def analyze_action(
    action: dict[str, Any],
    action_index: int,
    session: str,
    cue_separation: float,
    forward_axis: np.ndarray,
    lateral_axis: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = action.get("samples") or []
    paired = [
        sample
        for sample in samples
        if sample.get("hasLeft")
        and sample.get("hasRight")
        and pos(sample.get("leftTableLocalPosition")) is not None
        and pos(sample.get("rightTableLocalPosition")) is not None
        and parse_time(sample.get("time")) is not None
    ]
    paired.sort(key=lambda sample: parse_time(sample.get("time")))
    left_times, left_points = trajectory_from_samples(
        paired, "leftTableLocalPosition", "hasLeft"
    )
    right_times, right_points = trajectory_from_samples(
        paired, "rightTableLocalPosition", "hasRight"
    )
    left_basic = basic_trajectory_metrics(left_times, left_points)
    right_basic = basic_trajectory_metrics(right_times, right_points)
    left_higher = higher_order_metrics(left_times, left_points)
    right_higher = higher_order_metrics(right_times, right_points)

    bilateral_classified = bool(
        action.get("usedLeft")
        and action.get("usedRight")
        and not action.get("isSingleHand")
    )
    basic_bilateral_eligible = bool(
        bilateral_classified
        and len(paired) >= BASIC_MIN_SAMPLES
        and left_basic["basic_eligible"]
        and right_basic["basic_eligible"]
    )

    direction_cosine = None
    path_difference = None
    left_signed_forward_displacement = None
    right_signed_forward_displacement = None
    mean_bilateral_signed_forward_displacement = None
    left_lateral_excursion = None
    right_lateral_excursion = None
    mean_bilateral_lateral_excursion = None
    separations: list[float] = []
    separation_errors: list[float] = []
    left_in: list[bool] = []
    right_in: list[bool] = []
    both_in: list[bool] = []
    if basic_bilateral_eligible:
        direction_cosine = cosine_similarity(
            left_points[-1] - left_points[0], right_points[-1] - right_points[0]
        )
        path_difference = abs(left_basic["path_m"] - right_basic["path_m"])
        left_signed_forward_displacement = float(
            np.dot(left_points[-1] - left_points[0], forward_axis)
        )
        right_signed_forward_displacement = float(
            np.dot(right_points[-1] - right_points[0], forward_axis)
        )
        mean_bilateral_signed_forward_displacement = mean(
            [left_signed_forward_displacement, right_signed_forward_displacement]
        )
        left_lateral_projection = left_points @ lateral_axis
        right_lateral_projection = right_points @ lateral_axis
        left_lateral_excursion = float(np.ptp(left_lateral_projection))
        right_lateral_excursion = float(np.ptp(right_lateral_projection))
        mean_bilateral_lateral_excursion = mean(
            [left_lateral_excursion, right_lateral_excursion]
        )
        for sample in paired:
            lp = pos(sample.get("leftTableLocalPosition"))
            rp = pos(sample.get("rightTableLocalPosition"))
            if lp is None or rp is None:
                continue
            separation = float(np.linalg.norm(rp - lp))
            separations.append(separation)
            separation_errors.append(abs(separation - cue_separation))
            left_flag = bool(sample.get("leftInLeftZone"))
            right_flag = bool(sample.get("rightInRightZone"))
            left_in.append(left_flag)
            right_in.append(right_flag)
            both_in.append(left_flag and right_flag)

    duration_s = elapsed(action.get("startTime"), action.get("endTime"))
    mean_bilateral_path = (
        mean([left_basic["path_m"], right_basic["path_m"]])
        if basic_bilateral_eligible
        else None
    )
    mean_bilateral_net = (
        mean([left_basic["net_m"], right_basic["net_m"]])
        if basic_bilateral_eligible
        else None
    )
    mean_bilateral_speed = (
        safe_div(mean_bilateral_path, duration_s) if basic_bilateral_eligible else None
    )

    action_row = {
        "session": session,
        "action_index": action_index,
        # Timestamp strings remain private working fields so that adjacent-action
        # gaps and event linkage can be calculated without exporting clock time.
        "_start_time": action.get("startTime"),
        "_end_time": action.get("endTime"),
        "duration_s": duration_s,
        "recorded_action": True,
        "bilateral_classified": bilateral_classified,
        "single_hand_classified": not bilateral_classified,
        "used_left": bool(action.get("usedLeft")),
        "used_right": bool(action.get("usedRight")),
        "is_single_hand_field": bool(action.get("isSingleHand")),
        "n_samples": len(samples),
        "n_paired_samples": len(paired),
        "basic_bilateral_eligible": basic_bilateral_eligible,
        "directional_consistency_cosine": direction_cosine,
        "left_right_path_length_difference_m": path_difference,
        "left_signed_forward_displacement_m": left_signed_forward_displacement,
        "right_signed_forward_displacement_m": right_signed_forward_displacement,
        "mean_bilateral_signed_forward_displacement_m": mean_bilateral_signed_forward_displacement,
        "left_lateral_excursion_m": left_lateral_excursion,
        "right_lateral_excursion_m": right_lateral_excursion,
        "mean_bilateral_lateral_excursion_m": mean_bilateral_lateral_excursion,
        "mean_hand_separation_m": mean(separations),
        "mean_abs_hand_separation_error_m": mean(separation_errors),
        "paired_quality_sample_count": len(separations),
        "hand_separation_sum_m": sum(separations),
        "hand_separation_error_sum_m": sum(separation_errors),
        "left_guidance_adherence": mean([float(v) for v in left_in]),
        "right_guidance_adherence": mean([float(v) for v in right_in]),
        "bilateral_guidance_adherence": mean([float(v) for v in both_in]),
        "bilateral_in_zone_sample_count": sum(both_in),
        "mean_bilateral_path_length_m": mean_bilateral_path,
        "mean_bilateral_net_displacement_m": mean_bilateral_net,
        "mean_bilateral_path_speed_m_s": mean_bilateral_speed,
    }

    hand_rows: list[dict[str, Any]] = []
    for hand, basic, higher, signed_forward, lateral_excursion in (
        (
            "Left",
            left_basic,
            left_higher,
            left_signed_forward_displacement,
            left_lateral_excursion,
        ),
        (
            "Right",
            right_basic,
            right_higher,
            right_signed_forward_displacement,
            right_lateral_excursion,
        ),
    ):
        hand_rows.append(
            {
                "session": session,
                "action_index": action_index,
                "hand": hand,
                "bilateral_classified": bilateral_classified,
                "basic_bilateral_eligible": basic_bilateral_eligible,
                "n_samples": basic["n"],
                "sampled_duration_s": basic["duration_s"],
                "path_length_m": basic["path_m"] if basic_bilateral_eligible else None,
                "net_displacement_m": basic["net_m"] if basic_bilateral_eligible else None,
                "path_efficiency": basic["path_efficiency"] if basic_bilateral_eligible else None,
                "mean_path_speed_m_s": basic["mean_speed_m_s"] if basic_bilateral_eligible else None,
                "signed_forward_displacement_m": signed_forward,
                "lateral_excursion_m": lateral_excursion,
                "guidance_adherence": (
                    action_row["left_guidance_adherence"]
                    if hand == "Left"
                    else action_row["right_guidance_adherence"]
                ),
                "higher_order_eligible": bool(
                    bilateral_classified and higher["higher_eligible"]
                ),
                "speed_variability_m_s": higher["speed_variability_m_s"]
                if bilateral_classified
                else None,
                "acceleration_rms_m_s2": higher["acceleration_rms_m_s2"]
                if bilateral_classified
                else None,
                "jerk_rms_m_s3": higher["jerk_rms_m_s3"]
                if bilateral_classified
                else None,
                "direction_change_rate_deg_s": higher["direction_change_rate_deg_s"]
                if bilateral_classified
                else None,
            }
        )
    return action_row, hand_rows


def analyze_session(label: str, folder: str | Path, profile: str) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    source = session_source(folder)
    p1 = read_json(source / "phase1_grab.json")
    p2 = read_json(source / "phase2_push.json")
    p3 = read_json(source / "phase3_return.json")
    meta = read_json(source / "session_meta.json")
    ids = {p1.get("sessionId"), p2.get("sessionId"), p3.get("sessionId"), meta.get("sessionId")}
    geometry = session_geometry(p2)
    forward_axis, lateral_axis = session_task_axes(p2)
    p1_summary, p1_rows = analyze_transfer_phase(p1, label, "p1")
    p3_summary, p3_rows = analyze_transfer_phase(p3, label, "p3")
    actions = p2.get("pushCycles") or []
    action_rows: list[dict[str, Any]] = []
    hand_rows: list[dict[str, Any]] = []
    for index, action in enumerate(actions, start=1):
        row, hands = analyze_action(
            action,
            index,
            label,
            float(geometry["cue_center_separation_m"]),
            forward_axis,
            lateral_axis,
        )
        action_rows.append(row)
        hand_rows.extend(hands)

    action_rows.sort(key=lambda row: parse_time(row["_start_time"]))
    bilateral_rows = [row for row in action_rows if row["bilateral_classified"]]
    gaps: list[float] = []
    for previous, current in zip(action_rows[:-1], action_rows[1:]):
        # Do not bridge across a single-hand-classified event: only an
        # originally adjacent bilateral–bilateral pair defines a valid gap.
        if not (previous["bilateral_classified"] and current["bilateral_classified"]):
            continue
        gap = elapsed(previous["_end_time"], current["_start_time"])
        if gap is not None:
            gaps.append(gap)

    basic_rows = [row for row in action_rows if row["basic_bilateral_eligible"]]
    basic_hands = [
        row for row in hand_rows if row["basic_bilateral_eligible"] and row["path_efficiency"] is not None
    ]
    higher_hands = [row for row in hand_rows if row["higher_order_eligible"]]

    duration_values = [row["duration_s"] for row in bilateral_rows]
    direction_values = [row["directional_consistency_cosine"] for row in basic_rows]
    path_diff_values = [row["left_right_path_length_difference_m"] for row in basic_rows]
    separation_values = [row["mean_hand_separation_m"] for row in basic_rows]
    separation_error_values = [row["mean_abs_hand_separation_error_m"] for row in basic_rows]
    adherence_values = [row["bilateral_guidance_adherence"] for row in basic_rows]
    mean_path_values = [row["mean_bilateral_path_length_m"] for row in basic_rows]
    mean_net_values = [row["mean_bilateral_net_displacement_m"] for row in basic_rows]
    mean_speed_values = [row["mean_bilateral_path_speed_m_s"] for row in basic_rows]
    path_eff_values = [row["path_efficiency"] for row in basic_hands]
    signed_forward_values = [
        row["mean_bilateral_signed_forward_displacement_m"] for row in basic_rows
    ]
    lateral_excursion_values = [
        row["mean_bilateral_lateral_excursion_m"] for row in basic_rows
    ]
    signed_forward_summary = summarize(signed_forward_values)
    lateral_excursion_summary = summarize(lateral_excursion_values)
    signed_forward_crosses_zero = bool(
        any(value < 0 for value in signed_forward_values)
        and any(value > 0 for value in signed_forward_values)
    )
    signed_forward_cv_interpretable = bool(
        signed_forward_summary["cv"] is not None
        and signed_forward_summary["mean"] is not None
        and signed_forward_summary["mean"] > 0
        and not signed_forward_crosses_zero
    )
    lateral_excursion_cv_interpretable = bool(
        lateral_excursion_summary["cv"] is not None
        and lateral_excursion_summary["mean"] is not None
        and lateral_excursion_summary["mean"] > 0
    )

    needle_events = p2.get("needleSafetyEvents") or []
    linked = 0
    linked_bilateral = 0
    for event in needle_events:
        event_time = parse_time(event.get("time"))
        if event_time is None:
            continue
        for row in action_rows:
            start = parse_time(row["_start_time"])
            end = parse_time(row["_end_time"])
            if start and end and start <= event_time <= end:
                linked += 1
                if row["bilateral_classified"]:
                    linked_bilateral += 1
                break

    p2_span = action_span(actions)
    summary = {
        "session": label,
        "profile": profile,
        "session_id_consistent": len(ids) == 1,
        "completed": str(meta.get("endReason", "")).casefold() == "completed"
        and not meta.get("isPartialSession"),
        "is_simulated": bool(meta.get("isSimulated")),
        "p2_exit_reason": p2.get("exitReason"),
        "generated_phase_files": 3,
        "generated_metadata_files": 1,
        **geometry,
        **p1_summary,
        **p3_summary,
        "p2_recorded_activity_span_s": p2_span,
        "p2_recorded_action_count": len(action_rows),
        "p2_bilateral_classified_count": len(bilateral_rows),
        "p2_single_hand_classified_count": len(action_rows) - len(bilateral_rows),
        "p2_bilateral_participation_rate": safe_div(len(bilateral_rows), len(action_rows)),
        "p2_basic_bilateral_action_count": len(basic_rows),
        "p2_higher_order_action_count": len({row["action_index"] for row in higher_hands}),
        "p2_higher_order_hand_action_count": len(higher_hands),
        "p2_recorded_frequency_per_min": safe_div(len(action_rows) * 60.0, p2_span),
        "p2_bilateral_frequency_per_min": safe_div(len(bilateral_rows) * 60.0, p2_span),
        "p2_action_duration_n": summarize(duration_values)["n"],
        "p2_action_duration_mean_s": summarize(duration_values)["mean"],
        "p2_action_duration_sd_s": summarize(duration_values)["sd"],
        "p2_action_duration_cv": summarize(duration_values)["cv"],
        "p2_action_duration_q1_s": summarize(duration_values)["q1"],
        "p2_action_duration_median_s": summarize(duration_values)["median"],
        "p2_action_duration_q3_s": summarize(duration_values)["q3"],
        "p2_inter_action_gap_n": summarize(gaps)["n"],
        "p2_inter_action_gap_mean_s": summarize(gaps)["mean"],
        "p2_inter_action_gap_sd_s": summarize(gaps)["sd"],
        "p2_inter_action_gap_cv": summarize(gaps)["cv"],
        "p2_inter_action_gap_q1_s": summarize(gaps)["q1"],
        "p2_inter_action_gap_median_s": summarize(gaps)["median"],
        "p2_inter_action_gap_q3_s": summarize(gaps)["q3"],
        "p2_path_efficiency_n_hand_actions": summarize(path_eff_values)["n"],
        "p2_path_efficiency_median": summarize(path_eff_values)["median"],
        "p2_path_efficiency_q1": summarize(path_eff_values)["q1"],
        "p2_path_efficiency_q3": summarize(path_eff_values)["q3"],
        "p2_directional_consistency_n_actions": summarize(direction_values)["n"],
        "p2_directional_consistency_mean": summarize(direction_values)["mean"],
        "p2_directional_consistency_sd": summarize(direction_values)["sd"],
        "p2_directional_consistency_q1": summarize(direction_values)["q1"],
        "p2_directional_consistency_median": summarize(direction_values)["median"],
        "p2_directional_consistency_q3": summarize(direction_values)["q3"],
        "p2_directional_consistency_negative_actions": sum(v < 0 for v in direction_values if v is not None),
        "p2_path_length_difference_mean_m": summarize(path_diff_values)["mean"],
        "p2_path_length_difference_median_m": summarize(path_diff_values)["median"],
        "p2_mean_bilateral_signed_forward_displacement_n_actions": signed_forward_summary["n"],
        "p2_mean_bilateral_signed_forward_displacement_mean_m": signed_forward_summary["mean"],
        "p2_mean_bilateral_signed_forward_displacement_sd_m": signed_forward_summary["sd"],
        "p2_mean_bilateral_signed_forward_displacement_cv": signed_forward_summary["cv"],
        "p2_mean_bilateral_signed_forward_displacement_cv_interpretable": signed_forward_cv_interpretable,
        "p2_mean_bilateral_signed_forward_displacement_q1_m": signed_forward_summary["q1"],
        "p2_mean_bilateral_signed_forward_displacement_median_m": signed_forward_summary["median"],
        "p2_mean_bilateral_signed_forward_displacement_q3_m": signed_forward_summary["q3"],
        "p2_mean_bilateral_signed_forward_displacement_negative_actions": sum(
            value < 0 for value in signed_forward_values
        ),
        "p2_mean_bilateral_lateral_excursion_n_actions": lateral_excursion_summary["n"],
        "p2_mean_bilateral_lateral_excursion_mean_m": lateral_excursion_summary["mean"],
        "p2_mean_bilateral_lateral_excursion_sd_m": lateral_excursion_summary["sd"],
        "p2_mean_bilateral_lateral_excursion_cv": lateral_excursion_summary["cv"],
        "p2_mean_bilateral_lateral_excursion_cv_interpretable": lateral_excursion_cv_interpretable,
        "p2_mean_bilateral_lateral_excursion_q1_m": lateral_excursion_summary["q1"],
        "p2_mean_bilateral_lateral_excursion_median_m": lateral_excursion_summary["median"],
        "p2_mean_bilateral_lateral_excursion_q3_m": lateral_excursion_summary["q3"],
        "p2_mean_hand_separation_mean_m": summarize(separation_values)["mean"],
        "p2_mean_hand_separation_sd_m": summarize(separation_values)["sd"],
        "p2_mean_hand_separation_cv": summarize(separation_values)["cv"],
        "p2_separation_error_mean_action_m": summarize(separation_error_values)["mean"],
        "p2_separation_error_median_action_m": summarize(separation_error_values)["median"],
        "p2_separation_error_q1_action_m": summarize(separation_error_values)["q1"],
        "p2_separation_error_q3_action_m": summarize(separation_error_values)["q3"],
        "p2_mean_hand_separation_pooled_m": safe_div(
            sum(row["hand_separation_sum_m"] for row in basic_rows),
            sum(row["paired_quality_sample_count"] for row in basic_rows),
        ),
        "p2_separation_error_pooled_m": safe_div(
            sum(row["hand_separation_error_sum_m"] for row in basic_rows),
            sum(row["paired_quality_sample_count"] for row in basic_rows),
        ),
        "p2_bilateral_adherence_mean_action": summarize(adherence_values)["mean"],
        "p2_bilateral_adherence_median_action": summarize(adherence_values)["median"],
        "p2_bilateral_adherence_q1_action": summarize(adherence_values)["q1"],
        "p2_bilateral_adherence_q3_action": summarize(adherence_values)["q3"],
        "p2_bilateral_adherence_pooled": safe_div(
            sum(row["bilateral_in_zone_sample_count"] for row in basic_rows),
            sum(row["paired_quality_sample_count"] for row in basic_rows),
        ),
        "p2_mean_bilateral_net_displacement_sd_m": summarize(mean_net_values)["sd"],
        "p2_mean_bilateral_net_displacement_cv": summarize(mean_net_values)["cv"],
        "p2_mean_bilateral_path_length_sd_m": summarize(mean_path_values)["sd"],
        "p2_mean_bilateral_path_length_cv": summarize(mean_path_values)["cv"],
        "p2_mean_bilateral_speed_sd_m_s": summarize(mean_speed_values)["sd"],
        "p2_mean_bilateral_speed_cv": summarize(mean_speed_values)["cv"],
        "p2_needle_events_raw": len(needle_events),
        "p2_needle_events_action_linked": linked,
        "p2_needle_events_outside_actions": len(needle_events) - linked,
        "p2_needle_events_bilateral_action_linked": linked_bilateral,
        "p2_needle_events_nonbilateral_or_outside": len(needle_events) - linked_bilateral,
    }
    for key in (
        "speed_variability_m_s",
        "acceleration_rms_m_s2",
        "jerk_rms_m_s3",
        "direction_change_rate_deg_s",
    ):
        values = [row[key] for row in higher_hands]
        summary[f"p2_{key}_mean"] = mean(values)
        summary[f"p2_{key}_median"] = q(values, 0.5)
        summary[f"p2_{key}_q1"] = q(values, 0.25)
        summary[f"p2_{key}_q3"] = q(values, 0.75)

    qa = {
        "session": label,
        "four_session_ids_consistent": len(ids) == 1,
        "p2_push_count_field": p2.get("pushCycleCount"),
        "p2_push_list_count": len(actions),
        "p2_all_samples_paired": all(
            row["n_samples"] == row["n_paired_samples"] for row in action_rows
        ),
        "higher_order_rule": {
            "minimum_samples": HIGHER_MIN_SAMPLES,
        },
        "task_axes": {
            "coordinate_space": p2.get("coordinateSpace"),
            "forward_source_fields": [
                "pushLeftZone.tableLocalForward",
                "pushRightZone.tableLocalForward",
            ],
            "lateral_source_fields": [
                "pushLeftZone.tableLocalRight",
                "pushRightZone.tableLocalRight",
            ],
            "forward_unit_vector": [float(value) for value in forward_axis],
            "lateral_unit_vector": [float(value) for value in lateral_axis],
            "left_right_zone_axes_matched": True,
            "forward_lateral_dot_product": float(
                np.dot(forward_axis, lateral_axis)
            ),
        },
        "axis_projected_action_metric_definitions": {
            "population": "basic bilateral eligible actions",
            "signed_forward_displacement": "Endpoint minus start position projected onto the logged unit table-local forward axis; positive values follow tableLocalForward. The bilateral action value is the arithmetic mean of the left- and right-hand signed projections.",
            "lateral_excursion": "Maximum minus minimum raw-sample position projected onto the logged unit table-local right axis. The bilateral action value is the arithmetic mean of the left- and right-hand excursions.",
            "session_stability": "Session summaries are calculated across per-action bilateral means; sample SD uses ddof=1 and CV is sample SD divided by the signed arithmetic mean.",
            "signed_forward_crosses_zero": signed_forward_crosses_zero,
            "signed_forward_cv_interpretable": signed_forward_cv_interpretable,
            "lateral_excursion_cv_interpretable": lateral_excursion_cv_interpretable,
        },
    }
    return summary, action_rows, hand_rows, p1_rows + p3_rows, qa


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if not key.startswith("_") and key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_assignment(value: str, option_name: str) -> tuple[str, str]:
    """Parse ``LABEL=VALUE`` used by the repeated command-line options."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"{option_name} must use LABEL=VALUE syntax; received {value!r}"
        )
    label, assigned = (part.strip() for part in value.split("=", 1))
    if not label or not assigned:
        raise argparse.ArgumentTypeError(
            f"{option_name} requires a non-empty label and value"
        )
    return label, assigned


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive the four VR-WCE outcome domains from four pseudonymized "
            "technical-session directories."
        )
    )
    parser.add_argument(
        "--session",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Session mapping; provide exactly one each for S1, S2, U1 and U2. "
            "Each directory must contain the four JSON files listed in README.md."
        ),
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        metavar="LABEL=TEXT",
        help="Optional display-profile override, for example S2=Program-generated profile 2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV, JSON and figure outputs (default: ./results).",
    )
    parser.add_argument(
        "--figure-stem",
        default="Fig5_VRWCE_technical_sessions",
        help="Filename stem for the main four-domain figure.",
    )
    parser.add_argument(
        "--supplementary-stem",
        default="VRWCE_higher_order_stability",
        help="Filename stem for the detailed stability figure.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Write numerical outputs without rendering figures.",
    )
    return parser.parse_args(argv)


def configure_analysis(args: argparse.Namespace) -> None:
    """Validate pseudonymous session mappings and initialize output settings."""
    global SESSION_SPECS, OUT, FIGURE_STEM, SUPPLEMENTARY_STEM

    required_labels = ("S1", "S2", "U1", "U2")
    session_map: dict[str, Path] = {}
    for raw in args.session:
        label, value = parse_assignment(raw, "--session")
        if label in session_map:
            raise ValueError(f"Duplicate --session mapping for {label}")
        session_map[label] = Path(value).expanduser().resolve()
    if set(session_map) != set(required_labels):
        raise ValueError(
            "--session mappings must contain exactly S1, S2, U1 and U2; "
            f"received {sorted(session_map)}"
        )

    profiles = dict(DEFAULT_PROFILES)
    for raw in args.profile:
        label, text_value = parse_assignment(raw, "--profile")
        if label not in required_labels:
            raise ValueError(f"Unknown profile label {label!r}")
        profiles[label] = text_value

    required_files = (
        "phase1_grab.json",
        "phase2_push.json",
        "phase3_return.json",
        "session_meta.json",
    )
    for label in required_labels:
        directory = session_map[label]
        missing = [name for name in required_files if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{label} is missing required files in {directory}: {', '.join(missing)}"
            )

    SESSION_SPECS = [
        (label, session_map[label], profiles[label]) for label in required_labels
    ]
    OUT = args.output_dir.expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURE_STEM = args.figure_stem
    SUPPLEMENTARY_STEM = args.supplementary_stem


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    configure_analysis(args)
    summaries: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    hands: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    qa: list[dict[str, Any]] = []
    for label, folder, profile in SESSION_SPECS:
        summary, action_rows, hand_rows, transfer_rows, session_qa = analyze_session(
            label, folder, profile
        )
        summaries.append(summary)
        actions.extend(action_rows)
        hands.extend(hand_rows)
        transfers.extend(transfer_rows)
        qa.append(session_qa)

    lateral_profiles, lateral_rows = time_normalized_lateral_profiles()
    lateral_summary_rows: list[dict[str, Any]] = []
    for session, _, _ in SESSION_SPECS:
        traces = lateral_profiles[session]["traces"]
        peak_to_peak = lateral_profiles[session]["raw_peak_to_peak_mm"]
        lateral_summary_rows.append(
            {
                "session": session,
                "eligible_bilateral_actions": len(traces),
                "raw_action_peak_to_peak_lateral_displacement_q1_mm": q(peak_to_peak, 0.25),
                "raw_action_peak_to_peak_lateral_displacement_median_mm": q(peak_to_peak, 0.50),
                "raw_action_peak_to_peak_lateral_displacement_q3_mm": q(peak_to_peak, 0.75),
                "median_profile_peak_to_peak_lateral_displacement_mm": float(
                    np.ptp(lateral_profiles[session]["median"])
                ),
            }
        )

    write_csv(OUT / "session_summary.csv", summaries)
    write_csv(OUT / "p2_action_metrics.csv", actions)
    write_csv(OUT / "p2_hand_action_metrics.csv", hands)
    write_csv(OUT / "transfer_episode_metrics.csv", transfers)
    write_csv(OUT / "p2_time_normalized_lateral_profiles.csv", lateral_rows)
    write_csv(OUT / "p2_lateral_profile_summary.csv", lateral_summary_rows)
    with (OUT / "analysis_qa.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "analysis_populations": {
                    "workload": "all runtime-recorded actions; bilateral participation reported separately",
                    "bilateral_timing": "bilateral-classified action durations; inter-action gaps only for originally adjacent action pairs in which both actions were bilateral-classified, without bridging across an excluded single-hand action",
                    "basic_bilateral_trajectory": f"bilateral-classified actions with at least {BASIC_MIN_SAMPLES} synchronized samples and non-zero paths",
                    "time_normalized_lateral_visualization": "usedLeft and usedRight, not single-hand-classified, at least 3 unique paired samples, positive sampled duration, and non-zero paths for both hands; bilateral-midpoint displacement along table-local right; linear interpolation to 101 points from 0% to 100% action time; visualization only",
                    "higher_order": f"bilateral hand-action trajectories with at least {HIGHER_MIN_SAMPLES} synchronized samples; finite-difference derivatives were not smoothed",
                },
                "sessions": qa,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    if not args.skip_figures:
        # Keep Matplotlib and the optional presentation layer out of all
        # numerical-only imports and --skip-figures runs.
        from plot_vrwce_outputs import render_outputs

        render_outputs(
            summaries,
            actions,
            hands,
            lateral_profiles,
            output_dir=OUT,
            figure_stem=FIGURE_STEM,
            supplementary_stem=SUPPLEMENTARY_STEM,
            session_specs=SESSION_SPECS,
            parse_time_fn=parse_time,
            elapsed_fn=elapsed,
            cv_fn=cv,
        )

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
