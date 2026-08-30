#!/usr/bin/env python3
"""Cycle-balanced phase-specific wrist-use regions for the retained cohort.

This public script starts from a de-identified, phase-labelled trajectory table.
The upstream decisions that reduce 220 candidate intervals to the 210-cycle
retained cohort are inputs, not recreated here.  See README.md for the input
contract and the boundary of this release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


PHASES = ("P1", "P2", "P3")
HANDS = ("left", "right")
CELL_SIZE_SW = 0.025
BANDWIDTH_CANDIDATES_SW = (0.025, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18)
MIN_VALID_SAMPLES = 10
MIN_PHASE_AVAILABILITY = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate equally cycle-weighted 50% and 90% wrist HDRs."
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="Phase-labelled, needle-centred wrist trajectories in SW units.",
    )
    parser.add_argument(
        "--cycle-audit",
        type=Path,
        required=True,
        help="One-row-per-cycle phase completeness table.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cycles", type=int, default=210)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_inputs(
    trajectory_path: Path, audit_path: Path, expected_cycles: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(trajectory_path)
    audit = pd.read_csv(audit_path)
    require_columns(
        data,
        {
            "cycle_key",
            "video_id",
            "work_cycle_id",
            "phase",
            "hand",
            "x_from_needle_sw",
            "y_from_needle_sw",
        },
        "trajectory table",
    )
    require_columns(audit, {"cycle_key"}, "cycle audit")
    for phase in PHASES:
        require_columns(
            audit,
            {
                f"{phase}_retained_frames",
                f"{phase}_left_valid_frames",
                f"{phase}_right_valid_frames",
            },
            "cycle audit",
        )

    data["cycle_key"] = data["cycle_key"].astype(str)
    audit["cycle_key"] = audit["cycle_key"].astype(str)
    if audit.cycle_key.duplicated().any():
        raise ValueError("cycle audit must contain one row per retained cycle")
    if len(audit) != expected_cycles or data.cycle_key.nunique() != expected_cycles:
        raise ValueError(
            f"Expected {expected_cycles} retained cycles; found "
            f"{len(audit)} audit rows and {data.cycle_key.nunique()} trajectory cycles"
        )
    if not set(data.phase.dropna().unique()).issubset(PHASES):
        raise ValueError("phase must contain only P1, P2, and P3")
    if not set(data.hand.dropna().unique()).issubset(HANDS):
        raise ValueError("hand must contain only left and right")
    if not np.isfinite(data[["x_from_needle_sw", "y_from_needle_sw"]]).all().all():
        raise ValueError("trajectory coordinates must be finite")
    return data, audit


def primary_eligible_cycles(audit: pd.DataFrame) -> dict[str, pd.Index]:
    eligible: dict[str, pd.Index] = {}
    for phase in PHASES:
        retained = pd.to_numeric(audit[f"{phase}_retained_frames"], errors="raise")
        left = pd.to_numeric(audit[f"{phase}_left_valid_frames"], errors="raise")
        right = pd.to_numeric(audit[f"{phase}_right_valid_frames"], errors="raise")
        if (retained <= 0).any():
            raise ValueError(f"{phase} retained-frame counts must be positive")
        keep = (
            (left >= MIN_VALID_SAMPLES)
            & (right >= MIN_VALID_SAMPLES)
            & (left / retained >= MIN_PHASE_AVAILABILITY)
            & (right / retained >= MIN_PHASE_AVAILABILITY)
        )
        eligible[phase] = pd.Index(audit.loc[keep, "cycle_key"])
    return eligible


def make_edges(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    margin = 4 * max(BANDWIDTH_CANDIDATES_SW)
    x = data.x_from_needle_sw.to_numpy(float)
    y = data.y_from_needle_sw.to_numpy(float)
    xmin = np.floor((np.min(x) - margin) / CELL_SIZE_SW) * CELL_SIZE_SW
    xmax = np.ceil((np.max(x) + margin) / CELL_SIZE_SW) * CELL_SIZE_SW
    ymin = np.floor((np.min(y) - margin) / CELL_SIZE_SW) * CELL_SIZE_SW
    ymax = np.ceil((np.max(y) + margin) / CELL_SIZE_SW) * CELL_SIZE_SW
    return (
        np.arange(xmin, xmax + 0.5 * CELL_SIZE_SW, CELL_SIZE_SW),
        np.arange(ymin, ymax + 0.5 * CELL_SIZE_SW, CELL_SIZE_SW),
    )


def cycle_histograms(
    group: pd.DataFrame, x_edges: np.ndarray, y_edges: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    histograms: list[np.ndarray] = []
    keys: list[str] = []
    for key, cycle in group.groupby("cycle_key", sort=True):
        histogram, _, _ = np.histogram2d(
            cycle.x_from_needle_sw, cycle.y_from_needle_sw, bins=[x_edges, y_edges]
        )
        if histogram.sum() > 0:
            histograms.append((histogram / histogram.sum()).astype(np.float32))
            keys.append(str(key))
    if not histograms:
        raise ValueError("A requested phase-hand group has no non-empty trajectories")
    return np.stack(histograms), keys


def smooth_stack(stack: np.ndarray, bandwidth_sw: float) -> np.ndarray:
    sigma_cells = bandwidth_sw / CELL_SIZE_SW
    smoothed = gaussian_filter(
        stack, sigma=(0, sigma_cells, sigma_cells), mode="constant", truncate=8.0
    )
    totals = smoothed.sum(axis=(1, 2), keepdims=True)
    return np.divide(smoothed, totals, out=np.zeros_like(smoothed), where=totals > 0)


def leave_one_video_out_score(
    stack: np.ndarray, keys: list[str], smoothed: np.ndarray
) -> tuple[float, float]:
    """Return mean and SD of cycle-equal held-out log scores."""
    floor = 1e-15
    videos = np.asarray([key.split(":", 1)[0] for key in keys])
    scores: list[float] = []
    for video in np.unique(videos):
        test = videos == video
        if (~test).sum() == 0:
            raise ValueError("Leave-one-video-out scoring requires multiple videos")
        density = smoothed[~test].mean(axis=0)
        density /= density.sum()
        scores.extend(
            (stack[test] * np.log(np.maximum(density, floor)))
            .sum(axis=(1, 2))
            .tolist()
        )
    values = np.asarray(scores, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def select_shared_bandwidth(
    data: pd.DataFrame,
    eligible: dict[str, pd.Index],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict] = []
    for phase in PHASES:
        for hand in HANDS:
            group = data[
                data.phase.eq(phase)
                & data.hand.eq(hand)
                & data.cycle_key.isin(eligible[phase])
            ]
            stack, keys = cycle_histograms(group, x_edges, y_edges)
            for bandwidth in BANDWIDTH_CANDIDATES_SW:
                mean_score, sd_score = leave_one_video_out_score(
                    stack, keys, smooth_stack(stack, bandwidth)
                )
                rows.append(
                    {
                        "phase": phase,
                        "hand": hand,
                        "bandwidth_sw": bandwidth,
                        "contributing_cycles": len(stack),
                        "mean_heldout_cycle_log_score": mean_score,
                        "sd_heldout_cycle_log_score": sd_score,
                    }
                )
    scores = pd.DataFrame(rows)
    aggregate = (
        scores.groupby("bandwidth_sw", as_index=False)
        .mean(numeric_only=True)[["bandwidth_sw", "mean_heldout_cycle_log_score"]]
        .rename(
            columns={
                "mean_heldout_cycle_log_score": "equal_group_mean_log_score"
            }
        )
    )
    best_bandwidth = float(
        aggregate.loc[aggregate.equal_group_mean_log_score.idxmax(), "bandwidth_sw"]
    )
    paired = scores.pivot(
        index=["phase", "hand"],
        columns="bandwidth_sw",
        values="mean_heldout_cycle_log_score",
    )
    losses = paired[best_bandwidth].to_numpy()[:, None] - paired.to_numpy()
    paired_loss = pd.DataFrame(
        {
            "bandwidth_sw": paired.columns.astype(float),
            "paired_mean_loss_from_best": losses.mean(axis=0),
            "paired_se_loss_from_best": losses.std(axis=0, ddof=1)
            / np.sqrt(losses.shape[0]),
        }
    )
    aggregate = aggregate.merge(paired_loss, on="bandwidth_sw", how="left")
    # Same paired one-standard-error preference used in the reported analysis:
    # choose the largest candidate whose loss from the best is no greater than
    # the paired SE of that loss across the six phase-hand groups.
    candidates = aggregate[
        aggregate.paired_mean_loss_from_best
        <= aggregate.paired_se_loss_from_best + 1e-12
    ]
    selected = float(candidates.bandwidth_sw.max())
    scores["selected_shared_bandwidth"] = np.isclose(
        scores.bandwidth_sw, selected
    )
    return selected, scores


def hdr_threshold(mass: np.ndarray, coverage: float) -> float:
    ordered = np.sort(mass.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    index = int(np.searchsorted(cumulative, coverage, side="left"))
    return float(ordered[min(index, len(ordered) - 1)])


def summarize_density(
    phase: str,
    hand: str,
    group: pd.DataFrame,
    mass: np.ndarray,
    bandwidth: float,
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    analysis_set: str,
) -> list[dict]:
    xx, yy = np.meshgrid(x_centres, y_centres, indexing="ij")
    rows: list[dict] = []
    for coverage in (0.50, 0.90):
        threshold = hdr_threshold(mass, coverage)
        mask = mass >= threshold
        selected = mass * mask
        x_centroid = float((xx * selected).sum() / selected.sum())
        y_centroid = float((yy * selected).sum() / selected.sum())
        rows.append(
            {
                "analysis_set": analysis_set,
                "phase": phase,
                "hand": hand,
                "coverage": coverage,
                "contributing_cycles": int(group.cycle_key.nunique()),
                "valid_wrist_observations": int(len(group)),
                "area_sw2": float(mask.sum() * CELL_SIZE_SW**2),
                "achieved_probability_mass": float(mass[mask].sum()),
                "x_centroid_sw": x_centroid,
                "y_centroid_sw": y_centroid,
                "centroid_distance_from_needle_sw": float(
                    np.hypot(x_centroid, y_centroid)
                ),
                "probability_mass_threshold": threshold,
                "cell_size_sw": CELL_SIZE_SW,
                "isotropic_bandwidth_sw": bandwidth,
            }
        )
    return rows


def fit_group(
    group: pd.DataFrame,
    bandwidth: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    stack, keys = cycle_histograms(group, x_edges, y_edges)
    mass = smooth_stack(stack, bandwidth).mean(axis=0)
    mass /= mass.sum()
    return mass, keys


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data, audit = load_inputs(args.trajectories, args.cycle_audit, args.expected_cycles)
    eligible = primary_eligible_cycles(audit)
    x_edges, y_edges = make_edges(data)
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2
    bandwidth, cv = select_shared_bandwidth(data, eligible, x_edges, y_edges)

    primary_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    grid_rows: list[dict] = []
    for phase in PHASES:
        for hand in HANDS:
            all_group = data[data.phase.eq(phase) & data.hand.eq(hand)]
            primary_group = all_group[all_group.cycle_key.isin(eligible[phase])]
            primary_mass, _ = fit_group(
                primary_group, bandwidth, x_edges, y_edges
            )
            primary_rows.extend(
                summarize_density(
                    phase,
                    hand,
                    primary_group,
                    primary_mass,
                    bandwidth,
                    x_centres,
                    y_centres,
                    "primary_completeness_screened",
                )
            )
            sensitivity_mass, _ = fit_group(
                all_group, bandwidth, x_edges, y_edges
            )
            sensitivity_rows.extend(
                summarize_density(
                    phase,
                    hand,
                    all_group,
                    sensitivity_mass,
                    bandwidth,
                    x_centres,
                    y_centres,
                    "all_nonempty",
                )
            )
            xx, yy = np.meshgrid(x_centres, y_centres, indexing="ij")
            threshold_50 = hdr_threshold(primary_mass, 0.50)
            threshold_90 = hdr_threshold(primary_mass, 0.90)
            grid_rows.extend(
                {
                    "phase": phase,
                    "hand": hand,
                    "x_from_needle_sw": x,
                    "y_from_needle_sw": y,
                    "probability_mass": probability,
                    "inside_50_percent_hdr": bool(probability >= threshold_50),
                    "inside_90_percent_hdr": bool(probability >= threshold_90),
                }
                for x, y, probability in zip(
                    xx.ravel(), yy.ravel(), primary_mass.ravel()
                )
            )

    primary = pd.DataFrame(primary_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    grid = pd.DataFrame(grid_rows)
    eligibility = pd.DataFrame(
        {
            "phase": PHASES,
            "retained_cohort_cycles": args.expected_cycles,
            "primary_bilateral_eligible_cycles": [
                len(eligible[phase]) for phase in PHASES
            ],
        }
    )
    expected_primary = {"P1": 194, "P2": 210, "P3": 181}
    observed_primary = {
        phase: int(len(eligible[phase])) for phase in PHASES
    }
    if args.expected_cycles == 210 and observed_primary != expected_primary:
        raise RuntimeError(
            f"Primary eligibility differs from the reported result: {observed_primary}"
        )
    if not np.allclose(
        grid.groupby(["phase", "hand"]).probability_mass.sum(), 1.0, atol=1e-6
    ):
        raise RuntimeError("At least one exported density grid does not sum to one")
    if (primary.achieved_probability_mass + 1e-8 < primary.coverage).any():
        raise RuntimeError("At least one HDR does not attain its target coverage")

    cv.to_csv(output / "bandwidth_cross_validation.csv", index=False)
    eligibility.to_csv(output / "primary_phase_eligibility.csv", index=False)
    primary.to_csv(output / "primary_hdr_summary.csv", index=False)
    sensitivity.to_csv(output / "all_nonempty_sensitivity_summary.csv", index=False)
    grid.to_csv(output / "primary_density_grid.csv", index=False)
    manifest = {
        "retained_cycles": args.expected_cycles,
        "primary_eligible_cycles": observed_primary,
        "cell_size_sw": CELL_SIZE_SW,
        "selected_shared_isotropic_bandwidth_sw": bandwidth,
        "primary_rule": {
            "minimum_valid_samples_per_wrist": MIN_VALID_SAMPLES,
            "minimum_phase_availability_per_wrist": MIN_PHASE_AVAILABILITY,
            "bilateral": True,
        },
        "aggregation": (
            "Each eligible phase-hand-cycle occupancy histogram is normalized "
            "to unit mass before equal-cycle averaging."
        ),
        "sensitivity_rule": "all non-empty phase-hand trajectories",
        "upstream_boundary": (
            "The 220-to-210 quality-control decisions and P1-P3 phase annotations "
            "are supplied as machine-readable inputs and are not inferred here."
        ),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
