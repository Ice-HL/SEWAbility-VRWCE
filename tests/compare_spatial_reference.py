"""Compare public numerical outputs with an independently generated reference.

Only numerical tables are read; no figures or identifiable footage are needed.
This optional regression check accepts local paths rather than embedding them.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare_tables(label, actual, reference, keys, columns, tolerance):
    if actual.duplicated(keys).any() or reference.duplicated(keys).any():
        raise ValueError(f"{label}: duplicate comparison keys")
    actual = actual.set_index(keys).sort_index()
    reference = reference.set_index(keys).sort_index()
    keys_match = bool(actual.index.equals(reference.index))
    result = {"rows": len(actual), "reference_rows": len(reference),
              "keys_match": keys_match, "columns": {}}
    for column in columns:
        if not keys_match:
            result["columns"][column] = {"passed": False}
            continue
        left = actual[column].to_numpy()
        right = reference[column].to_numpy()
        exact = bool(np.array_equal(left, right, equal_nan=True))
        if pd.api.types.is_numeric_dtype(actual[column]):
            difference = np.abs(left.astype(float) - right.astype(float))
            maximum = float(np.nanmax(difference)) if len(difference) else 0.0
            passed = bool(np.allclose(left, right, atol=tolerance, rtol=0,
                                      equal_nan=True))
            result["columns"][column] = {
                "exactly_equal": exact, "max_absolute_difference": maximum,
                "passed": passed,
            }
        else:
            result["columns"][column] = {"exactly_equal": exact, "passed": exact}
    result["passed"] = keys_match and all(
        item["passed"] for item in result["columns"].values()
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-output", required=True, type=Path)
    parser.add_argument("--reference-source-data", required=True, type=Path)
    parser.add_argument("--supplementary-current", type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    output = args.public_output
    source = args.reference_source_data
    comparisons = {}

    def check(label, actual, reference, keys, columns):
        comparisons[label] = compare_tables(
            label, actual, reference, keys, columns, args.absolute_tolerance
        )

    cv = pd.read_csv(output / "bandwidth_cross_validation.csv")
    reference_cv = pd.read_csv(source / "isotropic_bandwidth_cross_validation_210.csv")
    reference_cv = reference_cv[
        reference_cv.cross_validation_scheme.eq("leave_one_video_out")
    ]
    check("all_180_phase_hand_lovo_scores", cv, reference_cv,
          ["phase", "hand", "bandwidth_sw"], [
              "contributing_cycles", "mean_heldout_cycle_log_score",
              "sd_heldout_cycle_log_score", "equal_group_mean_loo_log_score",
              "score_rank", "selected_shared_bandwidth",
          ])

    aggregate = pd.read_csv(output / "shared_bandwidth_selection_30_candidates.csv")
    aggregate_columns = ["equal_group_mean_loo_log_score", "score_rank",
                         "selected_shared_bandwidth"]
    check("all_30_aggregate_scores", aggregate,
          pd.read_csv(source / "shared_bandwidth_selection_30_candidates.csv"),
          ["bandwidth_sw"], aggregate_columns)
    if args.supplementary_current:
        check("supplementary_all_30_aggregate_scores", aggregate, pd.read_csv(
            args.supplementary_current / "shared_bandwidth_selection_30_candidates.csv"
        ), ["bandwidth_sw"], aggregate_columns)
        supplementary_cv = pd.read_csv(
            args.supplementary_current / "isotropic_bandwidth_cross_validation_210.csv"
        )
        supplementary_cv = supplementary_cv[
            supplementary_cv.cross_validation_scheme.eq("leave_one_video_out")
        ]
        check("supplementary_all_180_phase_hand_lovo_scores", cv, supplementary_cv,
              ["phase", "hand", "bandwidth_sw"], [
                  "contributing_cycles", "mean_heldout_cycle_log_score",
                  "sd_heldout_cycle_log_score", "equal_group_mean_loo_log_score",
                  "score_rank", "selected_shared_bandwidth",
              ])

    summary_columns = [
        "contributing_cycles", "valid_wrist_observations", "area_sw2",
        "achieved_probability_mass", "x_centroid_sw", "y_centroid_sw",
        "centroid_distance_from_needle_sw", "probability_mass_threshold",
        "cell_size_sw", "isotropic_bandwidth_sw",
    ]
    for label, public_name, reference_name in [
        ("primary_hdr", "primary_hdr_summary.csv",
         "phase_hand_region_summary_isotropic_210.csv"),
        ("all_nonempty_hdr", "all_nonempty_sensitivity_summary.csv",
         "phase_hand_region_nonempty_cycle_sensitivity_210.csv"),
    ]:
        reference = pd.read_csv(source / reference_name).rename(
            columns={"area_shoulder_width_squared": "area_sw2"}
        )
        check(label, pd.read_csv(output / public_name), reference,
              ["phase", "hand", "coverage"], summary_columns)

    check("primary_phase_eligibility", pd.read_csv(output / "primary_phase_eligibility.csv"),
          pd.read_csv(source / "phase_primary_eligibility_210.csv"), ["phase"],
          ["retained_cohort_cycles", "primary_bilateral_eligible_cycles"])
    grid = pd.read_csv(output / "primary_density_grid.csv")
    check("every_exported_density_cell", grid,
          pd.read_csv(source / "cycle_balanced_density_grid_isotropic_210.csv"),
          ["phase", "hand", "x_from_needle_sw", "y_from_needle_sw"],
          ["probability_mass", "inside_50_percent_hdr", "inside_90_percent_hdr"])

    manifest = json.loads((output / "analysis_manifest.json").read_text())
    reference_manifest = json.loads((source.parent / "analysis_manifest_210.json").read_text())
    manifest_checks = {
        key: manifest[key] == reference_manifest[key]
        for key in ["retained_cycles", "cell_size_sw", "grid_shape", "grid_extent_sw",
                    "bandwidth_candidates_sw", "selected_shared_isotropic_bandwidth_sw"]
    }
    mass_sums = grid.groupby(["phase", "hand"]).probability_mass.sum()
    invariants = {
        "thirty_candidates": len(aggregate) == 30,
        "six_phase_hand_groups_per_candidate": bool(
            cv.groupby("bandwidth_sw").size().eq(6).all()
        ),
        "one_data_driven_selected_candidate": int(aggregate.selected_shared_bandwidth.sum()) == 1,
        "density_grids_sum_to_one": bool(np.allclose(mass_sums, 1, atol=1e-6, rtol=0)),
        "all_primary_hdrs_attain_target_coverage": bool((
            pd.read_csv(output / "primary_hdr_summary.csv").achieved_probability_mass
            + 1e-8 >= pd.read_csv(output / "primary_hdr_summary.csv").coverage
        ).all()),
    }
    report = {
        "absolute_tolerance": args.absolute_tolerance,
        "comparisons": comparisons,
        "manifest_checks": manifest_checks,
        "invariants": invariants,
        "density_mass_sums": {f"{phase}_{hand}": float(value)
                              for (phase, hand), value in mass_sums.items()},
        "selected_shared_isotropic_bandwidth_sw": manifest[
            "selected_shared_isotropic_bandwidth_sw"
        ],
        "passed": (all(item["passed"] for item in comparisons.values())
                   and all(manifest_checks.values()) and all(invariants.values())),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
