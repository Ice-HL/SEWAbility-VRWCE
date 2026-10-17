# Spatial bandwidth validation records

These aggregate numerical records document bandwidth selection for the
manuscript's retained cohort of 210 Task A cycles. They do not include videos,
worker names, or the full wrist trajectories. See the [spatial analysis
README](../README.md) for the public input contract and execution command.

## Current selection rule

The analysis evaluates 30 shared isotropic Gaussian bandwidths from 0.01 to
0.30 shoulder-width units (SW), inclusive, in 0.01-SW steps. Leave-one-video-out
scoring gives held-out cycles equal weight within each phase–hand group; the
six phase–hand mean log scores then receive equal weight. The bandwidth with
the highest shared mean log score is selected. An exact tie is resolved by the
smallest bandwidth.

For the manuscript dataset, this rule selects 0.04 SW; it is not a fixed input.
Supplementary Table S19 displays only the ten candidates from 0.01 to 0.10 SW.
The full search and the 0.11–0.30-SW check are retained here rather than treated
as additional displayed rows in S19.

## Files

| File | Role |
|---|---|
| [shared_bandwidth_selection_30_candidates.csv](shared_bandwidth_selection_30_candidates.csv) | Shared scores and selection across all 30 candidates |
| [isotropic_bandwidth_cross_validation_210.csv](isotropic_bandwidth_cross_validation_210.csv) | Phase–hand cross-validation scores, including primary leave-one-video-out scores and additional leave-one-cycle-out diagnostics |
| [s19_display_candidates_001_010.csv](s19_display_candidates_001_010.csv) | The ten-candidate 0.01–0.10-SW subset displayed in S19 |
| [extended_range_check_011_030.csv](extended_range_check_011_030.csv) | The additional 0.11–0.30-SW candidates |
| [isotropic_bandwidth_area_sensitivity_210.csv](isotropic_bandwidth_area_sensitivity_210.csv) | HDR area sensitivity to bandwidth in the retained cohort |
| [rule_comparison_on_same_30_candidates.csv](rule_comparison_on_same_30_candidates.csv) | Current and historical selection rules applied to the same 30 candidates |

## Revision provenance and interpretation

The earlier repository implementation at commit
`0a6a9a901fb9c697491755ab59ccfa1add1d3191` used eight candidates (0.025, 0.04,
0.06, 0.08, 0.10, 0.12, 0.15, and 0.18 SW) and a paired one-standard-error
preference for the largest qualifying bandwidth. Both that candidate grid
and that selection rule have been revised; the earlier version remains in Git
history. This is a documented analysis revision, not a claim of
preregistration.

On the same 30-candidate grid, the historical paired one-standard-error rule
selects 0.05 SW, while the current highest-score rule selects 0.04 SW. The
comparison file isolates the selector difference from the grid change; it is
not an alternative primary analysis or a claim that the historical code used
30 candidates.

The reference cross-validation file contains 360 candidate–group rows:
180 leave-one-video-out rows and 180 leave-one-cycle-out diagnostic rows.
Only the leave-one-video-out rows enter the primary selector. The public CLI
exports its 180 primary rows as `bandwidth_cross_validation.csv`; it does not
generate the additional leave-one-cycle-out diagnostics.

These files support inspection of the numerical search and its reporting.
They do not reproduce source-video quality-control decisions or replace the
de-identified trajectory and cycle-audit tables needed to rerun the full
analysis. The spatial estimates inform a functional two-dimensional VR mapping,
not calibrated three-dimensional geometry or validated work-capacity or safety
thresholds.
