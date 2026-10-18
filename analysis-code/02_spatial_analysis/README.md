# Cycle-balanced spatial analysis

`cycle_balanced_wrist_regions.py` reproduces the principal spatial-estimation
step used for the retained Task A cohort. It estimates phase- and hand-specific
50% core and 90% principal highest-density regions (HDRs) with equal
contribution from every eligible work cycle.

## Scope and analysis boundary

The script intentionally begins **after** video review, cycle quality control,
and P1–P3 annotation. Those upstream decisions reduced 220 candidate work
cycles to 210 retained cycles. They are not regenerated from video by this
release because the source videos contain identifiable participant images and
the final adjudication depended on source-video review rather than a single
automated script. The retained-cohort records are documented in Supplementary
Table S19.

The code therefore requires two de-identified, machine-readable inputs:

### 1. Phase-labelled trajectory CSV

One row per valid wrist observation, with the following columns:

| Column | Meaning |
|---|---|
| `video_id` | Numeric study video identifier |
| `work_cycle_id` | Cycle number within the video |
| `cycle_key` | Stable de-identified key, e.g. `14:6` |
| `phase` | `P1`, `P2`, or `P3` |
| `hand` | `left` or `right` |
| `x_from_needle_sw` | Needle-centred lateral coordinate in shoulder-width units (SW) |
| `y_from_needle_sw` | Needle-centred vertical image-plane coordinate in SW; positive upward |

The coordinates must already reflect the study preprocessing: normalized
image coordinates converted using the source image width and height, followed
by division by the cycle-specific median image-based shoulder width.

### 2. Cycle-audit CSV

Exactly one row per retained cycle. Required columns are:

- `cycle_key`
- `{phase}_retained_frames`
- `{phase}_left_valid_frames`
- `{phase}_right_valid_frames`

where `{phase}` is `P1`, `P2`, or `P3`.

## Primary rules implemented

- Retained cohort: 210 cycles.
- Primary phase eligibility: both wrists have at least 10 valid samples and at
  least 50% availability within that phase.
- Grid-cell width: 0.025 SW.
- Each phase–hand–cycle occupancy histogram is normalized to unit mass before
  aggregation.
- The shared isotropic Gaussian bandwidth is selected by leave-one-video-out
  scoring over 30 candidates: 0.01–0.30 SW in 0.01-SW steps. Within each of the
  six phase–hand groups, held-out cycles contribute equally to the mean log
  score. The six group means then contribute equally to the shared score.
- Select the candidate with the highest shared mean log score. If candidates
  tie exactly, select the smallest bandwidth. There is no one-standard-error
  preference in the current selector.
- Primary 50% and 90% HDRs are accompanied by a sensitivity analysis that uses
  every non-empty phase–hand trajectory.

For the manuscript analysis dataset, the primary eligible-cycle counts are
P1 = 194, P2 = 210, and P3 = 181.

## Bandwidth results and method history

The manuscript dataset selects 0.04 SW. This is an observed result of the
candidate search, not a bandwidth fixed in the code. Supplementary Table S19
displays ten candidates from 0.01 to 0.10 SW. The numerical records in
[`validation/`](validation/) contain all 30 candidates, the 0.11–0.30-SW
extended-range check, and the corresponding area-sensitivity results.

The candidate grid and selection rule are revisions to the earlier repository
implementation at commit `0a6a9a901fb9c697491755ab59ccfa1add1d3191`, which used
eight candidates (0.025, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, and 0.18 SW) and
selected the largest candidate within a paired one-standard-error loss from
the best mean score across the six phase–hand groups. That implementation
remains archived in Git history; it is not the current analysis rule. This
revision is documented for reproducibility and is not described as
preregistered.

The validation records compare both selectors on the same 30-candidate grid:
the current maximum-score rule selects 0.04 SW, whereas the historical paired
one-standard-error rule would select 0.05 SW. This separates the effect of the
selection rule from the change in candidate grid.

## Connection to VR spatial mapping

The outputs are shoulder-normalized, two-dimensional wrist-use densities and
regions. Together with movement pathways and workstation context, they inform
the global VR task-space organization; this is not metric 3D reconstruction.
The P1 left-wrist 90% region provides the area reference for the tabletop
placement-completion zone. The compact P2 distributions inform the
needle-adjacent propulsion location, not the dimensions of the local hand
guidance cues. The out-of-frame floor-level region is specified from on-site
observation and factory consultation rather than from a wrist-density estimate.

The corresponding spatial results and implemented global geometry are reported
in Tables 2 and 3 of the main manuscript, respectively. Local propulsion
parameters are described in Table 1 and in `03_propulsion_analysis`.

## Example

```bash
python cycle_balanced_wrist_regions.py \
  --trajectories data/phase_labelled_wrist_trajectories_210.csv \
  --cycle-audit data/cycle_and_phase_sample_audit_210.csv \
  --output-dir results/spatial
```

No personal names or local computer paths are embedded in the script.

## Outputs and verification boundary

The output directory contains `bandwidth_cross_validation.csv`,
`shared_bandwidth_selection_30_candidates.csv`,
`primary_phase_eligibility.csv`, `primary_hdr_summary.csv`,
`all_nonempty_sensitivity_summary.csv`, `primary_density_grid.csv`, and
`analysis_manifest.json`. The manifest records the selected bandwidth and
analysis rules. The cross-validation export contains 180 leave-one-video-out
candidate–group rows (30 candidates × six phase–hand groups), with aggregate
score/rank fields; the shared-selection export contains one row per candidate.
The script checks retained-cycle and primary-eligibility
counts, density normalization, and achieved HDR coverage.

The released [validation records](validation/) document the manuscript-data
bandwidth analysis. The reference cross-validation file also retains
leave-one-cycle-out diagnostic rows; those diagnostics do not enter the
primary leave-one-video-out selector and are not additional CLI outputs.
The records are aggregate numerical evidence, not substitutes
for the trajectory and cycle-audit inputs required to rerun the full spatial
analysis. Reproducing the 220-to-210 source-video adjudication remains outside
the scope of this public script.

