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
- A shared isotropic Gaussian bandwidth is selected by leave-one-video-out
  scoring from the prespecified candidate set.
- Primary 50% and 90% HDRs are accompanied by a sensitivity analysis that uses
  every non-empty phase–hand trajectory.

For the manuscript analysis dataset, the primary eligible-cycle counts are
P1 = 194, P2 = 210, and P3 = 181.

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
  --trajectories /path/to/phase_labelled_wrist_trajectories_210.csv \
  --cycle-audit /path/to/cycle_and_phase_sample_audit_210.csv \
  --output-dir /path/to/spatial_outputs
```

No personal names or local computer paths are embedded in the script.
