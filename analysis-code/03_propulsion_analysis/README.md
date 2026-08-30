# P2 automatic kinematic-candidate analysis

This module contains the two core scripts used to derive the real-work P2
propulsion characteristics reported for the retained Task A cohort.

1. `detect_automatic_kinematic_candidates.py` detects bilateral-forward
   candidate intervals inside the retained P2 intervals and merges adjacent
   cores only when neither wrist shows meaningful return movement.
2. `summarize_propulsion_characteristics.py` estimates cycle-level propulsion
   direction and derives movement geometry, duration, true inter-action gap,
   bilateral wrist separation, and wrist-midpoint path speed.

## Terminology and validation boundary

The 4,916 detected intervals are called **automatic kinematic candidates**.
They satisfy the prespecified motion rule but were not manually confirmed one
by one as semantic sewing actions. The code therefore does not label them as
ground-truth pushes and does not derive a clinical or employment threshold.

## Inputs

### Retained P2 interval CSV

Exactly one P2 row for each of the 210 retained cycles, with these columns:

| Column | Meaning |
|---|---|
| `video_id` or `Video ID` | Numeric study video identifier |
| `work_cycle_id` or `Work cycle ID` | Cycle number within video |
| `phase` or `Phase` | Must equal `P2` |
| `start_frame` or `Start frame` | Inclusive P2 start frame |
| `end_frame_exclusive` or `End frame (exclusive)` | Exclusive P2 end frame |
| `fps` or `FPS` | Source-video frame rate |

The 220-to-210 quality-control decisions and the P1–P3 annotations are
documented in Supplementary Table S20 and are treated as upstream inputs. This
module does not recreate those review decisions from identifiable videos.

### Per-video trajectory CSVs

The directory supplied with `--trajectory-dir` must contain one file named
`{video_id}_trajectory.csv` per video. Required columns are:

- `frame`
- `left_shoulder_x`, `left_shoulder_y`
- `right_shoulder_x`, `right_shoulder_y`
- `left_wrist_x`, `left_wrist_y`
- `right_wrist_x`, `right_wrist_y`

Coordinates are normalized image coordinates. The scripts convert them into
an equal-pixel 3840 × 2160 image plane and use cycle-specific median shoulder
width for normalization.

## Run order

```bash
python detect_automatic_kinematic_candidates.py \
  --phase-intervals /path/to/phase_intervals_210.csv \
  --trajectory-dir /path/to/trajectory_csvs \
  --output-dir /path/to/candidate_outputs

python summarize_propulsion_characteristics.py \
  --candidates /path/to/candidate_outputs/automatic_kinematic_candidates.csv \
  --cycle-counts /path/to/candidate_outputs/candidate_counts_by_cycle.csv \
  --trajectory-dir /path/to/trajectory_csvs \
  --output-dir /path/to/propulsion_outputs
```

## Aggregation rules made explicit

- Workload is summarized from one automatic-candidate count per cycle.
- Direction is estimated once per retained cycle using PCA of bilateral
  wrist-midpoint samples from its candidate intervals; circular statistics give
  each cycle equal weight. Confidence limits use a video-cluster bootstrap.
- Axial displacement (`D`) and lateral excursion (`W`) are summarized by first
  taking the median across candidates within each cycle and then the P90 across
  the 210 cycle-level medians.
- Inter-action gap is the next candidate onset minus the preceding candidate
  end within the same cycle. It is not an onset-to-onset interval. For 4,916
  candidates nested in 210 cycles, this yields 4,706 adjacent-candidate pairs.
- Wrist separation, duration, and path speed use pooled candidate-level
  distributions, as specified in the manuscript table.
- Multiplication by the 0.40-m reference shoulder width produces nominal
  reference-scaled design values; it does not recover metric distances from
  the top-down videos.

No local usernames, personal names, or machine-specific paths are embedded in
the scripts.
