# Workplace-video analysis

This directory contains the core, de-identified scripts used to characterize the
workplace-video corpus and to relate segment-level motion clusters to candidate
sewing work cycles. Local absolute paths and investigator usernames have been
removed. Source videos are not distributed because they contain potentially
identifiable images of workers.

## Recommended execution order

1. `01_extract_video_level_88_features.py`
   extracts one 88-feature vector per video from pose-trajectory CSV files.
2. `02_cluster_video_level_features.py`
   standardizes those vectors and evaluates K-means solutions in the Raw88,
   PCA90, and PCA95 representations.
3. `03_detect_work_cycle_boundaries.py`
   trains the study-specific random-forest boundary detector from de-identified
   manual frame annotations and generates candidate work-cycle intervals.
4. `04_extract_segment_level_88_features.py`
   extracts overlapping 1-s motion windows at a 0.5-s step from the videos.
5. `05_cluster_segment_level_features.py`
   evaluates K-means solutions for the segment-level Raw88, PCA90, and PCA95
   representations.
6. `06_map_segment_clusters_to_work_cycles.py`
   maps segment assignments to candidate work cycles and summarizes their
   normalized temporal distributions.

The scripts expose all file locations as command-line arguments. Run any script
with `--help` to see the complete interface.

## Inputs and outputs

### 01 — video-level feature extraction

Inputs:

- A directory of `<video_id>_trajectory.csv` files. Each file must contain the
  two-dimensional columns `<joint>_x` and `<joint>_y` for bilateral shoulder,
  elbow, and wrist landmarks.
- An FPS table with `Filename` and `FPS` columns.

Output: one CSV containing `Video`, `FPS`, and 88 motion-feature columns.

Example:

```bash
python 01_extract_video_level_88_features.py \
  --trajectory-dir data/trajectories \
  --fps-csv data/video_fps.csv \
  --output-csv results/video_level_88_features.csv
```

### 02 — video-level clustering

Input: the output of script 01. `Video` and `FPS` are treated as metadata; the
remaining 88 numeric columns are standardized before clustering. The preserved
analysis evaluates K = 2–13 with K-means++ initialization, 50 initializations,
and `random_state=42`.

```bash
python 02_cluster_video_level_features.py \
  --input-csv results/video_level_88_features.csv \
  --output-dir results/video_level_clustering
```

Outputs include assignments for every K, silhouette-score summaries, PCA
variance summaries, and descriptive two-dimensional PCA plots.

### 03 — candidate work-cycle boundaries

Inputs:

- De-identified MP4 files named `<video_id>.mp4`.
- Corresponding `<video_id>_trajectory.csv` files containing `frame` plus
  bilateral shoulder, elbow, and wrist x/y coordinates.
- A JSON object mapping de-identified video IDs to manually annotated
  cloth-change frame numbers, for example:

```json
{
  "V001": [1250, 2480],
  "V002": [930, 2010, 3150]
}
```

```bash
python 03_detect_work_cycle_boundaries.py \
  --video-dir data/videos \
  --trajectory-dir data/trajectories \
  --manual-boundaries-json data/manual_boundaries.json \
  --output-dir results/work_cycle_boundaries
```

The preserved detector uses a 2-s feature window, grouped leave-one-video-out
validation, a random forest with `random_state=42`, and the thresholds defined
in the script. Its master candidate-cycle table is
`TaskA_all_work_cycles.csv`. Subsequent cycle-level quality control, rather
than this detector alone, produced the retained 210-cycle analytic cohort
reported in the manuscript.

### 04 — segment-level feature extraction

Inputs are the de-identified MP4 directory and the video-level CSV from script
01 (used for its `Video` and `FPS` columns). The script uses MediaPipe Pose and
the study's fixed 1-s window and 0.5-s step.

```bash
python 04_extract_segment_level_88_features.py \
  --video-dir data/videos \
  --fps-csv results/video_level_88_features.csv \
  --output-dir results/segment_level_88_features
```

Output: one `<video_id>_segments.csv` file per video plus an extraction summary.

### 05 — segment-level clustering

```bash
python 05_cluster_segment_level_features.py \
  --input-dir results/segment_level_88_features \
  --output-dir results/segment_level_clustering
```

The preserved analysis evaluates K = 2–6 in Raw88, PCA90, and PCA95 spaces,
uses `random_state=42`, and estimates silhouette scores from at most 10,000
segments. Outputs include the cluster-assignment CSV required by script 06.

### 06 — temporal mapping to candidate work cycles

```bash
python 06_map_segment_clusters_to_work_cycles.py \
  --segment-clusters-csv results/segment_level_clustering/cluster_assignments/cluster_assignment_PCA90_k3.csv \
  --work-cycles-csv results/work_cycle_boundaries/TaskA_all_work_cycles.csv \
  --output-dir results/work_cycle_cluster_mapping
```

The script assigns a segment to a cycle using its center frame, then summarizes
cluster occupancy across the normalized 0–100% work-cycle timeline.

## Important feature-definition distinction

Scripts 01 and 04 each produce 88 columns, but their shoulder-angle definitions
are not identical and have intentionally **not** been harmonized in this
released implementation:

- Script 01 (video-level features): shoulder angle is
  `elbow–shoulder–opposite shoulder`.
- Script 04 (segment-level windows): shoulder angle is
  `hip–shoulder–elbow`.
- Both scripts define elbow angle as `shoulder–elbow–wrist`.

This distinction was present in the source analysis. The two matrices serve
different analyses and must not be merged, substituted for one another, or
interpreted as identically defined 88-feature representations. The difference
should be resolved prospectively if a single unified feature definition is used
in future studies.

## Reproducibility boundaries

- Video IDs must be pseudonymous study identifiers; filenames must not contain
  participant names or other direct identifiers.
- MediaPipe and OpenCV versions can affect decoded frame counts and pose
  estimates. Use the repository dependency specification for the intended
  environment.
- The original videos are not included. These scripts therefore document the
  principal computational steps but cannot independently recreate pose data
  from unavailable identifiable source recordings.
- Cluster labels are arbitrary numeric labels. Functional interpretations were
  assigned only after reviewing temporal location and source-video context.
