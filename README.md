# SEWAbility-VRWCE

This repository accompanies a study that translated observed sewing-work demands into an executable and measurable virtual-reality work-capacity evaluation (VR-WCE) prototype.

It contains the study-specific code implementing the principal offline analytical steps and the supplementary materials cited in the manuscript. The earlier SEWAbility framework is available separately at [Ice-HL/SEWAbility](https://github.com/Ice-HL/SEWAbility).

## Repository structure

```text
SEWAbility-VRWCE/
├── README.md
├── analysis-code/
└── supplementary-material/
```

### Analysis code

The code is organized by manuscript workflow:

1. [`01_video_analysis`](analysis-code/01_video_analysis/) — video-level motion features, K-means/PCA sensitivity analyses, candidate work-cycle detection, 1-s window analysis, and temporal mapping of motion clusters.
2. [`02_spatial_analysis`](analysis-code/02_spatial_analysis/) — retained-cohort inputs and cycle-balanced wrist-distribution/HDR analysis.
3. [`03_propulsion_analysis`](analysis-code/03_propulsion_analysis/) — automatic bilateral-forward kinematic candidates, cycle-level direction estimates, circular summaries, video-clustered bootstrap analysis, and propulsion-parameter aggregation.
4. [`04_vr_outcome_analysis`](analysis-code/04_vr_outcome_analysis/) — phase-specific JSON processing, action classification, and derivation of the four VR-WCE outcome domains.

Install the Python dependencies with:

```bash
python -m pip install -r analysis-code/requirements.txt
```

Each module contains its own README describing its inputs, outputs, execution order, assumptions, and interpretation boundary.

## Input formats

The scripts operate on de-identified numerical records rather than worker names. Required input classes include:

- pose-trajectory CSV files with frame/time fields and two-dimensional upper-extremity key points;
- an 88-feature table indexed by numeric video identifier;
- candidate-cycle and adjudicated phase-interval tables indexed by numeric video and cycle identifiers;
- phase-labelled wrist-trajectory and propulsion-event tables;
- phase-specific VR JSON records containing timestamps, event counts, hand-use classifications, and wrist-proxy positions.

Exact required fields are documented in the README and command-line help for each module. Paths are supplied as command-line arguments; no local usernames or computer-specific directories are embedded in the released code.

## Supplementary material

The [`supplementary-material`](supplementary-material/) directory contains the three supplementary figures and Supplementary Tables S1–S20 cited in the accompanying manuscript.

## Data and privacy notes

- Workplace videos are not included because they contain potentially identifiable images of workers.
- Numeric video and session identifiers are study identifiers rather than worker names.
- The released code covers the principal offline video, spatial, propulsion, and VR-record analyses. It does not include the complete Unity project used to implement the VR prototype.
