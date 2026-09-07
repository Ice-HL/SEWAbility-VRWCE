# SEWAbility-VRWCE

This repository accompanies a study that translated observed sewing-work demands into an executable and measurable virtual-reality work-capacity evaluation (VR-WCE) prototype.

It contains the study-specific code implementing the principal offline analytical steps and the supplementary materials cited in the manuscript. The earlier SEWAbility framework is available separately at [Ice-HL/SEWAbility](https://github.com/Ice-HL/SEWAbility).

## Five evidence-to-VR mappings

The study links real-work evidence and assessment guidance to VR task specifications through five paired mappings:

| Mapping | Evidence or guidance | VR specification or output |
|---|---|---|
| Exemplar task | Workplace-video motion analysis and source-video review | An assessment task based on the operations of zipper sewing (Task A), without reproducing every product detail |
| Task sequence | Recurrent work cycles and their functional phases | Ordered preparation, propulsion, and return states with phase-completion events |
| Functional workspace | Cycle-balanced two-dimensional wrist-use distributions, movement pathways, and workstation context | Floor-level material exchange, tabletop placement, and a needle-adjacent propulsion site |
| Workload and propulsion | Cycle-level candidate counts and observed propulsion characteristics | Bounded workload and hand-specific guidance parameters; timing and speed retained as non-mandatory comparison references |
| Outcome measurement | Work-performance constructs informed by Hong Kong Labour Department productivity-assessment guidance | Indicators of work speed, work quantity, movement-related accuracy, and work stability derived from VR records |

Spatial mapping preserves functional relationships rather than reconstructing metrically calibrated three-dimensional geometry from video. The floor-level region also requires on-site and factory context because it is outside the video field of view. Global task sites and local hand-guidance cues are specified separately.

The prototype was examined in four preliminary technical sessions: two program-generated and two researcher-operated. These sessions demonstrated task execution, structured recording, and indicator derivation; they do not establish clinical validity or work-capacity thresholds.

## Repository structure

```text
SEWAbility-VRWCE/
├── README.md
├── analysis-code/
└── supplementary-material/
```

### Analysis code

The principal offline computations are organized into four modules supporting these mappings. The modules are not a one-to-one implementation of all five mappings; source-video interpretation, contextual design decisions, and the complete Unity implementation are outside their scope.

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

The [`supplementary-material`](supplementary-material/) directory contains Supplementary Tables S1–S19 and Supplementary Figures S1–S3, with an [ordered index](supplementary-material/README.md) describing each file. Their numbering matches the accompanying manuscript. Operational definitions, real-work reference values, and VR uses of propulsion characteristics are provided together in Table 1 of the main manuscript.

## Data and privacy notes

- Workplace videos are not included because they contain potentially identifiable images of workers.
- Numeric video and session identifiers are study identifiers rather than worker names.
- The released code covers the principal offline video, spatial, propulsion, and VR-record analyses. It does not include the complete Unity project used to implement the VR prototype.
