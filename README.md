# SEWAbility-VRWCE

Core offline analysis code and supplementary materials for the video-informed
VR work-capacity evaluation study. The code is organized by analysis stage,
not as one drawing script per manuscript figure.

## Analysis modules

| Module | Main outputs |
| --- | --- |
| [01_video_analysis](analysis-code/01_video_analysis/) | Video/window motion features, clustering, candidate cycle boundaries, and cluster-to-cycle mapping |
| [02_spatial_analysis](analysis-code/02_spatial_analysis/) | Cycle-balanced wrist-use densities and highest-density regions |
| [03_propulsion_analysis](analysis-code/03_propulsion_analysis/) | Automatic kinematic candidates, cycle-level directions, and propulsion summaries |
| [04_vr_outcome_analysis](analysis-code/04_vr_outcome_analysis/) | Work speed, quantity, movement-related accuracy, and stability from session records |

Install dependencies and follow the input specifications and commands in each
module's README:

```bash
python -m pip install -r analysis-code/requirements.txt
```

The VR module can export numerical results without loading its optional
Matplotlib drawing code by using `--skip-figures`. Its optional descriptive
plots are not a pixel-identical reconstruction of the final manuscript layout.
The release does not supply a complete set of final figure-layout scripts.

## Inputs and scope

Inputs include pose-trajectory CSVs, per-video sampling-rate records, cycle and
phase annotations, and phase-specific VR JSON files. Use numeric or pseudonymous
video/session identifiers. Sampling-rate values must match the trajectory
records supplied to an analysis; the feature scripts do not independently
validate the source video's frame timestamps.

Source-video interpretation, final manual quality-control decisions, and the
complete Unity project are outside this release. The spatial and propulsion
modules start from the retained-cohort inputs documented in Supplementary Table
S19. Identifiable workplace videos are not included. Consequently, code
availability does not imply that all private inputs are publicly downloadable.

## Spatial method and verification

The current shared bandwidth search evaluates 0.01–0.30 SW in 0.01-SW steps and
selects the highest mean leave-one-video-out log score, with equal weights for
the six phase–hand groups. Exact ties use the smallest bandwidth. The manuscript
dataset selects 0.04 SW; S19 displays the first ten candidates. The
[spatial README](analysis-code/02_spatial_analysis/) and
[validation records](analysis-code/02_spatial_analysis/validation/) retain the
full search and explain the revision from the earlier paired-one-standard-error
rule. Neither version is described as preregistered.

Run the focused code checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

Tests check computational behavior, not clinical or employment validity.
Needle-event fields retained for JSON compatibility are diagnostics, not reported
paper outcomes or validated safety measures.

## Supplementary material and privacy

[Supplementary material](supplementary-material/) contains Tables S1–S19 and
Figures S1–S3, with an [ordered index](supplementary-material/README.md).

Paths are provided at runtime rather than hard-coded to a researcher's machine.
Examples use project-relative paths. Local console messages may display the
paths supplied by the person running a script; review new logs and generated
files before publishing them. Do not commit private video/session files or
identifying filenames. Full figure-reproduction working files and local audit
reports are not part of this public code release.
