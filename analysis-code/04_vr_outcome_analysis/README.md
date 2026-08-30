# VR-WCE outcome analysis

This directory contains the study-specific offline analysis used to derive the four VR-WCE outcome domains from four pseudonymized technical sessions (`S1`, `S2`, `U1`, and `U2`). The script reads the JSON files written by the prototype, applies the prespecified action-population rules, writes action- and session-level tables, and optionally renders the descriptive four-domain figure.

## Files

- `analyze_vrwce_sessions.py`: numerical derivation, quality checks, and descriptive plotting.

The original Unity project and its C# runtime-logging source are **not** included in this repository. This directory begins at the offline JSON-analysis stage; it does not recreate the executable VR task or generate the source JSON records.

The script requires Python 3.10 or later, NumPy, and Matplotlib. Install the shared dependencies listed for the repository before running it.

## Expected input layout

Provide one directory for each pseudonymous session. Directory names may be arbitrary because the labels are assigned on the command line.

```text
sessions/
├── session_01/
│   ├── phase1_grab.json
│   ├── phase2_push.json
│   ├── phase3_return.json
│   └── session_meta.json
├── session_02/
│   └── ...
├── session_03/
│   └── ...
└── session_04/
    └── ...
```

Do not place participant names, staff names, exact collection dates, or other direct identifiers in public directory or file names. The script does not export source-directory paths, JSON session identifiers, or absolute clock timestamps.

### Required JSON fields

The principal fields consumed by the analysis are listed below. Three-dimensional positions are workstation-local coordinates in metres. Timestamps must be ISO-8601-compatible strings and must be ordered within each trajectory.

`phase1_grab.json` and `phase3_return.json`

- top level: `sessionId`, `grabSessions`
- each grab episode: `startTime`, `endTime`, `hand`, `samples`
- each sample: `time`, `tableLocalPosition` with numeric `x`, `y`, and `z`

`phase2_push.json`

- top level: `sessionId`, `pushCycles`, `pushCycleCount`, `needleSafetyEvents`, `pushLeftZone`, and `pushRightZone`
- each push action: `startTime`, `endTime`, `usedLeft`, `usedRight`, `isSingleHand`, and `samples`
- each synchronized sample: `time`, `hasLeft`, `hasRight`, `leftTableLocalPosition`, `rightTableLocalPosition`, `leftInLeftZone`, and `rightInRightZone`
- each guidance zone: `width`, `length`, `tableLocalCenter`, `tableLocalForward`, and `tableLocalRight`
- each needle event: `time`

`session_meta.json`

- `sessionId`, `endReason`, `isPartialSession`, and `isSimulated`

Additional fields may be present and are ignored unless referenced by the script.

## Analysis populations

The code keeps runtime task progression separate from offline movement eligibility:

1. **Work quantity:** every runtime-recorded P2 action contributes to the recorded action count. Bilateral participation is reported separately.
2. **Timing:** action-duration summaries use bilateral-classified actions. An inter-action gap is included only when the two originally adjacent runtime actions are both bilateral-classified; gaps are not bridged across a single-hand-classified action.
3. **Basic bilateral trajectories:** an action must be bilateral-classified, contain at least three synchronized valid samples for each hand, have positive sampled duration, and have nonzero accumulated paths.
4. **Within-action movement dynamics:** a hand trajectory must come from a bilateral-classified action and contain at least ten synchronized valid samples. No additional minimum-duration rule is applied. Finite-difference derivatives are not smoothed.

A P2 action is classified as bilateral when `usedLeft` and `usedRight` are both true and `isSingleHand` is false. Transfer-attempt counts merge temporally overlapping left- and right-hand grasp records into one attempt.

## Outcomes

The script produces measures in the four manuscript domains:

- **Work quantity:** runtime-recorded P2 action count, bilateral-classified action count, and bilateral participation rate.
- **Work speed:** P1-P3 recorded activity spans, P2 action frequency, bilateral action duration, and non-bridged inter-action gap.
- **Movement-related accuracy:** transfer path efficiency and attempts; P2 path efficiency, bilateral directional consistency, bilateral path-length difference, signed forward displacement, lateral excursion, hand-separation error, hand-specific and bilateral guidance adherence, and needle-contact events.
- **Work stability:** between-action SD/CV summaries and within-action speed variability, acceleration RMS, jerk RMS, and direction-change rate.

Time-normalized bilateral-midpoint lateral profiles are also generated for descriptive visualization; they are not scored outcomes.

## Run

From this directory, supply one mapping for each pseudonymous session:

```bash
python analyze_vrwce_sessions.py \
  --session S1=/path/to/session_01 \
  --session S2=/path/to/session_02 \
  --session U1=/path/to/session_03 \
  --session U2=/path/to/session_04 \
  --output-dir results
```

Optional `--profile LABEL=TEXT` arguments change display labels without changing calculations. Use `--skip-figures` to produce numerical outputs only. Run `python analyze_vrwce_sessions.py --help` for all options.

## Outputs

The output directory contains:

- `session_summary.csv`
- `p2_action_metrics.csv`
- `p2_hand_action_metrics.csv`
- `transfer_episode_metrics.csv`
- `p2_time_normalized_lateral_profiles.csv`
- `p2_lateral_profile_summary.csv`
- `analysis_qa.json`
- `Fig5_VRWCE_technical_sessions.{svg,pdf,png,tiff}` unless `--skip-figures` is used
- `VRWCE_higher_order_stability.{svg,pdf,png}` unless `--skip-figures` is used

The analyses are descriptive technical demonstrations. They do not produce a composite work-capacity score, a clinical diagnosis, or an employment-decision threshold.
