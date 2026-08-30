import argparse
import os
import json
import time
import subprocess
import warnings

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import clone
import joblib


warnings.filterwarnings("ignore")


# ============================================================
# 1. Runtime paths (configured from command-line arguments)
# ============================================================

TASK_VIDEO_DIR = None
TRAJ_DIR = None
OUTPUT_DIR = None
MODEL_DIR = None
PREDICTION_DIR = None
BOUNDARY_DIR = None
CYCLE_DIR = None
PLOT_DIR = None
SCREENSHOT_DIR = None


# Manual frame labels are data, not source code. They are loaded from a
# de-identified JSON file supplied with --manual-boundaries-json.
MANUAL_BOUNDARIES = {}


RANDOM_SEED = 42

WINDOW_SEC = 2.0

POSITIVE_AUGMENT_OFFSETS_SEC = [-0.5, 0.0, 0.5]

NEGATIVE_EXCLUSION_SEC = 8.0

NEGATIVE_SAMPLE_STEP_SEC = 3.0

NEGATIVE_RATIO = 4

DETECTION_STEP_SEC = 0.5

SMOOTH_PROB_SEC = 2.0

BASE_PROB_THRESHOLD = 0.55

VIDEO_PROB_QUANTILE = 0.90

MIN_BOUNDARY_GAP_SEC = 30.0

MIN_WORK_CYCLE_SEC = 30.0

EDGE_IGNORE_SEC = 5.0

SAVE_SCREENSHOTS = True

SAVE_PROBABILITY_PLOTS = True


JOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

X_COLS = [f"{j}_x" for j in JOINTS]
Y_COLS = [f"{j}_y" for j in JOINTS]
COORD_COLS = X_COLS + Y_COLS

REQUIRED_COLS = ["frame"] + COORD_COLS


def normalize_video_id(x):
    s = str(x).strip()
    s = os.path.basename(s)

    if s.endswith("_trajectory.csv"):
        s = s.replace("_trajectory.csv", "")

    s = os.path.splitext(s)[0]

    try:
        return str(int(float(s)))
    except Exception:
        return s


def format_time(seconds):
    if seconds is None or pd.isna(seconds):
        return ""

    seconds = float(seconds)
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes}m {sec}s"


def get_ffprobe_duration(video_path):
    if not os.path.exists(video_path):
        return None

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            return None

        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        return duration

    except Exception:
        return None


def get_video_fps(video_id, df):
    """

    """
    video_path = os.path.join(TASK_VIDEO_DIR, f"{video_id}.mp4")

    true_frame_count = int(df["frame"].max()) + 1

    duration = get_ffprobe_duration(video_path)

    if duration is not None and duration > 0:
        fps = true_frame_count / duration
        return fps, duration, "trajectory_frame_count / ffprobe_duration"

    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            if fps is not None and fps > 0:
                duration = true_frame_count / fps
                return fps, duration, "OpenCV CAP_PROP_FPS fallback"

    fps = 20.0
    duration = true_frame_count / fps
    return fps, duration, "default_20fps_fallback"


def read_trajectory_csv(traj_path):
    df = pd.read_csv(traj_path)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {traj_path}: {missing_cols}")

    df = df.copy()
    df["frame"] = df["frame"].astype(int)
    df = df.sort_values("frame").reset_index(drop=True)

    return df


def safe_nan_stat(values, stat_name):
    arr = np.asarray(values, dtype=float)

    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan

    if stat_name == "mean":
        return np.nanmean(arr)
    elif stat_name == "median":
        return np.nanmedian(arr)
    elif stat_name == "std":
        return np.nanstd(arr)
    elif stat_name == "min":
        return np.nanmin(arr)
    elif stat_name == "max":
        return np.nanmax(arr)
    elif stat_name == "range":
        return np.nanmax(arr) - np.nanmin(arr)
    elif stat_name == "q25":
        return np.nanpercentile(arr, 25)
    elif stat_name == "q75":
        return np.nanpercentile(arr, 75)
    else:
        raise ValueError(f"Unknown stat_name: {stat_name}")


def add_basic_stats(features, prefix, values):
    for stat in ["mean", "median", "std", "min", "max", "range", "q25", "q75"]:
        features[f"{prefix}_{stat}"] = safe_nan_stat(values, stat)


def prepare_window(df, center_frame, half_window_frames):
    start_frame = int(center_frame - half_window_frames)
    end_frame = int(center_frame + half_window_frames)

    dfw = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)].copy()

    return dfw


def interpolate_window(dfw):
    df_int = dfw.copy()

    for col in COORD_COLS:
        df_int[col] = df_int[col].interpolate(limit_direction="both")

    return df_int


def compute_joint_speed_features(features, df_int, fps):
    all_speeds = []
    wrist_speeds = []
    shoulder_speeds = []
    elbow_speeds = []

    for joint in JOINTS:
        x = df_int[f"{joint}_x"].to_numpy(dtype=float)
        y = df_int[f"{joint}_y"].to_numpy(dtype=float)

        if len(x) < 2 or np.all(np.isnan(x)) or np.all(np.isnan(y)):
            speeds = np.array([np.nan])
        else:
            dx = np.diff(x)
            dy = np.diff(y)
            speeds = np.sqrt(dx ** 2 + dy ** 2) * fps

        add_basic_stats(features, f"{joint}_speed", speeds)

        all_speeds.extend(list(speeds))

        if "wrist" in joint:
            wrist_speeds.extend(list(speeds))
        elif "shoulder" in joint:
            shoulder_speeds.extend(list(speeds))
        elif "elbow" in joint:
            elbow_speeds.extend(list(speeds))

    add_basic_stats(features, "all_joint_speed", all_speeds)
    add_basic_stats(features, "wrist_speed", wrist_speeds)
    add_basic_stats(features, "shoulder_speed", shoulder_speeds)
    add_basic_stats(features, "elbow_speed", elbow_speeds)


def compute_distance_features(features, df_int):
    pairs = [
        ("wrist_distance", "left_wrist", "right_wrist"),
        ("elbow_distance", "left_elbow", "right_elbow"),
        ("shoulder_distance", "left_shoulder", "right_shoulder"),
        ("left_upper_arm_distance", "left_shoulder", "left_elbow"),
        ("right_upper_arm_distance", "right_shoulder", "right_elbow"),
        ("left_forearm_distance", "left_elbow", "left_wrist"),
        ("right_forearm_distance", "right_elbow", "right_wrist"),
    ]

    for prefix, j1, j2 in pairs:
        x1 = df_int[f"{j1}_x"].to_numpy(dtype=float)
        y1 = df_int[f"{j1}_y"].to_numpy(dtype=float)
        x2 = df_int[f"{j2}_x"].to_numpy(dtype=float)
        y2 = df_int[f"{j2}_y"].to_numpy(dtype=float)

        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        add_basic_stats(features, prefix, dist)


def compute_pre_post_features(features, df_int):
    n = len(df_int)

    if n < 6:
        return

    k = max(2, n // 3)

    first = df_int.iloc[:k].copy()
    last = df_int.iloc[-k:].copy()

    def median_col(data, col):
        return safe_nan_stat(data[col].to_numpy(dtype=float), "median")

    for region_name, joints in {
        "shoulder": ["left_shoulder", "right_shoulder"],
        "elbow": ["left_elbow", "right_elbow"],
        "wrist": ["left_wrist", "right_wrist"],
    }.items():
        first_y_vals = []
        last_y_vals = []

        for j in joints:
            first_y_vals.extend(first[f"{j}_y"].to_numpy(dtype=float))
            last_y_vals.extend(last[f"{j}_y"].to_numpy(dtype=float))

        first_med = safe_nan_stat(first_y_vals, "median")
        last_med = safe_nan_stat(last_y_vals, "median")

        features[f"delta_{region_name}_y_last_minus_first"] = last_med - first_med

    def wrist_dist(data):
        return np.sqrt(
            (data["left_wrist_x"] - data["right_wrist_x"]) ** 2 +
            (data["left_wrist_y"] - data["right_wrist_y"]) ** 2
        )

    first_wd = wrist_dist(first)
    last_wd = wrist_dist(last)

    features["delta_wrist_distance_last_minus_first"] = (
        safe_nan_stat(last_wd, "median") - safe_nan_stat(first_wd, "median")
    )


def extract_window_features(df, center_frame, fps, window_sec=WINDOW_SEC):
    half_window_frames = int(round((window_sec / 2) * fps))
    half_window_frames = max(3, half_window_frames)

    dfw = prepare_window(df, center_frame, half_window_frames)

    features = {
        "center_frame": int(center_frame),
        "window_start_frame": int(center_frame - half_window_frames),
        "window_end_frame": int(center_frame + half_window_frames),
        "window_n_rows": len(dfw),
    }

    if dfw.empty:
        for col in COORD_COLS:
            features[f"{col}_median"] = np.nan
        return features

    # --------------------------------------------------------
    # --------------------------------------------------------
    coord_missing_ratio = dfw[COORD_COLS].isna().mean().mean()
    row_any_missing_ratio = dfw[COORD_COLS].isna().any(axis=1).mean()

    features["coord_missing_ratio"] = coord_missing_ratio
    features["row_any_missing_ratio"] = row_any_missing_ratio

    visible_joint_counts = []

    for _, row in dfw.iterrows():
        count = 0
        for joint in JOINTS:
            if not pd.isna(row[f"{joint}_x"]) and not pd.isna(row[f"{joint}_y"]):
                count += 1
        visible_joint_counts.append(count)

    add_basic_stats(features, "visible_joint_count", visible_joint_counts)

    left_cols = [
        "left_shoulder_x", "left_shoulder_y",
        "left_elbow_x", "left_elbow_y",
        "left_wrist_x", "left_wrist_y",
    ]
    right_cols = [
        "right_shoulder_x", "right_shoulder_y",
        "right_elbow_x", "right_elbow_y",
        "right_wrist_x", "right_wrist_y",
    ]

    features["left_limb_all_missing_ratio"] = dfw[left_cols].isna().all(axis=1).mean()
    features["right_limb_all_missing_ratio"] = dfw[right_cols].isna().all(axis=1).mean()

    # --------------------------------------------------------
    # --------------------------------------------------------
    df_int = interpolate_window(dfw)

    # --------------------------------------------------------
    # --------------------------------------------------------
    for col in COORD_COLS:
        add_basic_stats(features, col, df_int[col].to_numpy(dtype=float))

    shoulder_y = pd.concat([
        df_int["left_shoulder_y"],
        df_int["right_shoulder_y"],
    ], axis=0).to_numpy(dtype=float)

    elbow_y = pd.concat([
        df_int["left_elbow_y"],
        df_int["right_elbow_y"],
    ], axis=0).to_numpy(dtype=float)

    wrist_y = pd.concat([
        df_int["left_wrist_y"],
        df_int["right_wrist_y"],
    ], axis=0).to_numpy(dtype=float)

    all_y = df_int[Y_COLS].to_numpy(dtype=float).flatten()
    all_x = df_int[X_COLS].to_numpy(dtype=float).flatten()

    add_basic_stats(features, "shoulder_y", shoulder_y)
    add_basic_stats(features, "elbow_y", elbow_y)
    add_basic_stats(features, "wrist_y", wrist_y)
    add_basic_stats(features, "all_y", all_y)
    add_basic_stats(features, "all_x", all_x)

    y_ranges = []
    x_ranges = []

    for joint in JOINTS:
        y_ranges.append(
            safe_nan_stat(df_int[f"{joint}_y"].to_numpy(dtype=float), "range")
        )
        x_ranges.append(
            safe_nan_stat(df_int[f"{joint}_x"].to_numpy(dtype=float), "range")
        )

    features["y_all_range_mean"] = safe_nan_stat(y_ranges, "mean")
    features["x_all_range_mean"] = safe_nan_stat(x_ranges, "mean")
    features["y_all_range_max"] = safe_nan_stat(y_ranges, "max")
    features["x_all_range_max"] = safe_nan_stat(x_ranges, "max")

    # --------------------------------------------------------
    # --------------------------------------------------------
    compute_distance_features(features, df_int)

    # --------------------------------------------------------
    # --------------------------------------------------------
    compute_joint_speed_features(features, df_int, fps)

    # --------------------------------------------------------
    # --------------------------------------------------------
    compute_pre_post_features(features, df_int)

    return features


def build_training_dataset():
    print("=" * 90)
    print("Building training dataset from manual cloth-change annotations")
    print("=" * 90)

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    for video_id, manual_frames in MANUAL_BOUNDARIES.items():
        traj_path = os.path.join(TRAJ_DIR, f"{video_id}_trajectory.csv")

        if not os.path.exists(traj_path):
            print(f"⚠ Missing trajectory CSV for annotated video {video_id}: {traj_path}")
            continue

        df = read_trajectory_csv(traj_path)
        fps, duration, fps_source = get_video_fps(video_id, df)

        n_frames = int(df["frame"].max()) + 1
        half_window_frames = int(round((WINDOW_SEC / 2) * fps))

        print(f"\n📌 Video {video_id}.mp4")
        print(f"   - Trajectory rows: {len(df)}")
        print(f"   - FPS: {fps:.3f} ({fps_source})")
        print(f"   - Duration: {duration:.3f} sec")
        print(f"   - Manual positive frames: {manual_frames}")

        # ----------------------------------------------------
        # ----------------------------------------------------
        positive_centers = set()

        for f in manual_frames:
            for offset_sec in POSITIVE_AUGMENT_OFFSETS_SEC:
                center = int(round(f + offset_sec * fps))

                if center < half_window_frames:
                    continue
                if center > n_frames - half_window_frames - 1:
                    continue

                positive_centers.add(center)

        for center in sorted(positive_centers):
            feats = extract_window_features(df, center, fps, WINDOW_SEC)
            feats["Video"] = video_id
            feats["label"] = 1
            feats["sample_type"] = "positive"
            feats["fps"] = fps
            feats["center_time_sec"] = center / fps
            rows.append(feats)

        # ----------------------------------------------------
        # ----------------------------------------------------
        exclusion_frames = int(round(NEGATIVE_EXCLUSION_SEC * fps))
        neg_step_frames = int(round(NEGATIVE_SAMPLE_STEP_SEC * fps))
        neg_step_frames = max(1, neg_step_frames)

        candidate_negatives = []

        for center in range(half_window_frames, n_frames - half_window_frames, neg_step_frames):
            min_dist = min([abs(center - mf) for mf in manual_frames])
            if min_dist > exclusion_frames:
                candidate_negatives.append(center)

        n_positive = len(positive_centers)
        n_negative_target = min(len(candidate_negatives), max(1, n_positive * NEGATIVE_RATIO))

        if len(candidate_negatives) > 0:
            selected_negatives = rng.choice(
                candidate_negatives,
                size=n_negative_target,
                replace=False,
            )
        else:
            selected_negatives = []

        for center in sorted(selected_negatives):
            feats = extract_window_features(df, int(center), fps, WINDOW_SEC)
            feats["Video"] = video_id
            feats["label"] = 0
            feats["sample_type"] = "negative"
            feats["fps"] = fps
            feats["center_time_sec"] = int(center) / fps
            rows.append(feats)

        print(f"   - Positive samples after augmentation: {len(positive_centers)}")
        print(f"   - Negative samples: {len(selected_negatives)}")

    if len(rows) == 0:
        raise RuntimeError("No training samples were created. Please check paths and manual labels.")

    dataset = pd.DataFrame(rows)

    out_path = os.path.join(MODEL_DIR, "training_feature_dataset.csv")
    dataset.to_csv(out_path, index=False)

    print("\n✅ Training dataset saved:")
    print(f"   {out_path}")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Positive samples: {(dataset['label'] == 1).sum()}")
    print(f"   Negative samples: {(dataset['label'] == 0).sum()}")

    return dataset


def get_feature_columns(dataset):
    non_feature_cols = {
        "Video",
        "label",
        "sample_type",
        "center_frame",
        "window_start_frame",
        "window_end_frame",
        "center_time_sec",
        "fps",
    }

    feature_cols = []

    for col in dataset.columns:
        if col in non_feature_cols:
            continue

        if pd.api.types.is_numeric_dtype(dataset[col]):
            feature_cols.append(col)

    return feature_cols


def train_model(dataset, feature_cols):
    X = dataset[feature_cols].copy()
    y = dataset["label"].astype(int).copy()
    groups = dataset["Video"].astype(str).copy()

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Leave-one-video-out cross validation
    # --------------------------------------------------------
    unique_groups = sorted(groups.unique())

    cv_report_path = os.path.join(MODEL_DIR, "leave_one_video_out_cv_report.txt")
    cv_pred_path = os.path.join(MODEL_DIR, "leave_one_video_out_predictions.csv")

    cv_rows = []

    if len(unique_groups) >= 2:
        logo = LeaveOneGroupOut()

        y_true_all = []
        y_pred_all = []
        y_prob_all = []

        for train_idx, test_idx in logo.split(X, y, groups):
            train_groups = groups.iloc[train_idx].unique()
            test_group = groups.iloc[test_idx].unique()[0]

            fold_model = clone(model)
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])

            prob = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)

            for i, idx in enumerate(test_idx):
                cv_rows.append({
                    "Test_Video": test_group,
                    "Sample_Index": int(idx),
                    "True_Label": int(y.iloc[idx]),
                    "Pred_Label": int(pred[i]),
                    "Pred_Prob": float(prob[i]),
                    "Center_Frame": int(dataset.iloc[idx]["center_frame"]),
                    "Center_Time_sec": float(dataset.iloc[idx]["center_time_sec"]),
                    "Sample_Type": dataset.iloc[idx]["sample_type"],
                })

            y_true_all.extend(list(y.iloc[test_idx]))
            y_pred_all.extend(list(pred))
            y_prob_all.extend(list(prob))

        cv_df = pd.DataFrame(cv_rows)
        cv_df.to_csv(cv_pred_path, index=False)

        report = classification_report(
            y_true_all,
            y_pred_all,
            labels=[0, 1],
            target_names=["non_boundary", "cloth_change_boundary"],
            zero_division=0,
        )

        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

        with open(cv_report_path, "w", encoding="utf-8") as f:
            f.write("Leave-one-video-out cross validation\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
            f.write("\n\nConfusion matrix labels: [0, 1]\n")
            f.write(str(cm))
            f.write("\n")

        print("\n✅ Cross-validation report saved:")
        print(f"   {cv_report_path}")
        print(f"   {cv_pred_path}")

    else:
        print("⚠ Not enough videos for leave-one-video-out cross validation.")

    # --------------------------------------------------------
    # --------------------------------------------------------
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, "cloth_change_boundary_random_forest.joblib")
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "manual_boundaries": MANUAL_BOUNDARIES,
            "window_sec": WINDOW_SEC,
            "detection_step_sec": DETECTION_STEP_SEC,
        },
        model_path,
    )

    print("\n✅ Final model saved:")
    print(f"   {model_path}")

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------
    rf = model.named_steps["rf"]
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_path = os.path.join(MODEL_DIR, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)

    print("\n✅ Feature importance saved:")
    print(f"   {importance_path}")
    print("\nTop 15 important features:")
    print(importance_df.head(15).to_string(index=False))

    return model


def rolling_median(values, window_size):
    s = pd.Series(values)
    return s.rolling(window_size, center=True, min_periods=1).median().to_numpy()


def find_probability_peaks(pred_df, fps):
    if pred_df.empty:
        return []

    p = pred_df["prob_smooth"].to_numpy(dtype=float)
    frames = pred_df["center_frame"].to_numpy(dtype=int)
    times = pred_df["center_time_sec"].to_numpy(dtype=float)

    threshold_quantile = safe_nan_stat(p, "q75")
    adaptive_threshold = np.nanquantile(p, VIDEO_PROB_QUANTILE)
    threshold = max(BASE_PROB_THRESHOLD, adaptive_threshold)

    min_gap_frames = int(round(MIN_BOUNDARY_GAP_SEC * fps))
    edge_ignore_frames = int(round(EDGE_IGNORE_SEC * fps))

    candidates = []

    for i in range(1, len(p) - 1):
        if frames[i] < edge_ignore_frames:
            continue
        if frames[i] > frames[-1] - edge_ignore_frames:
            continue

        is_local_peak = p[i] >= p[i - 1] and p[i] >= p[i + 1]

        if is_local_peak and p[i] >= threshold:
            candidates.append({
                "center_frame": int(frames[i]),
                "time_sec": float(times[i]),
                "probability": float(p[i]),
                "threshold": float(threshold),
            })

    if len(candidates) == 0:
        max_idx = int(np.nanargmax(p))
        if p[max_idx] >= BASE_PROB_THRESHOLD:
            candidates.append({
                "center_frame": int(frames[max_idx]),
                "time_sec": float(times[max_idx]),
                "probability": float(p[max_idx]),
                "threshold": float(threshold),
            })

    candidates = sorted(candidates, key=lambda x: x["probability"], reverse=True)

    selected = []

    for cand in candidates:
        keep = True

        for kept in selected:
            if abs(cand["center_frame"] - kept["center_frame"]) < min_gap_frames:
                keep = False
                break

        if keep:
            selected.append(cand)

    selected = sorted(selected, key=lambda x: x["center_frame"])

    for i, b in enumerate(selected, start=1):
        b["Boundary_ID"] = i

    return selected


def save_boundary_screenshot(video_path, time_sec, out_path):
    if not os.path.exists(video_path):
        return False

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
    ok, frame = cap.read()

    if ok:
        cv2.imwrite(out_path, frame)

    cap.release()

    return bool(ok)


def plot_probability_curve(video_id, pred_df, boundaries, manual_frames, fps):
    if pred_df.empty:
        return

    plt.figure(figsize=(14, 5))

    x_min = pred_df["center_time_sec"] / 60.0

    plt.plot(x_min, pred_df["probability"], alpha=0.35, label="Raw probability")
    plt.plot(x_min, pred_df["prob_smooth"], linewidth=2, label="Smoothed probability")

    for b in boundaries:
        plt.axvline(
            b["time_sec"] / 60.0,
            linestyle="--",
            linewidth=1.5,
            label="Predicted boundary" if b["Boundary_ID"] == 1 else None,
        )

    if manual_frames is not None:
        for i, mf in enumerate(manual_frames):
            plt.axvline(
                (mf / fps) / 60.0,
                linestyle=":",
                linewidth=1.5,
                label="Manual boundary" if i == 0 else None,
            )

    plt.xlabel("Time (min)")
    plt.ylabel("Boundary probability")
    plt.title(f"Cloth-change boundary probability: {video_id}.mp4")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"{video_id}_boundary_probability.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def detect_boundaries_for_video(video_id, model, feature_cols):
    traj_path = os.path.join(TRAJ_DIR, f"{video_id}_trajectory.csv")

    if not os.path.exists(traj_path):
        print(f"⚠ Missing trajectory CSV: {traj_path}")
        return None, None, None

    df = read_trajectory_csv(traj_path)
    fps, duration, fps_source = get_video_fps(video_id, df)

    n_frames = int(df["frame"].max()) + 1
    half_window_frames = int(round((WINDOW_SEC / 2) * fps))
    step_frames = max(1, int(round(DETECTION_STEP_SEC * fps)))

    print(f"\n📂 Detecting {video_id}.mp4")
    print(f"   - Trajectory: {traj_path}")
    print(f"   - Rows: {len(df)}")
    print(f"   - FPS: {fps:.3f} ({fps_source})")
    print(f"   - Duration: {format_time(duration)}")
    print(f"   - Sliding step: {step_frames} frames ({DETECTION_STEP_SEC}s)")

    centers = list(range(
        half_window_frames,
        n_frames - half_window_frames,
        step_frames,
    ))

    feature_rows = []

    for center in centers:
        feats = extract_window_features(df, center, fps, WINDOW_SEC)
        feats["Video"] = video_id
        feats["center_time_sec"] = center / fps
        feature_rows.append(feats)

    if len(feature_rows) == 0:
        print("   ⚠ No sliding-window features generated.")
        return None, None, None

    pred_feature_df = pd.DataFrame(feature_rows)

    for col in feature_cols:
        if col not in pred_feature_df.columns:
            pred_feature_df[col] = np.nan

    X_pred = pred_feature_df[feature_cols].copy()

    probability = model.predict_proba(X_pred)[:, 1]

    smooth_window_points = max(3, int(round(SMOOTH_PROB_SEC / DETECTION_STEP_SEC)))
    prob_smooth = rolling_median(probability, smooth_window_points)

    pred_df = pd.DataFrame({
        "Video": video_id,
        "center_frame": pred_feature_df["center_frame"].astype(int),
        "center_time_sec": pred_feature_df["center_time_sec"].astype(float),
        "probability": probability,
        "prob_smooth": prob_smooth,
        "FPS": fps,
    })

    pred_csv_path = os.path.join(PREDICTION_DIR, f"{video_id}_boundary_probability.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    boundaries = find_probability_peaks(pred_df, fps)

    boundary_rows = []

    for b in boundaries:
        boundary_rows.append({
            "Video": f"{video_id}.mp4",
            "Boundary_ID": b["Boundary_ID"],
            "Boundary_Frame": int(b["center_frame"]),
            "Boundary_Time_sec": round(b["time_sec"], 3),
            "Boundary_Time": format_time(b["time_sec"]),
            "Boundary_Probability": round(b["probability"], 4),
            "Threshold_Used": round(b["threshold"], 4),
            "FPS": round(fps, 6),
        })

    boundary_df = pd.DataFrame(boundary_rows)

    boundary_csv_path = os.path.join(BOUNDARY_DIR, f"{video_id}_predicted_boundaries.csv")
    boundary_df.to_csv(boundary_csv_path, index=False)

    print(f"   ✅ Predicted boundaries: {len(boundary_df)}")
    print(f"   ✅ Saved probability curve CSV: {pred_csv_path}")
    print(f"   ✅ Saved boundary CSV: {boundary_csv_path}")

    if SAVE_SCREENSHOTS and len(boundary_rows) > 0:
        video_path = os.path.join(TASK_VIDEO_DIR, f"{video_id}.mp4")
        video_shot_dir = os.path.join(SCREENSHOT_DIR, video_id)
        os.makedirs(video_shot_dir, exist_ok=True)

        for row in boundary_rows:
            shot_path = os.path.join(
                video_shot_dir,
                f"boundary_{row['Boundary_ID']}_frame_{row['Boundary_Frame']}.jpg",
            )
            save_boundary_screenshot(
                video_path=video_path,
                time_sec=row["Boundary_Time_sec"],
                out_path=shot_path,
            )

    if SAVE_PROBABILITY_PLOTS:
        manual_frames = MANUAL_BOUNDARIES.get(video_id, None)
        plot_probability_curve(
            video_id=video_id,
            pred_df=pred_df,
            boundaries=boundaries,
            manual_frames=manual_frames,
            fps=fps,
        )

    cycle_rows = []

    if len(boundary_rows) >= 2:
        for i in range(len(boundary_rows) - 1):
            b1 = boundary_rows[i]
            b2 = boundary_rows[i + 1]

            start_sec = b1["Boundary_Time_sec"]
            end_sec = b2["Boundary_Time_sec"]
            duration_sec = end_sec - start_sec

            if duration_sec < MIN_WORK_CYCLE_SEC:
                continue

            cycle_rows.append({
                "Video": f"{video_id}.mp4",
                "Work_Cycle_ID": len(cycle_rows) + 1,
                "Start_Frame": int(b1["Boundary_Frame"]),
                "End_Frame": int(b2["Boundary_Frame"]),
                "Start_sec": round(start_sec, 3),
                "End_sec": round(end_sec, 3),
                "Duration_sec": round(duration_sec, 3),
                "Start_Time": format_time(start_sec),
                "End_Time": format_time(end_sec),
                "Duration": format_time(duration_sec),
                "Start_Boundary_Probability": b1["Boundary_Probability"],
                "End_Boundary_Probability": b2["Boundary_Probability"],
                "FPS": round(fps, 6),
            })

    cycle_df = pd.DataFrame(cycle_rows)

    cycle_csv_path = os.path.join(CYCLE_DIR, f"{video_id}_work_cycles.csv")
    cycle_df.to_csv(cycle_csv_path, index=False)

    if len(cycle_df) > 0:
        print(f"   ✅ Saved work cycles: {cycle_csv_path}")
    else:
        print("   ⚠ No valid work cycles generated.")

    return pred_df, boundary_df, cycle_df


def compare_predictions_with_manual(all_boundary_df):
    rows = []

    if all_boundary_df is None or all_boundary_df.empty:
        return pd.DataFrame()

    for video_id, manual_frames in MANUAL_BOUNDARIES.items():
        sub = all_boundary_df[all_boundary_df["Video"] == f"{video_id}.mp4"].copy()

        if sub.empty:
            continue

        pred_frames = sub["Boundary_Frame"].astype(int).tolist()
        fps = float(sub["FPS"].iloc[0])

        for mf in manual_frames:
            if len(pred_frames) == 0:
                rows.append({
                    "Video": f"{video_id}.mp4",
                    "Manual_Frame": mf,
                    "Nearest_Predicted_Frame": None,
                    "Frame_Error": None,
                    "Time_Error_sec": None,
                    "FPS": fps,
                })
                continue

            nearest = min(pred_frames, key=lambda x: abs(x - mf))
            frame_error = nearest - mf
            time_error = frame_error / fps

            rows.append({
                "Video": f"{video_id}.mp4",
                "Manual_Frame": int(mf),
                "Manual_Time": format_time(mf / fps),
                "Nearest_Predicted_Frame": int(nearest),
                "Nearest_Predicted_Time": format_time(nearest / fps),
                "Frame_Error": int(frame_error),
                "Abs_Frame_Error": int(abs(frame_error)),
                "Time_Error_sec": round(time_error, 3),
                "Abs_Time_Error_sec": round(abs(time_error), 3),
                "FPS": round(fps, 6),
            })

    compare_df = pd.DataFrame(rows)

    if not compare_df.empty:
        out_path = os.path.join(OUTPUT_DIR, "manual_vs_predicted_boundary_error.csv")
        compare_df.to_csv(out_path, index=False)

        summary_path = os.path.join(OUTPUT_DIR, "manual_vs_predicted_boundary_error_summary.csv")
        summary_df = compare_df.groupby("Video").agg(
            Manual_Boundary_Count=("Manual_Frame", "count"),
            Mean_Abs_Frame_Error=("Abs_Frame_Error", "mean"),
            Median_Abs_Frame_Error=("Abs_Frame_Error", "median"),
            Mean_Abs_Time_Error_sec=("Abs_Time_Error_sec", "mean"),
            Median_Abs_Time_Error_sec=("Abs_Time_Error_sec", "median"),
        ).reset_index()

        summary_df.to_csv(summary_path, index=False)

        print("\n✅ Manual vs predicted comparison saved:")
        print(f"   {out_path}")
        print(f"   {summary_path}")

    return compare_df


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the study-specific random-forest boundary detector and "
            "derive candidate work-cycle intervals."
        )
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing de-identified MP4 files named <video_id>.mp4.",
    )
    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Directory containing <video_id>_trajectory.csv files.",
    )
    parser.add_argument(
        "--manual-boundaries-json",
        required=True,
        help=(
            "JSON object mapping de-identified video IDs to lists of manually "
            "annotated cloth-change frame numbers."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/work_cycle_boundaries",
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="Do not save boundary screenshots.",
    )
    parser.add_argument(
        "--no-probability-plots",
        action="store_true",
        help="Do not save probability-curve plots.",
    )
    return parser.parse_args()


def configure_runtime(args):
    global TASK_VIDEO_DIR, TRAJ_DIR, OUTPUT_DIR
    global MODEL_DIR, PREDICTION_DIR, BOUNDARY_DIR, CYCLE_DIR, PLOT_DIR, SCREENSHOT_DIR
    global MANUAL_BOUNDARIES, SAVE_SCREENSHOTS, SAVE_PROBABILITY_PLOTS

    TASK_VIDEO_DIR = os.path.abspath(args.video_dir)
    TRAJ_DIR = os.path.abspath(args.trajectory_dir)
    OUTPUT_DIR = os.path.abspath(args.output_dir)

    MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
    PREDICTION_DIR = os.path.join(OUTPUT_DIR, "prediction_curves")
    BOUNDARY_DIR = os.path.join(OUTPUT_DIR, "boundaries")
    CYCLE_DIR = os.path.join(OUTPUT_DIR, "work_cycles")
    PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
    SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "screenshots")
    for directory in [
        MODEL_DIR,
        PREDICTION_DIR,
        BOUNDARY_DIR,
        CYCLE_DIR,
        PLOT_DIR,
        SCREENSHOT_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)

    with open(args.manual_boundaries_json, "r", encoding="utf-8") as handle:
        raw_boundaries = json.load(handle)
    if not isinstance(raw_boundaries, dict) or not raw_boundaries:
        raise ValueError("The manual-boundaries JSON must be a non-empty object.")
    MANUAL_BOUNDARIES = {
        normalize_video_id(video_id): [int(frame) for frame in frames]
        for video_id, frames in raw_boundaries.items()
    }

    SAVE_SCREENSHOTS = not args.no_screenshots
    SAVE_PROBABILITY_PLOTS = not args.no_probability_plots


def main():
    args = parse_args()
    configure_runtime(args)
    start_time = time.time()

    print("=" * 90)
    print("03_detect_work_cycle_boundaries.py")
    print("Supervised cloth-change boundary detection for work-cycle segmentation")
    print("=" * 90)
    print(f"Task video folder: {TASK_VIDEO_DIR}")
    print(f"Trajectory CSV folder: {TRAJ_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")

    if not os.path.exists(TRAJ_DIR):
        raise FileNotFoundError(f"Trajectory folder does not exist: {TRAJ_DIR}")

    # ========================================================
    # ========================================================
    dataset = build_training_dataset()
    feature_cols = get_feature_columns(dataset)

    feature_cols_path = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_cols_path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(col + "\n")

    print(f"\n✅ Number of feature columns: {len(feature_cols)}")
    print(f"✅ Feature column list saved: {feature_cols_path}")

    # ========================================================
    # ========================================================
    model = train_model(dataset, feature_cols)

    # ========================================================
    # ========================================================
    traj_files = sorted([
        f for f in os.listdir(TRAJ_DIR)
        if f.lower().endswith("_trajectory.csv")
    ])

    if len(traj_files) == 0:
        raise RuntimeError(f"No trajectory CSV found in: {TRAJ_DIR}")

    print("\n" + "=" * 90)
    print("Detecting cloth-change boundaries for all trajectory CSV files")
    print("=" * 90)
    print(f"Found {len(traj_files)} trajectory CSV files.")

    all_boundary_dfs = []
    all_cycle_dfs = []

    for traj_file in traj_files:
        video_id = normalize_video_id(traj_file)

        try:
            pred_df, boundary_df, cycle_df = detect_boundaries_for_video(
                video_id=video_id,
                model=model,
                feature_cols=feature_cols,
            )

            if boundary_df is not None and not boundary_df.empty:
                all_boundary_dfs.append(boundary_df)

            if cycle_df is not None and not cycle_df.empty:
                all_cycle_dfs.append(cycle_df)

        except Exception as e:
            print(f"❌ Error processing {traj_file}: {e}")

    # ========================================================
    # ========================================================
    if len(all_boundary_dfs) > 0:
        all_boundary_df = pd.concat(all_boundary_dfs, ignore_index=True)
    else:
        all_boundary_df = pd.DataFrame()

    if len(all_cycle_dfs) > 0:
        all_cycle_df = pd.concat(all_cycle_dfs, ignore_index=True)
    else:
        all_cycle_df = pd.DataFrame()

    all_boundary_path = os.path.join(OUTPUT_DIR, "TaskA_all_predicted_boundaries.csv")
    all_cycle_path = os.path.join(OUTPUT_DIR, "TaskA_all_work_cycles.csv")

    all_boundary_df.to_csv(all_boundary_path, index=False)
    all_cycle_df.to_csv(all_cycle_path, index=False)

    print("\n✅ Master outputs saved:")
    print(f"   {all_boundary_path}")
    print(f"   {all_cycle_path}")

    # ========================================================
    # ========================================================
    compare_predictions_with_manual(all_boundary_df)

    elapsed = time.time() - start_time

    print("\n" + "=" * 90)
    print("🎯 Done.")
    print(f"Total time: {format_time(elapsed)}")
    print(f"All outputs are in: {OUTPUT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()
