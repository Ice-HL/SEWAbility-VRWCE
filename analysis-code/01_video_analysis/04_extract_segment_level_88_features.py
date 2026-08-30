import argparse
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from scipy.fft import fft, fftfreq


# ============================================================
# 1. Runtime paths (configured from command-line arguments)
# ============================================================

FPS_CSV = None
VIDEO_DIR = None
OUTPUT_DIR = None


WINDOW_SEC = 1.0
STEP_SEC = 0.5

DEFAULT_FPS = 20.0


JOINTS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

MOTION_JOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def normalize_video_id(x):
    """
    1 -> 1
    1.0 -> 1
    1.mp4 -> 1
    1_trajectory.csv -> 1
    """
    s = str(x).strip()
    s = os.path.basename(s)
    s = s.replace(".mp4", "")
    s = s.replace("_trajectory.csv", "")

    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_fps_map(fps_csv):
    if not os.path.exists(fps_csv):
        raise FileNotFoundError(f"FPS CSV not found: {fps_csv}")

    fps_df = pd.read_csv(fps_csv)

    if "Video" not in fps_df.columns:
        raise ValueError("FPS CSV must contain a 'Video' column.")

    if "FPS" not in fps_df.columns:
        raise ValueError("FPS CSV must contain an 'FPS' column.")

    fps_df["Video_ID"] = fps_df["Video"].apply(normalize_video_id)
    fps_df["FPS"] = fps_df["FPS"].astype(float)

    fps_map = dict(zip(fps_df["Video_ID"], fps_df["FPS"]))

    print(f"✅ Loaded FPS records: {len(fps_map)}")
    print(f"📄 FPS file: {fps_csv}")

    return fps_map


def safe_interpolate_series(x):
    s = pd.Series(x, dtype="float64")

    if s.isna().all():
        return np.zeros(len(s), dtype=float)

    s = s.interpolate(limit_direction="both").bfill().ffill()
    return s.values.astype(float)


def compute_features(x, fps):
    """
    avg_speed, std_speed, amplitude, total_energy, dominant_freq, centroid

    vel = np.diff(x) * fps
    """
    x = safe_interpolate_series(x)

    n = len(x)
    if n < 2:
        return [0, 0, 0, 0, 0, 0]

    vel = np.diff(x) * fps

    avg_speed = np.mean(np.abs(vel))
    std_speed = np.std(vel)
    amplitude = np.max(x) - np.min(x)

    yf = np.abs(fft(x))[:n // 2]
    xf = fftfreq(n, 1 / fps)[:n // 2]

    total_energy = np.sum(yf)

    if len(yf) > 1 and np.any(yf[1:] > 0):
        dominant_freq = xf[np.argmax(yf[1:]) + 1]
    else:
        dominant_freq = 0

    centroid = np.sum(xf * yf) / total_energy if total_energy > 0 else 0

    return [
        avg_speed,
        std_speed,
        amplitude,
        total_energy,
        dominant_freq,
        centroid,
    ]


def compute_joint_angle(p1, p2, p3):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)

    if np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or np.any(np.isnan(p3)):
        return np.nan

    a = p1 - p2
    b = p3 - p2

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-6 or norm_b < 1e-6:
        return np.nan

    cos_angle = np.dot(a, b) / (norm_a * norm_b)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cos_angle))
    return angle


def generate_feature_names():
    feature_names = []

    base_feats = [
        "avg_speed",
        "std_speed",
        "amplitude",
        "total_energy",
        "dominant_freq",
        "centroid",
    ]

    # 6 joints × 2 axes × 6 features = 72
    for joint in MOTION_JOINTS:
        for axis in ["x", "y"]:
            for feat in base_feats:
                feature_names.append(f"{joint}_{feat}_{axis}")

    # 2 sides × 2 angles × 4 features = 16
    for side in ["left", "right"]:
        for joint in ["shoulder", "elbow"]:
            feature_names += [
                f"{side}_{joint}_angle_mean",
                f"{side}_{joint}_angle_std",
                f"{side}_{joint}_angle_change_mean",
                f"{side}_{joint}_angle_change_std",
            ]

    return feature_names


def extract_pose_landmarks_from_video(video_path, expected_frame_count=None):
    """
    bilateral shoulder, elbow, wrist, hip.

    """
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    metadata_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if expected_frame_count is None:
        expected_frame_count = metadata_frame_count if metadata_frame_count > 0 else None

    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_landmarks = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            coords = []

            for joint_name in JOINTS:
                joint_id = JOINTS[joint_name]
                lm = results.pose_landmarks.landmark[joint_id]

                coords.append((lm.x, lm.y))
        else:
            coords = [(np.nan, np.nan)] * len(JOINTS)

        all_landmarks.append(coords)

    cap.release()
    pose.close()

    all_landmarks = np.array(all_landmarks, dtype=float)

    decoded_count = len(all_landmarks)

    # --------------------------------------------------------
    # --------------------------------------------------------
    if expected_frame_count is not None and expected_frame_count > 0:
        if decoded_count != expected_frame_count:
            print(
                f"   ⚠ Decoded frames ({decoded_count}) != expected frames ({expected_frame_count}). "
                f"Resampling to expected frame count."
            )

            if decoded_count > 1 and expected_frame_count > 1:
                selected_indices = np.linspace(
                    0,
                    decoded_count - 1,
                    expected_frame_count,
                ).round().astype(int)

                all_landmarks = all_landmarks[selected_indices]
            elif expected_frame_count == 1:
                all_landmarks = all_landmarks[:1]

    return all_landmarks, metadata_frame_count, decoded_count


def extract_88_features_from_window(window, fps):
    """
    window shape = [window_frames, 8 joints, 2 coords]
    """
    feats = []

    joint_names = list(JOINTS.keys())

    # --------------------------------------------------------
    # --------------------------------------------------------
    for joint in MOTION_JOINTS:
        j_idx = joint_names.index(joint)

        for axis in range(2):
            series = window[:, j_idx, axis]
            feats.extend(compute_features(series, fps))

    # --------------------------------------------------------
    # --------------------------------------------------------
    for side in ["left", "right"]:

        if side == "left":
            shoulder_idx = joint_names.index("left_shoulder")
            elbow_idx = joint_names.index("left_elbow")
            wrist_idx = joint_names.index("left_wrist")
            hip_idx = joint_names.index("left_hip")
        else:
            shoulder_idx = joint_names.index("right_shoulder")
            elbow_idx = joint_names.index("right_elbow")
            wrist_idx = joint_names.index("right_wrist")
            hip_idx = joint_names.index("right_hip")

        shoulder_angles = []
        elbow_angles = []

        for i in range(len(window)):
            shoulder = window[i, shoulder_idx]
            elbow = window[i, elbow_idx]
            wrist = window[i, wrist_idx]
            hip = window[i, hip_idx]

            # shoulder angle = hip - shoulder - elbow
            # This segment-level definition is intentionally preserved from
            # the manuscript analysis. It differs from the video-level
            # elbow-shoulder-opposite-shoulder definition in script 01.
            shoulder_angle = compute_joint_angle(hip, shoulder, elbow)

            # elbow angle = shoulder - elbow - wrist
            elbow_angle = compute_joint_angle(shoulder, elbow, wrist)

            shoulder_angles.append(shoulder_angle)
            elbow_angles.append(elbow_angle)

        for angles in [shoulder_angles, elbow_angles]:
            angles = safe_interpolate_series(angles)

            if len(angles) > 1:
                diffs = np.diff(angles) * fps
            else:
                diffs = np.array([0])

            feats += [
                np.mean(angles),
                np.std(angles),
                np.mean(np.abs(diffs)),
                np.std(diffs),
            ]

    return feats


def process_single_video(video_path, fps):
    video_file = os.path.basename(video_path)
    video_id = normalize_video_id(video_file)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_file}")
        return None

    metadata_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    expected_frame_count = metadata_frame_count if metadata_frame_count > 0 else None

    window_frames = int(round(WINDOW_SEC * fps))
    step_frames = int(round(STEP_SEC * fps))

    window_frames = max(2, window_frames)
    step_frames = max(1, step_frames)

    print(f"\n🎞 Processing: {video_file}")
    print(f"   - FPS: {fps:.3f}")
    print(f"   - Window: {window_frames} frames ({WINDOW_SEC}s)")
    print(f"   - Step: {step_frames} frames ({STEP_SEC}s)")
    print(f"   - Metadata frame count: {metadata_frame_count}")

    all_landmarks, metadata_count, decoded_count = extract_pose_landmarks_from_video(
        video_path=video_path,
        expected_frame_count=expected_frame_count,
    )

    n_frames = len(all_landmarks)

    print(f"   - Decoded frames before resampling: {decoded_count}")
    print(f"   - Frames used for feature extraction: {n_frames}")

    if n_frames < window_frames:
        print(f"⚠️ Video {video_id} has fewer frames than one window. Skipped.")
        return None

    features_all = []
    segment_ids = []
    start_frames = []
    end_frames = []

    for start in range(0, n_frames - window_frames + 1, step_frames):
        end = start + window_frames - 1
        window = all_landmarks[start:start + window_frames]

        feats = extract_88_features_from_window(window, fps)

        features_all.append(feats)
        segment_ids.append(f"segment_{len(segment_ids) + 1:04d}")
        start_frames.append(start)
        end_frames.append(end)

    feature_names = generate_feature_names()

    df = pd.DataFrame(features_all, columns=feature_names)

    df.insert(0, "Video", video_id)
    df.insert(1, "Segment_ID", segment_ids)
    df.insert(2, "Start_Frame", start_frames)
    df.insert(3, "End_Frame", end_frames)
    df.insert(4, "Start_sec", np.array(start_frames) / fps)
    df.insert(5, "End_sec", np.array(end_frames) / fps)
    df.insert(6, "FPS", fps)

    output_csv = os.path.join(OUTPUT_DIR, f"{video_id}_segments.csv")
    df.to_csv(output_csv, index=False)

    print(f"✅ Output saved: {output_csv}")
    print(f"   - Segments: {len(df)}")
    print(f"   - Columns: {df.shape[1]}")

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract overlapping 1-s/0.5-s-step segment-level 88-feature "
            "vectors directly from de-identified videos."
        )
    )
    parser.add_argument("--video-dir", required=True, help="Directory containing MP4 files.")
    parser.add_argument(
        "--fps-csv",
        required=True,
        help="CSV containing de-identified Video and FPS columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/segment_level_88_features",
        help="Output directory (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    global FPS_CSV, VIDEO_DIR, OUTPUT_DIR
    args = parse_args()
    FPS_CSV = os.path.abspath(args.fps_csv)
    VIDEO_DIR = os.path.abspath(args.video_dir)
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fps_map = load_fps_map(FPS_CSV)

    video_files = sorted([
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(".mp4")
    ])

    if len(video_files) == 0:
        print(f"⚠️ No mp4 videos found in: {VIDEO_DIR}")
        return

    print(f"\n📁 Video folder: {VIDEO_DIR}")
    print(f"📁 Output folder: {OUTPUT_DIR}")
    print(f"🎬 Found {len(video_files)} video(s).")

    all_summary = []

    for filename in video_files:
        video_id = normalize_video_id(filename)

        if video_id not in fps_map:
            print(f"\n⚠️ FPS not found for {filename}. Use DEFAULT_FPS={DEFAULT_FPS}.")
            fps = DEFAULT_FPS
        else:
            fps = fps_map[video_id]

        video_path = os.path.join(VIDEO_DIR, filename)

        df_segments = process_single_video(video_path, fps)

        if df_segments is not None:
            all_summary.append({
                "Video": video_id,
                "FPS": fps,
                "Segments": len(df_segments),
                "Output_CSV": os.path.join(OUTPUT_DIR, f"{video_id}_segments.csv"),
            })

    if len(all_summary) > 0:
        summary_df = pd.DataFrame(all_summary)
        summary_path = os.path.join(OUTPUT_DIR, "segment_extraction_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n✅ Summary saved: {summary_path}")

    print(f"\n🎯 Done. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
