import argparse
import os
import time
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_FPS = 20.0

MAX_WORKERS = 10

JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist"
]


def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def normalize_name(name):
    """
    1.mp4 -> 1
    1_trajectory.csv -> 1
    """
    name = str(name).strip()
    name = os.path.basename(name)
    stem, _ = os.path.splitext(name)

    if stem.endswith("_trajectory"):
        stem = stem[:-11]

    return stem.lower()


def load_fps_mapping(fps_csv_path):
    if not os.path.exists(fps_csv_path):
        raise FileNotFoundError(f"FPS CSV not found: {fps_csv_path}")

    df_fps = pd.read_csv(fps_csv_path)

    required_cols = {"Filename", "FPS"}
    missing = required_cols - set(df_fps.columns)
    if missing:
        raise ValueError(f"FPS CSV missing required column(s): {missing}")

    fps_map = {}

    for _, row in df_fps.iterrows():
        filename = row["Filename"]
        fps = row["FPS"]

        if pd.isna(filename) or pd.isna(fps):
            continue

        key = normalize_name(filename)
        fps_value = float(fps)

        if fps_value > 0:
            fps_map[key] = fps_value

    if not fps_map:
        raise ValueError("No valid FPS records were loaded from the FPS CSV.")

    return fps_map


def extract_features_from_axis(axis, fps):
    axis = np.array(axis, dtype=float)

    s = pd.Series(axis).interpolate(limit_direction="both").bfill().ffill()
    axis = s.values

    n = len(axis)
    if n < 2:
        return [0, 0, 0, 0, 0, 0]

    vel = np.diff(axis) * fps
    avg_speed = np.mean(np.abs(vel))
    std_speed = np.std(vel)
    amplitude = np.max(axis) - np.min(axis)

    yf = np.abs(fft(axis))[:n // 2]
    xf = fftfreq(n, 1 / fps)[:n // 2]

    total_energy = np.sum(yf)

    if len(yf) > 1 and np.any(yf[1:] > 0):
        dominant_freq = xf[np.argmax(yf[1:]) + 1]
    else:
        dominant_freq = 0

    centroid = np.sum(xf * yf) / total_energy if total_energy > 0 else 0

    return [avg_speed, std_speed, amplitude, total_energy, dominant_freq, centroid]


def compute_joint_angle(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return np.nan

    cos_angle = np.dot(a, b) / (norm_a * norm_b)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def extract_joint_angle_stats(df, fps):
    """
    shoulder angle = elbow - shoulder - opposite shoulder
    elbow angle    = shoulder - elbow - wrist

    This video-level definition is intentionally preserved from the analysis
    used for the manuscript. It differs from the hip-shoulder-elbow definition
    in 04_extract_segment_level_88_features.py; the two feature matrices must
    therefore not be substituted for one another.

    """
    stats_features = []

    for side in ["left", "right"]:
        if side == "left":
            shoulder = df[["left_shoulder_x", "left_shoulder_y"]].values
            elbow = df[["left_elbow_x", "left_elbow_y"]].values
            wrist = df[["left_wrist_x", "left_wrist_y"]].values
            opposite_shoulder = df[["right_shoulder_x", "right_shoulder_y"]].values
        else:
            shoulder = df[["right_shoulder_x", "right_shoulder_y"]].values
            elbow = df[["right_elbow_x", "right_elbow_y"]].values
            wrist = df[["right_wrist_x", "right_wrist_y"]].values
            opposite_shoulder = df[["left_shoulder_x", "left_shoulder_y"]].values

        shoulder_angles = []
        elbow_angles = []

        for i in range(len(df)):
            shoulder_ang = compute_joint_angle(elbow[i], shoulder[i], opposite_shoulder[i])
            elbow_ang = compute_joint_angle(shoulder[i], elbow[i], wrist[i])

            shoulder_angles.append(shoulder_ang)
            elbow_angles.append(elbow_ang)

        for angles in [shoulder_angles, elbow_angles]:
            angles = pd.Series(angles).interpolate(limit_direction="both").bfill().ffill().values

            diffs = np.diff(angles) * fps

            stats_features += [
                np.mean(angles),
                np.std(angles),
                np.mean(np.abs(diffs)) if len(diffs) > 0 else 0,
                np.std(diffs) if len(diffs) > 0 else 0
            ]

    return stats_features


def generate_feature_names():
    names = []
    base_features = ["avg_speed", "std_speed", "amplitude", "total_energy", "dominant_freq", "centroid"]

    for joint in JOINTS:
        for axis in ["x", "y"]:
            for feat in base_features:
                names.append(f"{joint}_{feat}_{axis}")

    for side in ["left", "right"]:
        for joint in ["shoulder", "elbow"]:
            names += [
                f"{side}_{joint}_angle_mean",
                f"{side}_{joint}_angle_std",
                f"{side}_{joint}_angle_change_mean",
                f"{side}_{joint}_angle_change_std"
            ]

    return names


def extract_features_from_trajectory_csv_worker(csv_path, fps):
    start_time = time.time()
    file_name = os.path.basename(csv_path)

    try:
        df = pd.read_csv(csv_path)

        all_features = []

        for joint in JOINTS:
            x = df[f"{joint}_x"].values
            y = df[f"{joint}_y"].values

            all_features.extend(extract_features_from_axis(x, fps))
            all_features.extend(extract_features_from_axis(y, fps))

        all_features.extend(extract_joint_angle_stats(df, fps))

        elapsed = time.time() - start_time

        return {
            "status": "ok",
            "file": file_name,
            "video": file_name.replace("_trajectory.csv", ""),
            "fps": fps,
            "features": np.array(all_features),
            "time_sec": elapsed
        }

    except Exception as e:
        return {
            "status": "error",
            "file": file_name,
            "fps": fps,
            "message": str(e)
        }


def process_all_trajectory_csvs_parallel(
    folder,
    fps_map,
    output_csv,
    fps_csv_path,
    default_fps=DEFAULT_FPS,
    max_workers=MAX_WORKERS,
):
    files = sorted([f for f in os.listdir(folder) if f.endswith("_trajectory.csv")])

    if not files:
        print("⚠️ No trajectory CSV files found.")
        return

    total_tasks = len(files)
    completed_tasks = 0
    successful_times = []
    results = []

    print(f"📁 Found {total_tasks} trajectory CSV file(s).")
    print(f"📄 FPS table loaded from: {fps_csv_path}")
    print(f"🚀 Start parallel feature extraction with MAX_WORKERS = {max_workers}")

    total_start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for file in files:
            csv_path = os.path.join(folder, file)
            key = normalize_name(file)
            fps = fps_map.get(key, default_fps)

            if key not in fps_map:
                print(f"⚠️ FPS not found for {file}, fallback to DEFAULT_FPS = {default_fps}")

            future = executor.submit(extract_features_from_trajectory_csv_worker, csv_path, fps)
            futures[future] = file

        for future in as_completed(futures):
            completed_tasks += 1
            progress = completed_tasks / total_tasks * 100
            remaining_tasks = total_tasks - completed_tasks

            try:
                result = future.result()

                if result["status"] == "ok":
                    results.append(result)
                    successful_times.append(result["time_sec"])

                    avg_time = sum(successful_times) / len(successful_times)
                    eta_seconds = avg_time * remaining_tasks

                    print(
                        f"✅ [{completed_tasks}/{total_tasks} | {progress:5.1f}%] "
                        f"{result['file']} done | "
                        f"FPS: {result['fps']:.3f} | "
                        f"Time: {result['time_sec']:.1f}s | "
                        f"ETA: {format_seconds(eta_seconds)}"
                    )
                else:
                    eta_text = format_seconds((sum(successful_times) / len(successful_times)) * remaining_tasks) if successful_times else "unknown"
                    print(
                        f"❌ [{completed_tasks}/{total_tasks} | {progress:5.1f}%] "
                        f"{result['file']} failed | "
                        f"FPS: {result['fps']:.3f} | "
                        f"{result['message']} | ETA: {eta_text}"
                    )

            except Exception as e:
                eta_text = format_seconds((sum(successful_times) / len(successful_times)) * remaining_tasks) if successful_times else "unknown"
                print(
                    f"❌ [{completed_tasks}/{total_tasks} | {progress:5.1f}%] "
                    f"Unexpected error: {e} | ETA: {eta_text}"
                )

    if not results:
        print("❌ No valid feature results were produced.")
        return

    results = sorted(results, key=lambda x: x["video"])

    dataset = [r["features"] for r in results]
    names = [r["video"] for r in results]
    fps_used = [r["fps"] for r in results]

    df_out = pd.DataFrame(dataset, columns=generate_feature_names())
    df_out.insert(0, "FPS", fps_used)
    df_out.insert(0, "Video", names)
    output_parent = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_parent, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    total_elapsed = time.time() - total_start
    print(f"\n✅ Feature dataset saved to: {output_csv}")
    print(f"🎯 All files processed. Total time: {format_seconds(total_elapsed)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the study's 88 video-level motion features from trajectory CSV files."
    )
    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Directory containing de-identified *_trajectory.csv files.",
    )
    parser.add_argument(
        "--fps-csv",
        required=True,
        help="CSV containing Filename and FPS columns.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/video_level_88_features.csv",
        help="Output feature table (default: %(default)s).",
    )
    parser.add_argument("--default-fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fps_mapping = load_fps_mapping(args.fps_csv)
    process_all_trajectory_csvs_parallel(
        folder=args.trajectory_dir,
        fps_map=fps_mapping,
        output_csv=args.output_csv,
        fps_csv_path=args.fps_csv,
        default_fps=args.default_fps,
        max_workers=args.max_workers,
    )
