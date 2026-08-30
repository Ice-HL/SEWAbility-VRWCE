import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. Runtime paths (configured from command-line arguments)
# ============================================================

INPUT_DIR = None
OUTPUT_DIR = None
ASSIGNMENT_DIR = None
PLOT_DIR = None
SUMMARY_DIR = None


K_VALUES = [2, 3, 4, 5, 6]

RANDOM_STATE = 42

SILHOUETTE_SAMPLE_SIZE = 10000

PLOT_SAMPLE_SIZE = 12000

N_INIT = 20
MAX_ITER = 500


META_COLS = [
    "Video",
    "Segment_ID",
    "Start_Frame",
    "End_Frame",
    "Start_sec",
    "End_sec",
    "FPS",
]

EXCLUDE_PATTERNS = [
    "cluster",
    "pca_",
    "tsne_",
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


def normalize_video_id(x):
    """
    1.mp4 -> 1
    1_segments.csv -> 1
    1.0 -> 1
    """
    s = str(x).strip()
    s = os.path.basename(s)
    s = s.replace("_segments.csv", "")
    s = s.replace(".mp4", "")

    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_and_merge_segments(input_dir):
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith("_segments.csv")
    ])

    if len(files) == 0:
        raise FileNotFoundError(f"No *_segments.csv files found in: {input_dir}")

    all_dfs = []

    print("=" * 90)
    print("Loading segment-level feature files")
    print("=" * 90)
    print(f"Input folder: {input_dir}")
    print(f"Found {len(files)} segment CSV files.")

    for f in files:
        path = os.path.join(input_dir, f)
        df = pd.read_csv(path)

        video_id_from_file = normalize_video_id(f)

        if "Video" not in df.columns:
            df.insert(0, "Video", video_id_from_file)
        else:
            df["Video"] = df["Video"].apply(normalize_video_id)

        if "Segment_ID" not in df.columns:
            df.insert(
                1,
                "Segment_ID",
                [f"segment_{i+1:04d}" for i in range(len(df))]
            )

        all_dfs.append(df)

        print(f"✅ {f}: {len(df)} segments, {df.shape[1]} columns")

    merged_df = pd.concat(all_dfs, ignore_index=True)

    merged_path = os.path.join(OUTPUT_DIR, "TaskA_all_segments_merged.csv")
    merged_df.to_csv(merged_path, index=False)

    print("\n✅ Merged segment dataset saved:")
    print(f"   {merged_path}")
    print(f"   Total segments: {len(merged_df)}")
    print(f"   Total columns: {merged_df.shape[1]}")

    return merged_df


def get_feature_columns(df):
    feature_cols = []

    for col in df.columns:
        if col in META_COLS:
            continue

        col_lower = col.lower()

        if any(pattern in col_lower for pattern in EXCLUDE_PATTERNS):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def prepare_feature_matrix(df, feature_cols):
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    for col in X.columns:
        if X[col].isna().all():
            X[col] = 0
        else:
            X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, scaler


def compute_silhouette_safe(X, labels):
    n = X.shape[0]

    if len(np.unique(labels)) < 2:
        return np.nan

    sample_size = min(SILHOUETTE_SAMPLE_SIZE, n)

    try:
        score = silhouette_score(
            X,
            labels,
            sample_size=sample_size,
            random_state=RANDOM_STATE,
        )
        return score
    except Exception as e:
        print(f"⚠ Silhouette calculation failed: {e}")
        return np.nan


def save_cluster_size_summary(labels_1based, setting_name, k):
    counts = pd.Series(labels_1based).value_counts().sort_index()

    cluster_size_text = ",".join([str(int(v)) for v in counts.values])

    summary_df = pd.DataFrame({
        "Cluster": counts.index.astype(int),
        "Segment_Count": counts.values.astype(int),
        "Percentage": counts.values / counts.values.sum() * 100,
    })

    path = os.path.join(
        SUMMARY_DIR,
        f"cluster_sizes_{setting_name}_k{k}.csv"
    )

    summary_df.to_csv(path, index=False)

    return cluster_size_text, path


def save_video_cluster_summary(assignment_df, setting_name, k):
    count_df = (
        assignment_df
        .groupby(["Video", "Cluster"])
        .size()
        .reset_index(name="Segment_Count")
    )

    total_df = (
        assignment_df
        .groupby("Video")
        .size()
        .reset_index(name="Video_Total_Segments")
    )

    out_df = count_df.merge(total_df, on="Video", how="left")
    out_df["Percentage"] = out_df["Segment_Count"] / out_df["Video_Total_Segments"] * 100

    path = os.path.join(
        SUMMARY_DIR,
        f"video_cluster_distribution_{setting_name}_k{k}.csv"
    )

    out_df.to_csv(path, index=False)

    return path


def save_pca_plot(X_for_plot, labels_1based, setting_name, k):
    n = X_for_plot.shape[0]

    rng = np.random.default_rng(RANDOM_STATE)

    if n > PLOT_SAMPLE_SIZE:
        sample_idx = rng.choice(n, size=PLOT_SAMPLE_SIZE, replace=False)
    else:
        sample_idx = np.arange(n)

    X_sample = X_for_plot[sample_idx]
    labels_sample = np.asarray(labels_1based)[sample_idx]

    if X_sample.shape[1] < 2:
        return None

    pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
    X_vis = pca_vis.fit_transform(X_sample)

    plt.figure(figsize=(8, 6))

    for cluster_id in sorted(np.unique(labels_sample)):
        mask = labels_sample == cluster_id
        plt.scatter(
            X_vis[mask, 0],
            X_vis[mask, 1],
            s=8,
            alpha=0.55,
            label=f"Cluster {cluster_id} (n={mask.sum()})",
        )

    plt.title(f"{setting_name}, K={k} segment clustering")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()

    plot_path = os.path.join(
        PLOT_DIR,
        f"pca2_scatter_{setting_name}_k{k}.png"
    )

    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


def run_kmeans_for_setting(
    df,
    X_cluster,
    setting_name,
    feature_space_description,
    pca_n_components=None,
    explained_variance=None,
):
    results = []

    print("\n" + "=" * 90)
    print(f"Running KMeans for setting: {setting_name}")
    print("=" * 90)
    print(f"Feature space: {feature_space_description}")
    print(f"Shape: {X_cluster.shape}")

    for k in K_VALUES:
        start_time = time.time()

        print(f"\n🔹 {setting_name} | K={k}")

        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
            max_iter=MAX_ITER,
        )

        labels_0based = kmeans.fit_predict(X_cluster)
        labels_1based = labels_0based + 1

        sil = compute_silhouette_safe(X_cluster, labels_0based)

        cluster_size_text, cluster_size_path = save_cluster_size_summary(
            labels_1based=labels_1based,
            setting_name=setting_name,
            k=k,
        )

        assignment_cols = [
            c for c in META_COLS
            if c in df.columns
        ]

        assignment_df = df[assignment_cols].copy()
        assignment_df["Cluster"] = labels_1based

        assignment_df.insert(0, "Feature_Setting", setting_name)
        assignment_df.insert(1, "K", k)

        assignment_path = os.path.join(
            ASSIGNMENT_DIR,
            f"cluster_assignment_{setting_name}_k{k}.csv"
        )

        assignment_df.to_csv(assignment_path, index=False)

        video_summary_path = save_video_cluster_summary(
            assignment_df=assignment_df,
            setting_name=setting_name,
            k=k,
        )

        plot_path = save_pca_plot(
            X_for_plot=X_cluster,
            labels_1based=labels_1based,
            setting_name=setting_name,
            k=k,
        )

        elapsed = time.time() - start_time

        print(f"   Silhouette score: {sil:.4f}")
        print(f"   Cluster sizes: {cluster_size_text}")
        print(f"   Assignment saved: {assignment_path}")
        print(f"   Time: {format_seconds(elapsed)}")

        results.append({
            "Feature_Setting": setting_name,
            "K": k,
            "Silhouette_Score": sil,
            "Cluster_Sizes": cluster_size_text,
            "N_Segments": X_cluster.shape[0],
            "N_Features_Used": X_cluster.shape[1],
            "PCA_N_Components": pca_n_components,
            "PCA_Explained_Variance": explained_variance,
            "Assignment_CSV": assignment_path,
            "Cluster_Size_CSV": cluster_size_path,
            "Video_Cluster_Distribution_CSV": video_summary_path,
            "Plot_Path": plot_path,
        })

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster segment-level 88-feature vectors in Raw88, PCA90, and "
            "PCA95 representations."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_segments.csv files from script 04.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/segment_level_clustering",
        help="Output directory (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    global INPUT_DIR, OUTPUT_DIR, ASSIGNMENT_DIR, PLOT_DIR, SUMMARY_DIR
    args = parse_args()
    INPUT_DIR = os.path.abspath(args.input_dir)
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    ASSIGNMENT_DIR = os.path.join(OUTPUT_DIR, "cluster_assignments")
    PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
    SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summaries")
    for directory in [ASSIGNMENT_DIR, PLOT_DIR, SUMMARY_DIR]:
        os.makedirs(directory, exist_ok=True)

    total_start = time.time()

    df_all = load_and_merge_segments(INPUT_DIR)

    feature_cols = get_feature_columns(df_all)

    feature_cols_path = os.path.join(OUTPUT_DIR, "feature_columns_used.txt")
    with open(feature_cols_path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(col + "\n")

    print("\n" + "=" * 90)
    print("Feature column check")
    print("=" * 90)
    print(f"Number of feature columns detected: {len(feature_cols)}")
    print(f"Feature columns saved to: {feature_cols_path}")

    if len(feature_cols) != 88:
        print("⚠ Warning: The number of detected feature columns is not 88.")
        print("Please check whether metadata columns were accidentally included or feature columns are missing.")

    X_raw, X_scaled, scaler = prepare_feature_matrix(df_all, feature_cols)

    data_summary = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_before_scaling": X_raw.mean().values,
        "Std_before_scaling": X_raw.std().values,
        "Missing_Count": X_raw.isna().sum().values,
    })

    data_summary_path = os.path.join(OUTPUT_DIR, "feature_summary_before_scaling.csv")
    data_summary.to_csv(data_summary_path, index=False)

    all_results = []

    # ========================================================
    # Raw 88
    # ========================================================
    raw_results = run_kmeans_for_setting(
        df=df_all,
        X_cluster=X_scaled,
        setting_name="Raw88",
        feature_space_description="Standardized original 88-dimensional feature space",
        pca_n_components=None,
        explained_variance=None,
    )

    all_results.extend(raw_results)

    # ========================================================
    # PCA 90
    # ========================================================
    pca90 = PCA(n_components=0.90, svd_solver="full", random_state=RANDOM_STATE)
    X_pca90 = pca90.fit_transform(X_scaled)

    pca90_info = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(pca90.n_components_)],
        "Explained_Variance_Ratio": pca90.explained_variance_ratio_,
        "Cumulative_Explained_Variance": np.cumsum(pca90.explained_variance_ratio_),
    })

    pca90_info_path = os.path.join(OUTPUT_DIR, "PCA90_explained_variance.csv")
    pca90_info.to_csv(pca90_info_path, index=False)

    pca90_results = run_kmeans_for_setting(
        df=df_all,
        X_cluster=X_pca90,
        setting_name="PCA90",
        feature_space_description=f"PCA-reduced feature space retaining 90% variance ({pca90.n_components_} PCs)",
        pca_n_components=pca90.n_components_,
        explained_variance=float(np.sum(pca90.explained_variance_ratio_)),
    )

    all_results.extend(pca90_results)

    # ========================================================
    # PCA 95
    # ========================================================
    pca95 = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE)
    X_pca95 = pca95.fit_transform(X_scaled)

    pca95_info = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(pca95.n_components_)],
        "Explained_Variance_Ratio": pca95.explained_variance_ratio_,
        "Cumulative_Explained_Variance": np.cumsum(pca95.explained_variance_ratio_),
    })

    pca95_info_path = os.path.join(OUTPUT_DIR, "PCA95_explained_variance.csv")
    pca95_info.to_csv(pca95_info_path, index=False)

    pca95_results = run_kmeans_for_setting(
        df=df_all,
        X_cluster=X_pca95,
        setting_name="PCA95",
        feature_space_description=f"PCA-reduced feature space retaining 95% variance ({pca95.n_components_} PCs)",
        pca_n_components=pca95.n_components_,
        explained_variance=float(np.sum(pca95.explained_variance_ratio_)),
    )

    all_results.extend(pca95_results)

    # ========================================================
    # ========================================================
    summary_df = pd.DataFrame(all_results)

    summary_df = summary_df.sort_values(
        by=["Feature_Setting", "Silhouette_Score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    summary_path = os.path.join(OUTPUT_DIR, "kmeans_comparison_summary_segments.csv")
    summary_df.to_csv(summary_path, index=False)

    top3_df = (
        summary_df
        .sort_values(["Feature_Setting", "Silhouette_Score"], ascending=[True, False])
        .groupby("Feature_Setting")
        .head(3)
        .reset_index(drop=True)
    )

    top3_path = os.path.join(OUTPUT_DIR, "kmeans_top3_by_setting_segments.csv")
    top3_df.to_csv(top3_path, index=False)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 90)
    print("🎯 Segment-level clustering completed.")
    print("=" * 90)
    print(f"Total segments: {len(df_all)}")
    print(f"Feature columns used: {len(feature_cols)}")
    print(f"Summary saved: {summary_path}")
    print(f"Top 3 summary saved: {top3_path}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Total time: {format_seconds(total_elapsed)}")

    print("\nTop 3 clustering solutions by feature setting:")
    print(top3_df[[
        "Feature_Setting",
        "K",
        "Silhouette_Score",
        "Cluster_Sizes",
        "N_Features_Used",
        "PCA_N_Components",
        "PCA_Explained_Variance",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
