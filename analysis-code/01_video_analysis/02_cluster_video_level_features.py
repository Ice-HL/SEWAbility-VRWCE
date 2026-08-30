import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse

warnings.filterwarnings("ignore")

# =========================================================
# Configuration
# =========================================================
parser = argparse.ArgumentParser(
    description="Cluster standardized video-level 88-feature vectors and run PCA sensitivity analyses."
)
parser.add_argument(
    "--input-csv",
    required=True,
    help="Video-level feature CSV produced by 01_extract_video_level_88_features.py.",
)
parser.add_argument(
    "--output-dir",
    default="results/video_level_clustering",
    help="Output directory (default: %(default)s).",
)
args = parser.parse_args()

CSV_PATH = args.input_csv

# Try K from 2 to 13
K_RANGE = range(2, 14)

# KMeans settings
RANDOM_STATE = 42
N_INIT = 50
INIT_METHOD = "k-means++"

# Output directory
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Helper functions
# =========================================================
def draw_confidence_ellipse(points, ax, n_std=2.0, facecolor='none', **kwargs):
    if points.shape[0] < 2:
        return
    cov = np.cov(points, rowvar=False)
    if cov.ndim < 2:
        return
    mean = np.mean(points, axis=0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 1e-12))
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        **kwargs
    )
    ax.add_patch(ellipse)

def evaluate_setting(setting_name, X_for_clustering, X_2d_vis, df_original, video_names):
    setting_dir = os.path.join(OUTPUT_DIR, setting_name)
    os.makedirs(setting_dir, exist_ok=True)

    summary_rows = []
    kmeans_results = {}

    for k in K_RANGE:
        kmeans = KMeans(
            n_clusters=k,
            init=INIT_METHOD,
            n_init=N_INIT,
            random_state=RANDOM_STATE
        )
        labels = kmeans.fit_predict(X_for_clustering)
        sil = silhouette_score(X_for_clustering, labels)

        cluster_sizes = np.bincount(labels)
        kmeans_results[k] = (labels, sil)

        # Save cluster assignment table
        cluster_df = df_original.copy()
        cluster_df["Cluster"] = labels + 1   # 1-based indexing
        cluster_df["PC1"] = X_2d_vis[:, 0]
        cluster_df["PC2"] = X_2d_vis[:, 1]

        out_csv = os.path.join(setting_dir, f"cluster_assignment_k{k}.csv")
        cluster_df.to_csv(out_csv, index=False)

        summary_rows.append({
            "Setting": setting_name,
            "K": k,
            "Silhouette_Score": sil,
            "Cluster_Sizes": ",".join(map(str, cluster_sizes.tolist())),
            "Assignment_File": out_csv
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="Silhouette_Score", ascending=False).reset_index(drop=True)

    print(f"\n===== {setting_name} =====")
    print(summary_df[["K", "Silhouette_Score", "Cluster_Sizes"]].head(3))

    # Plot top 3 results
    cluster_colors = ['red', 'gold', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan']
    top_3 = summary_df.head(3)

    for rank, row in enumerate(top_3.itertuples(index=False), start=1):
        k = row.K
        sil = row.Silhouette_Score
        labels, _ = kmeans_results[k]

        fig, ax = plt.subplots(figsize=(9, 7))

        for cluster_id in np.unique(labels):
            points = X_2d_vis[labels == cluster_id]
            color = cluster_colors[cluster_id % len(cluster_colors)]

            ax.scatter(
                points[:, 0],
                points[:, 1],
                label=f"Cluster {cluster_id + 1}",
                color=color,
                edgecolor="k",
                s=60
            )
            draw_confidence_ellipse(points, ax, n_std=2.0, edgecolor=color, linestyle="--", linewidth=1.5)

        for i, name in enumerate(video_names):
            ax.text(X_2d_vis[i, 0], X_2d_vis[i, 1], str(name), fontsize=7)

        ax.set_title(f"{setting_name} | PCA 2D visualization | Top {rank} | K={k} | Silhouette={sil:.3f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(setting_dir, f"pca_plot_top{rank}_k{k}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    summary_path = os.path.join(setting_dir, f"{setting_name}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    return summary_df

# =========================================================
# Load data
# =========================================================
df = pd.read_csv(CSV_PATH)

print("All columns:")
for i, c in enumerate(df.columns):
    print(i, c)

# =========================================================
# Keep only the true 88 features
# =========================================================
meta_cols = ["Video", "FPS"]
feature_cols = [c for c in df.columns if c not in meta_cols]

print(f"\nMetadata columns: {meta_cols}")
print(f"Feature columns detected: {len(feature_cols)}")

if len(feature_cols) != 88:
    raise ValueError(f"Expected 88 feature columns, but got {len(feature_cols)}. Please check the file.")

df_original = df.copy()
video_names = df["Video"].tolist()

X = df[feature_cols].copy()

# Fill missing values if needed
if X.isna().sum().sum() > 0:
    print("⚠ Missing values detected. Filling with column means.")
    X = X.fillna(X.mean(numeric_only=True))

# =========================================================
# Standardization
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n✅ Original feature matrix shape: {X_scaled.shape}")

# =========================================================
# PCA cumulative explained variance
# =========================================================
pca_full = PCA(random_state=RANDOM_STATE)
X_pca_full = pca_full.fit_transform(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

n90 = np.argmax(cum_var >= 0.90) + 1
n95 = np.argmax(cum_var >= 0.95) + 1

print(f"✅ Components needed for 90% variance: {n90}")
print(f"✅ Components needed for 95% variance: {n95}")

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
plt.axhline(0.90, linestyle='--', label='90% variance')
plt.axhline(0.95, linestyle='--', label='95% variance')
plt.axvline(n90, linestyle=':', color='gray')
plt.axvline(n95, linestyle=':', color='gray')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_cumulative_explained_variance.png"), dpi=300)
plt.close()

# =========================================================
# PCA 2D for visualization only
# =========================================================
pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d_vis = pca_vis.fit_transform(X_scaled)

# =========================================================
# Setting 1: Raw 88 features
# =========================================================
summary_raw = evaluate_setting(
    setting_name="Raw_88",
    X_for_clustering=X_scaled,
    X_2d_vis=X_2d_vis,
    df_original=df_original,
    video_names=video_names
)

# =========================================================
# Setting 2: PCA 90%
# =========================================================
pca_90 = PCA(n_components=0.90, random_state=RANDOM_STATE)
X_pca_90 = pca_90.fit_transform(X_scaled)
print(f"\n✅ PCA_90 reduced dimension: {X_pca_90.shape[1]} | retained variance: {pca_90.explained_variance_ratio_.sum():.4f}")

summary_pca90 = evaluate_setting(
    setting_name=f"PCA_90pct_{X_pca_90.shape[1]}PCs",
    X_for_clustering=X_pca_90,
    X_2d_vis=X_2d_vis,
    df_original=df_original,
    video_names=video_names
)

# =========================================================
# Setting 3: PCA 95%
# =========================================================
pca_95 = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_pca_95 = pca_95.fit_transform(X_scaled)
print(f"\n✅ PCA_95 reduced dimension: {X_pca_95.shape[1]} | retained variance: {pca_95.explained_variance_ratio_.sum():.4f}")

summary_pca95 = evaluate_setting(
    setting_name=f"PCA_95pct_{X_pca_95.shape[1]}PCs",
    X_for_clustering=X_pca_95,
    X_2d_vis=X_2d_vis,
    df_original=df_original,
    video_names=video_names
)

# =========================================================
# Save combined comparison
# =========================================================
comparison_df = pd.concat([summary_raw, summary_pca90, summary_pca95], ignore_index=True)
comparison_df = comparison_df.sort_values(by=["Setting", "Silhouette_Score"], ascending=[True, False])

comparison_csv = os.path.join(OUTPUT_DIR, "kmeans_comparison_summary.csv")
comparison_xlsx = os.path.join(OUTPUT_DIR, "kmeans_comparison_summary.xlsx")

comparison_df.to_csv(comparison_csv, index=False)
comparison_df.to_excel(comparison_xlsx, index=False)

print(f"\n✅ Saved overall comparison table: {comparison_csv}")
print(f"✅ Saved overall comparison Excel: {comparison_xlsx}")

print("\n🎉 Done.")
print(f"All results saved in: {OUTPUT_DIR}")
