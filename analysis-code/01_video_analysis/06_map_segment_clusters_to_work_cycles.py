import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Runtime paths (configured from command-line arguments)
# ============================================================

SEGMENT_CLUSTER_CSV = None
WORK_CYCLE_CSV = None
OUTPUT_DIR = None
PLOT_DIR = None
VIDEO_PLOT_DIR = None


# CLUSTER_NAMES = {
#     1: "Fabric propulsion during sewing",
#     2: "Fabric pick-up from the ground",
#     3: "Fabric placement or return",
# }

CLUSTER_NAMES = {
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
}

BIN_SIZE_PERCENT = 5


def normalize_video_id(x):
    """
    1.mp4 -> 1
    1_segments.csv -> 1
    1_trajectory.csv -> 1
    1.0 -> 1
    """
    s = str(x).strip()
    s = os.path.basename(s)

    s = s.replace("_segments.csv", "")
    s = s.replace("_trajectory.csv", "")
    s = s.replace(".mp4", "")

    try:
        return str(int(float(s)))
    except Exception:
        return s


def video_order_key(x):
    s = str(x).strip()

    try:
        return int(float(s))
    except Exception:
        parts = re.split(r"(\d+)", s)
        return parts


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(f"Cannot find required column from candidates: {candidates}")

    return None


def format_time(seconds):
    seconds = float(seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


def load_data():
    if not os.path.exists(SEGMENT_CLUSTER_CSV):
        raise FileNotFoundError(f"Segment cluster CSV not found: {SEGMENT_CLUSTER_CSV}")

    if not os.path.exists(WORK_CYCLE_CSV):
        raise FileNotFoundError(f"Work cycle CSV not found: {WORK_CYCLE_CSV}")

    seg_df = pd.read_csv(SEGMENT_CLUSTER_CSV)
    cycle_df = pd.read_csv(WORK_CYCLE_CSV)

    # ---------- segment columns ----------
    seg_video_col = find_col(seg_df, ["Video"])
    seg_start_col = find_col(seg_df, ["Start_Frame"])
    seg_end_col = find_col(seg_df, ["End_Frame"])
    seg_cluster_col = find_col(seg_df, ["Cluster", "cluster"])

    seg_df = seg_df.copy()
    seg_df["Video_ID"] = seg_df[seg_video_col].apply(normalize_video_id)
    seg_df["Segment_Start_Frame"] = seg_df[seg_start_col].astype(float)
    seg_df["Segment_End_Frame"] = seg_df[seg_end_col].astype(float)
    seg_df["Segment_Center_Frame"] = (
        seg_df["Segment_Start_Frame"] + seg_df["Segment_End_Frame"]
    ) / 2.0
    seg_df["Cluster"] = seg_df[seg_cluster_col].astype(int)

    # ---------- work cycle columns ----------
    cycle_video_col = find_col(cycle_df, ["Video"])
    cycle_start_col = find_col(cycle_df, ["Start_Frame"])
    cycle_end_col = find_col(cycle_df, ["End_Frame"])

    cycle_id_col = find_col(
        cycle_df,
        ["Work_Cycle_ID", "Work Cycle", "Work_Cycle", "Cycle_ID", "Cycle"],
        required=False,
    )

    fps_col = find_col(cycle_df, ["FPS"], required=False)

    cycle_df = cycle_df.copy()
    cycle_df["Video_ID"] = cycle_df[cycle_video_col].apply(normalize_video_id)
    cycle_df["Cycle_Start_Frame"] = cycle_df[cycle_start_col].astype(float)
    cycle_df["Cycle_End_Frame"] = cycle_df[cycle_end_col].astype(float)

    if cycle_id_col is not None:
        cycle_df["Cycle_ID"] = cycle_df[cycle_id_col].astype(str)
    else:
        cycle_df["Cycle_ID"] = (
            cycle_df.groupby("Video_ID").cumcount() + 1
        ).apply(lambda x: f"Cycle {x}")

    if fps_col is not None:
        cycle_df["FPS"] = cycle_df[fps_col].astype(float)
    else:
        if "FPS" in seg_df.columns:
            fps_map = seg_df.groupby("Video_ID")["FPS"].first().to_dict()
            cycle_df["FPS"] = cycle_df["Video_ID"].map(fps_map).astype(float)
        else:
            raise ValueError("Cannot find FPS in either work cycle CSV or segment CSV.")

    cycle_df["Cycle_Duration_Frames"] = (
        cycle_df["Cycle_End_Frame"] - cycle_df["Cycle_Start_Frame"]
    )
    cycle_df["Cycle_Duration_sec"] = (
        cycle_df["Cycle_Duration_Frames"] / cycle_df["FPS"]
    )

    print("=" * 90)
    print("Data loaded")
    print("=" * 90)
    print(f"Segments: {len(seg_df)}")
    print(f"Work cycles: {len(cycle_df)}")
    print(f"Videos in segment file: {seg_df['Video_ID'].nunique()}")
    print(f"Videos in work cycle file: {cycle_df['Video_ID'].nunique()}")

    return seg_df, cycle_df


def map_segments_to_cycles(seg_df, cycle_df):
    mapped_rows = []

    video_ids = sorted(cycle_df["Video_ID"].unique(), key=video_order_key)

    cycle_global_index = 0

    for video_id in video_ids:
        seg_v = seg_df[seg_df["Video_ID"] == video_id].copy()
        cyc_v = cycle_df[cycle_df["Video_ID"] == video_id].copy()

        if seg_v.empty:
            print(f"⚠ No segments found for video {video_id}")
            continue

        if cyc_v.empty:
            print(f"⚠ No cycles found for video {video_id}")
            continue

        cyc_v = cyc_v.sort_values("Cycle_Start_Frame").reset_index(drop=True)

        for cycle_index, cycle in cyc_v.iterrows():
            c_start = float(cycle["Cycle_Start_Frame"])
            c_end = float(cycle["Cycle_End_Frame"])
            fps = float(cycle["FPS"])

            mask = (
                (seg_v["Segment_Center_Frame"] >= c_start) &
                (seg_v["Segment_Center_Frame"] <= c_end)
            )

            seg_c = seg_v[mask].copy()

            if seg_c.empty:
                continue

            cycle_global_index += 1

            seg_c["Cycle_Index_Global"] = cycle_global_index
            seg_c["Cycle_Number_In_Video"] = cycle_index + 1
            seg_c["Cycle_ID"] = cycle["Cycle_ID"]

            seg_c["Cycle_Start_Frame"] = c_start
            seg_c["Cycle_End_Frame"] = c_end
            seg_c["Cycle_Duration_Frames"] = c_end - c_start
            seg_c["Cycle_Duration_sec"] = cycle["Cycle_Duration_sec"]

            seg_c["Relative_Start_Frame"] = seg_c["Segment_Start_Frame"] - c_start
            seg_c["Relative_End_Frame"] = seg_c["Segment_End_Frame"] - c_start
            seg_c["Relative_Center_Frame"] = seg_c["Segment_Center_Frame"] - c_start

            seg_c["Relative_Start_sec"] = seg_c["Relative_Start_Frame"] / fps
            seg_c["Relative_End_sec"] = seg_c["Relative_End_Frame"] / fps
            seg_c["Relative_Center_sec"] = seg_c["Relative_Center_Frame"] / fps

            denom = c_end - c_start
            if denom <= 0:
                continue

            seg_c["Relative_Start_Percent"] = (
                seg_c["Relative_Start_Frame"] / denom * 100
            )
            seg_c["Relative_End_Percent"] = (
                seg_c["Relative_End_Frame"] / denom * 100
            )
            seg_c["Relative_Center_Percent"] = (
                seg_c["Relative_Center_Frame"] / denom * 100
            )

            seg_c["Relative_Start_Percent_Clipped"] = (
                seg_c["Relative_Start_Percent"].clip(0, 100)
            )
            seg_c["Relative_End_Percent_Clipped"] = (
                seg_c["Relative_End_Percent"].clip(0, 100)
            )
            seg_c["Relative_Center_Percent_Clipped"] = (
                seg_c["Relative_Center_Percent"].clip(0, 100)
            )

            mapped_rows.append(seg_c)

    if len(mapped_rows) == 0:
        raise RuntimeError("No segments were mapped to work cycles.")

    mapped_df = pd.concat(mapped_rows, ignore_index=True)

    out_path = os.path.join(OUTPUT_DIR, "segment_to_workcycle_mapping.csv")
    mapped_df.to_csv(out_path, index=False)

    print("\n" + "=" * 90)
    print("Segment-to-cycle mapping completed")
    print("=" * 90)
    print(f"Mapped segments: {len(mapped_df)}")
    print(f"Mapping CSV saved: {out_path}")

    return mapped_df


def summarize_overall_cluster_distribution(mapped_df):
    summary = (
        mapped_df["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    summary.columns = ["Cluster", "Segment_Count"]
    summary["Percentage"] = (
        summary["Segment_Count"] / summary["Segment_Count"].sum() * 100
    )
    summary["Cluster_Name"] = summary["Cluster"].map(CLUSTER_NAMES)

    out_path = os.path.join(
        OUTPUT_DIR,
        "overall_cluster_distribution_within_workcycles.csv"
    )
    summary.to_csv(out_path, index=False)

    print("\nOverall cluster distribution within mapped work cycles:")
    print(summary.to_string(index=False))
    print(f"Saved: {out_path}")

    return summary


def summarize_cycle_cluster_distribution(mapped_df):
    group_cols = [
        "Video_ID",
        "Cycle_Number_In_Video",
        "Cycle_ID",
        "Cycle_Start_Frame",
        "Cycle_End_Frame",
        "Cycle_Duration_sec",
        "Cluster",
    ]

    count_df = (
        mapped_df
        .groupby(group_cols)
        .size()
        .reset_index(name="Segment_Count")
    )

    total_df = (
        mapped_df
        .groupby([
            "Video_ID",
            "Cycle_Number_In_Video",
            "Cycle_ID",
            "Cycle_Start_Frame",
            "Cycle_End_Frame",
            "Cycle_Duration_sec",
        ])
        .size()
        .reset_index(name="Cycle_Total_Segments")
    )

    summary_df = count_df.merge(
        total_df,
        on=[
            "Video_ID",
            "Cycle_Number_In_Video",
            "Cycle_ID",
            "Cycle_Start_Frame",
            "Cycle_End_Frame",
            "Cycle_Duration_sec",
        ],
        how="left",
    )

    summary_df["Percentage"] = (
        summary_df["Segment_Count"] /
        summary_df["Cycle_Total_Segments"] * 100
    )
    summary_df["Cluster_Name"] = summary_df["Cluster"].map(CLUSTER_NAMES)

    out_path = os.path.join(OUTPUT_DIR, "cycle_cluster_distribution.csv")
    summary_df.to_csv(out_path, index=False)

    print(f"Cycle cluster distribution saved: {out_path}")

    # wide version
    wide_df = summary_df.pivot_table(
        index=[
            "Video_ID",
            "Cycle_Number_In_Video",
            "Cycle_ID",
            "Cycle_Start_Frame",
            "Cycle_End_Frame",
            "Cycle_Duration_sec",
        ],
        columns="Cluster",
        values="Percentage",
        fill_value=0,
    ).reset_index()

    new_cols = []
    for c in wide_df.columns:
        if isinstance(c, int):
            new_cols.append(f"Cluster_{c}_Percent")
        else:
            new_cols.append(c)

    wide_df.columns = new_cols

    wide_path = os.path.join(OUTPUT_DIR, "cycle_cluster_distribution_wide.csv")
    wide_df.to_csv(wide_path, index=False)

    print(f"Wide cycle cluster distribution saved: {wide_path}")

    return summary_df, wide_df


def summarize_normalized_timeline(mapped_df):
    bin_edges = np.arange(0, 100 + BIN_SIZE_PERCENT, BIN_SIZE_PERCENT)

    df = mapped_df.copy()

    df["Timeline_Bin"] = pd.cut(
        df["Relative_Center_Percent_Clipped"],
        bins=bin_edges,
        include_lowest=True,
        right=False,
    )

    bin_cluster = (
        df
        .groupby(["Timeline_Bin", "Cluster"], observed=False)
        .size()
        .reset_index(name="Segment_Count")
    )

    bin_total = (
        df
        .groupby("Timeline_Bin", observed=False)
        .size()
        .reset_index(name="Bin_Total")
    )

    bin_summary = bin_cluster.merge(bin_total, on="Timeline_Bin", how="left")

    bin_summary = bin_summary[bin_summary["Bin_Total"] > 0].copy()

    bin_summary["Percentage"] = (
        bin_summary["Segment_Count"] / bin_summary["Bin_Total"] * 100
    )
    bin_summary["Cluster_Name"] = bin_summary["Cluster"].map(CLUSTER_NAMES)
    bin_summary["Bin_Label"] = bin_summary["Timeline_Bin"].astype(str)

    def get_bin_center(interval):
        return (interval.left + interval.right) / 2

    bin_summary["Bin_Center_Percent"] = (
        bin_summary["Timeline_Bin"].apply(get_bin_center)
    )

    out_path = os.path.join(
        OUTPUT_DIR,
        "normalized_timeline_cluster_distribution.csv"
    )
    bin_summary.to_csv(out_path, index=False)

    print(f"Normalized timeline distribution saved: {out_path}")

    return bin_summary


def plot_all_cycles_raster(mapped_df):
    df = mapped_df.copy()

    df["Video_Order"] = df["Video_ID"].apply(video_order_key)
    df["Cycle_Order"] = df["Cycle_Number_In_Video"].astype(int)

    cycle_table = (
        df[["Video_ID", "Video_Order", "Cycle_Number_In_Video", "Cycle_Order"]]
        .drop_duplicates()
        .sort_values(["Video_Order", "Cycle_Order"])
        .reset_index(drop=True)
    )

    cycle_table["Y"] = np.arange(len(cycle_table))

    df = df.merge(
        cycle_table[["Video_ID", "Cycle_Number_In_Video", "Y"]],
        on=["Video_ID", "Cycle_Number_In_Video"],
        how="left",
    )

    clusters = sorted(df["Cluster"].unique())
    cmap = plt.get_cmap("tab10")
    cluster_color = {c: cmap(i % 10) for i, c in enumerate(clusters)}

    plt.figure(figsize=(14, max(8, len(cycle_table) * 0.045)))
    ax = plt.gca()

    for cluster in clusters:
        sub = df[df["Cluster"] == cluster]

        ax.hlines(
            y=sub["Y"],
            xmin=sub["Relative_Start_Percent_Clipped"],
            xmax=sub["Relative_End_Percent_Clipped"],
            colors=[cluster_color[cluster]],
            linewidth=1.2,
            alpha=0.85,
            label=f"{CLUSTER_NAMES.get(cluster, f'Cluster {cluster}')} (Cluster {cluster})",
        )

    ax.set_xlabel("Normalized work-cycle timeline (%)")
    ax.set_ylabel("Work cycles")
    ax.set_title("Distribution of segment clusters across normalized work-cycle timelines")
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, len(cycle_table) + 1)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "all_workcycles_segment_cluster_raster.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"All-cycle raster plot saved: {out_path}")


def plot_per_video_raster(mapped_df):
    clusters = sorted(mapped_df["Cluster"].unique())
    cmap = plt.get_cmap("tab10")
    cluster_color = {c: cmap(i % 10) for i, c in enumerate(clusters)}

    video_ids = sorted(mapped_df["Video_ID"].unique(), key=video_order_key)

    for video_id in video_ids:
        dfv = mapped_df[mapped_df["Video_ID"] == video_id].copy()

        if dfv.empty:
            continue

        cycles = (
            dfv[["Cycle_Number_In_Video"]]
            .drop_duplicates()
            .sort_values("Cycle_Number_In_Video")
            .reset_index(drop=True)
        )
        cycles["Y"] = np.arange(len(cycles))

        dfv = dfv.merge(cycles, on="Cycle_Number_In_Video", how="left")

        height = max(4, min(12, len(cycles) * 0.55))

        plt.figure(figsize=(12, height))
        ax = plt.gca()

        for cluster in clusters:
            sub = dfv[dfv["Cluster"] == cluster]

            if sub.empty:
                continue

            ax.hlines(
                y=sub["Y"],
                xmin=sub["Relative_Start_Percent_Clipped"],
                xmax=sub["Relative_End_Percent_Clipped"],
                colors=[cluster_color[cluster]],
                linewidth=4,
                alpha=0.85,
                label=f"{CLUSTER_NAMES.get(cluster, f'Cluster {cluster}')} (Cluster {cluster})",
            )

        ax.set_xlabel("Normalized work-cycle timeline (%)")
        ax.set_ylabel("Cycle number")
        ax.set_title(f"Segment cluster distribution across work cycles: Video {video_id}")

        ax.set_yticks(cycles["Y"].to_numpy())
        ax.set_yticklabels(cycles["Cycle_Number_In_Video"].astype(str).to_numpy())

        ax.set_xlim(0, 100)
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()

        out_path = os.path.join(
            VIDEO_PLOT_DIR,
            f"video_{video_id}_workcycle_cluster_timeline.png"
        )

        plt.savefig(out_path, dpi=300)
        plt.close()

    print(f"Per-video raster plots saved in: {VIDEO_PLOT_DIR}")


def plot_normalized_timeline_distribution(bin_summary):
    clusters = sorted(bin_summary["Cluster"].unique())

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    for cluster in clusters:
        sub = (
            bin_summary[bin_summary["Cluster"] == cluster]
            .sort_values("Bin_Center_Percent")
        )

        ax.plot(
            sub["Bin_Center_Percent"],
            sub["Percentage"],
            marker="o",
            linewidth=2,
            label=f"{CLUSTER_NAMES.get(cluster, f'Cluster {cluster}')} (Cluster {cluster})",
        )

    ax.set_xlabel("Normalized work-cycle timeline (%)")
    ax.set_ylabel("Segment proportion within timeline bin (%)")
    ax.set_title("Cluster distribution across normalized work-cycle timeline")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "normalized_timeline_cluster_proportion.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Normalized timeline proportion plot saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Map segment-cluster assignments to candidate work cycles and "
            "summarize their normalized temporal distribution."
        )
    )
    parser.add_argument(
        "--segment-clusters-csv",
        required=True,
        help="Cluster-assignment CSV produced by script 05.",
    )
    parser.add_argument(
        "--work-cycles-csv",
        required=True,
        help="Candidate work-cycle table produced by script 03.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/work_cycle_cluster_mapping",
        help="Output directory (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    global SEGMENT_CLUSTER_CSV, WORK_CYCLE_CSV, OUTPUT_DIR, PLOT_DIR, VIDEO_PLOT_DIR
    args = parse_args()
    SEGMENT_CLUSTER_CSV = os.path.abspath(args.segment_clusters_csv)
    WORK_CYCLE_CSV = os.path.abspath(args.work_cycles_csv)
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
    VIDEO_PLOT_DIR = os.path.join(PLOT_DIR, "per_video")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(VIDEO_PLOT_DIR, exist_ok=True)

    seg_df, cycle_df = load_data()

    mapped_df = map_segments_to_cycles(seg_df, cycle_df)

    summarize_overall_cluster_distribution(mapped_df)

    summarize_cycle_cluster_distribution(mapped_df)

    bin_summary = summarize_normalized_timeline(mapped_df)

    plot_all_cycles_raster(mapped_df)

    plot_per_video_raster(mapped_df)

    plot_normalized_timeline_distribution(bin_summary)

    print("\n" + "=" * 90)
    print("Done.")
    print("=" * 90)
    print(f"Output folder: {OUTPUT_DIR}")
    print("\nKey output files:")
    print("1. segment_to_workcycle_mapping.csv")
    print("2. overall_cluster_distribution_within_workcycles.csv")
    print("3. cycle_cluster_distribution.csv")
    print("4. cycle_cluster_distribution_wide.csv")
    print("5. normalized_timeline_cluster_distribution.csv")
    print("6. plots/all_workcycles_segment_cluster_raster.png")
    print("7. plots/normalized_timeline_cluster_proportion.png")
    print("8. plots/per_video/*.png")


if __name__ == "__main__":
    main()
