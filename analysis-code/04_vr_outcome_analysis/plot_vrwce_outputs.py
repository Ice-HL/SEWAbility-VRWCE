#!/usr/bin/env python3
"""Optional basic descriptive plots for the VR-WCE numerical analysis.

This module is imported only when figures are requested. It preserves the
original descriptive plotting implementation, not the final submission's
separately refined layout. Scientific numerical helpers are supplied by the
analysis entry point rather than duplicated here.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = Path("results")
FIGURE_STEM = "Fig5_VRWCE_technical_sessions"
SUPPLEMENTARY_STEM = "VRWCE_higher_order_stability"
SESSION_SPECS: list[tuple[str, Path, str]] = []

COLORS = {
    "S1": "#4C78A8",
    "S2": "#B279A2",
    "U1": "#59A14F",
    "U2": "#F28E2B",
}

OBSERVATION_SIZE = 7
OBSERVATION_ALPHA = 0.46
IQR_COLOR = "#596272"
IQR_LINE_WIDTH = 0.9
IQR_CAP_HALF_WIDTH = 0.035
MEDIAN_SIZE = 20
MEDIAN_EDGE_WIDTH = 0.55

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)


def jitter(count: int, width: float = 0.13) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    return np.linspace(-width, width, count)


def add_distribution(
    ax: plt.Axes,
    values_by_session: dict[str, list[float]],
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    reference: float | None = None,
    log_scale: bool = False,
) -> None:
    for x, session in enumerate([item[0] for item in SESSION_SPECS]):
        values = [float(v) for v in values_by_session.get(session, []) if v is not None and np.isfinite(v)]
        ax.scatter(
            x + jitter(len(values)),
            values,
            s=OBSERVATION_SIZE,
            color=COLORS[session],
            alpha=OBSERVATION_ALPHA,
            linewidth=0,
            rasterized=False,
            zorder=2,
        )
        if values:
            lo, med, hi = np.quantile(values, [0.25, 0.5, 0.75])
            ax.vlines(x, lo, hi, color=IQR_COLOR, lw=IQR_LINE_WIDTH, zorder=4)
            ax.hlines(
                [lo, hi],
                x - IQR_CAP_HALF_WIDTH,
                x + IQR_CAP_HALF_WIDTH,
                color=IQR_COLOR,
                lw=IQR_LINE_WIDTH,
                zorder=4,
            )
            ax.scatter(
                x,
                med,
                marker="D",
                s=MEDIAN_SIZE,
                color=COLORS[session],
                edgecolor="white",
                lw=MEDIAN_EDGE_WIDTH,
                zorder=5,
            )
    ax.set_xticks(range(4), [item[0] for item in SESSION_SPECS])
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.48, 3.48)
    if reference is not None:
        ax.axhline(reference, color="#777777", lw=0.7, ls="--", zorder=0)
    if ylim:
        ax.set_ylim(*ylim)
    if log_scale:
        ax.set_yscale("log")
    ax.tick_params(length=2.5, width=0.6)


def add_hand_distribution(
    ax: plt.Axes,
    values_by_session_and_hand: dict[str, dict[str, list[float]]],
    ylabel: str,
) -> None:
    """Plot hand-specific observations without collapsing left and right hands."""
    hand_styles = {
        "Left": (-0.090, "o"),
        "Right": (0.090, "^"),
    }
    sessions = [item[0] for item in SESSION_SPECS]
    for x, session in enumerate(sessions):
        for hand, (offset, marker) in hand_styles.items():
            values = [
                float(value)
                for value in values_by_session_and_hand.get(session, {}).get(hand, [])
                if value is not None and np.isfinite(value)
            ]
            center = x + offset
            ax.scatter(
                center + jitter(len(values), width=0.040),
                values,
                marker=marker,
                s=5,
                color=COLORS[session],
                alpha=OBSERVATION_ALPHA,
                linewidth=0,
                zorder=2,
            )
            if values:
                lo, med, hi = np.quantile(values, [0.25, 0.5, 0.75])
                ax.vlines(
                    center, lo, hi, color=IQR_COLOR, lw=IQR_LINE_WIDTH, zorder=4
                )
                ax.hlines(
                    [lo, hi],
                    center - 0.025,
                    center + 0.025,
                    color=IQR_COLOR,
                    lw=IQR_LINE_WIDTH,
                    zorder=4,
                )
                ax.scatter(
                    center,
                    med,
                    marker=marker,
                    s=13,
                    color=COLORS[session],
                    edgecolor="white",
                    lw=MEDIAN_EDGE_WIDTH,
                    zorder=5,
                )
    ax.set_xticks(range(4), sessions)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.48, 3.48)
    ax.tick_params(length=2.5, width=0.6)


def add_geometry_reference(ax: plt.Axes, value_mm: float, symbol: str) -> None:
    """Mark implemented cue geometry without implying a performance threshold."""
    ax.axhline(
        value_mm,
        color="#737B86",
        lw=0.8,
        ls=(0, (3, 2)),
        zorder=0,
    )
    ax.text(
        0.98,
        value_mm,
        f"geometry reference\n${symbol}$ = {value_mm:.0f} mm",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=5.1,
        color="#5C6470",
        linespacing=1.05,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5),
        zorder=6,
    )


def panel_label(
    ax: plt.Axes,
    letter: str,
    title: str,
    subtitle: str | None = None,
    heading_y: float = 1.15,
    subtitle_y: float = 1.055,
    gap_pt: float = 6.0,
) -> None:
    # Use a fixed point gap and a shared baseline so the letter-title spacing
    # is independent of panel width and letter glyph dimensions.
    ax.annotate(
        letter,
        xy=(0.0, heading_y),
        xycoords="axes fraction",
        xytext=(-gap_pt, 0.0),
        textcoords="offset points",
        ha="right",
        va="baseline",
        fontsize=9,
        fontweight="bold",
    )
    ax.annotate(
        title,
        xy=(0.0, heading_y),
        xycoords="axes fraction",
        xytext=(0.0, 0.0),
        textcoords="offset points",
        ha="left",
        va="baseline",
        fontsize=8.2,
        fontweight="bold",
    )
    if subtitle:
        ax.text(0.0, subtitle_y, subtitle, transform=ax.transAxes, fontsize=6.1, color="#5C6470", va="top")


def plot_main_figure(
    summaries: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    hands: list[dict[str, Any]],
    lateral_profiles: dict[str, dict[str, Any]],
) -> None:
    summary_by_session = {row["session"]: row for row in summaries}
    actions_by_session = {
        session: [row for row in actions if row["session"] == session]
        for session, _, _ in SESSION_SPECS
    }

    fig = plt.figure(figsize=(7.2, 10.05), facecolor="white")
    outer = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.88, 1.45, 1.05, 0.78],
        width_ratios=[0.80, 1.40],
        hspace=0.54,
        wspace=0.25,
        left=0.13,
        right=0.985,
        top=0.935,
        bottom=0.060,
    )

    # a. Quantity and bilateral participation.
    ax_a = fig.add_subplot(outer[0, 0])
    sessions = [item[0] for item in SESSION_SPECS]
    y = np.arange(4)
    bilateral = [summary_by_session[s]["p2_bilateral_classified_count"] for s in sessions]
    single = [summary_by_session[s]["p2_single_hand_classified_count"] for s in sessions]
    colors = [COLORS[s] for s in sessions]
    bar_height = 0.57
    bar_edge_width = 0.80
    ax_a.barh(
        y,
        bilateral,
        color=colors,
        height=bar_height,
        edgecolor=colors,
        linewidth=bar_edge_width,
        zorder=2,
    )
    ax_a.barh(
        y,
        single,
        left=bilateral,
        facecolor="white",
        edgecolor=colors,
        hatch="////",
        linewidth=bar_edge_width,
        height=bar_height,
        zorder=3,
    )
    for idx, session in enumerate(sessions):
        total = summary_by_session[session]["p2_recorded_action_count"]
        rate = 100.0 * summary_by_session[session]["p2_bilateral_participation_rate"]
        ax_a.text(total + 0.35, idx, f"{total}  ({rate:.0f}%)", va="center", fontsize=6.4)
    ax_a.axvline(20, color="#6B7280", ls="--", lw=0.8)
    ax_a.axvline(27, color="#6B7280", ls=":", lw=0.8)
    ax_a.set_yticks(y, sessions)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 31)
    ax_a.set_xlabel("Runtime-recorded P2 actions (count)")
    ax_a.set_xticks([0, 10, 20, 27])
    panel_label(
        ax_a,
        "a",
        "Work quantity and bilateral participation",
        "All runtime-recorded P2 actions",
    )

    # b. Work speed: bilateral-classified durations and non-bridged gaps.
    bgrid = outer[0, 1].subgridspec(1, 3, wspace=0.47)
    ax_b0 = fig.add_subplot(bgrid[0, 0])
    ax_b1 = fig.add_subplot(bgrid[0, 1])
    ax_b2 = fig.add_subplot(bgrid[0, 2])
    duration_values: dict[str, list[float]] = {}
    gap_values: dict[str, list[float]] = {}
    for session in sessions:
        all_rows = sorted(actions_by_session[session], key=lambda row: parse_time(row["_start_time"]))
        rows = [row for row in all_rows if row["bilateral_classified"]]
        duration_values[session] = [row["duration_s"] for row in rows]
        gaps = [
            elapsed(a["_end_time"], b["_start_time"])
            for a, b in zip(all_rows[:-1], all_rows[1:])
            if a["bilateral_classified"] and b["bilateral_classified"]
        ]
        gap_values[session] = [value for value in gaps if value is not None]
    activity_spans = [summary_by_session[s]["p2_recorded_activity_span_s"] for s in sessions]
    ax_b0.bar(
        np.arange(4),
        activity_spans,
        color=[COLORS[s] for s in sessions],
        width=0.62,
        edgecolor="white",
        linewidth=0.5,
    )
    for x, value in enumerate(activity_spans):
        ax_b0.text(x, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=5.5)
    ax_b0.set_xticks(range(4), sessions)
    ax_b0.set_ylabel("P2 activity span (s)")
    ax_b0.set_ylim(0, max(activity_spans) * 1.16)
    ax_b0.tick_params(length=2.5, width=0.6)

    add_distribution(ax_b1, duration_values, "Action duration (s)")
    add_distribution(ax_b2, gap_values, "Inter-action gap (s)")
    panel_label(
        ax_b0,
        "b",
        "Work speed",
        "P2 activity span; bilateral-only action timing",
        subtitle_y=1.075,
    )

    # c. Hand-specific and bilateral movement-related accuracy outputs.
    cgrid = outer[1, :].subgridspec(2, 4, hspace=0.54, wspace=0.54)
    c_axes = [fig.add_subplot(cgrid[row, column]) for row in range(2) for column in range(4)]
    path_values = {
        session: {
            hand: [
                row["path_efficiency"]
                for row in hands
                if row["session"] == session
                and row["hand"] == hand
                and row["basic_bilateral_eligible"]
                and row["path_efficiency"] is not None
            ]
            for hand in ("Left", "Right")
        }
        for session in sessions
    }
    forward_values = {
        session: {
            hand: [
                1000.0 * row["signed_forward_displacement_m"]
                for row in hands
                if row["session"] == session
                and row["hand"] == hand
                and row["basic_bilateral_eligible"]
                and row["signed_forward_displacement_m"] is not None
            ]
            for hand in ("Left", "Right")
        }
        for session in sessions
    }
    lateral_values = {
        session: {
            hand: [
                1000.0 * row["lateral_excursion_m"]
                for row in hands
                if row["session"] == session
                and row["hand"] == hand
                and row["basic_bilateral_eligible"]
                and row["lateral_excursion_m"] is not None
            ]
            for hand in ("Left", "Right")
        }
        for session in sessions
    }
    direction_values = {
        session: [
            row["directional_consistency_cosine"]
            for row in actions_by_session[session]
            if row["basic_bilateral_eligible"]
            and row["directional_consistency_cosine"] is not None
        ]
        for session in sessions
    }
    adherence_values = {
        session: {
            "Left": [
                100.0 * row["left_guidance_adherence"]
                for row in actions_by_session[session]
                if row["basic_bilateral_eligible"]
                and row["left_guidance_adherence"] is not None
            ],
            "Right": [
                100.0 * row["right_guidance_adherence"]
                for row in actions_by_session[session]
                if row["basic_bilateral_eligible"]
                and row["right_guidance_adherence"] is not None
            ],
        }
        for session in sessions
    }
    path_difference_values = {
        session: [
            1000.0 * row["left_right_path_length_difference_m"]
            for row in actions_by_session[session]
            if row["basic_bilateral_eligible"]
            and row["left_right_path_length_difference_m"] is not None
        ]
        for session in sessions
    }
    separation_values = {
        session: [
            1000.0 * row["mean_abs_hand_separation_error_m"]
            for row in actions_by_session[session]
            if row["basic_bilateral_eligible"]
            and row["mean_abs_hand_separation_error_m"] is not None
        ]
        for session in sessions
    }
    bilateral_adherence_values = {
        session: [
            100.0 * row["bilateral_guidance_adherence"]
            for row in actions_by_session[session]
            if row["basic_bilateral_eligible"]
            and row["bilateral_guidance_adherence"] is not None
        ]
        for session in sessions
    }
    add_hand_distribution(c_axes[0], path_values, "Path efficiency")
    c_axes[0].set_ylim(0, 1.10)

    add_hand_distribution(
        c_axes[1],
        forward_values,
        "Signed forward\ndisplacement (mm)",
    )
    forward_all = [
        value
        for session_values in forward_values.values()
        for hand_values in session_values.values()
        for value in hand_values
    ]
    forward_low = min([0.0, *forward_all])
    forward_high = max([0.0, *forward_all])
    forward_pad = max(10.0, 0.08 * (forward_high - forward_low))
    c_axes[1].set_ylim(forward_low - forward_pad, forward_high + forward_pad)
    c_axes[1].axhline(0.0, color="#AEB5BE", lw=0.65, zorder=0)
    c_axes[1].annotate(
        "1 bilateral action: net < 0",
        xy=(3.0, -4.5),
        xytext=(1.25, 8.0),
        ha="left",
        va="center",
        fontsize=5.1,
        color="#8B552A",
        arrowprops=dict(
            arrowstyle="->",
            color="#8B552A",
            lw=0.65,
            shrinkA=1.5,
            shrinkB=2.0,
        ),
        zorder=7,
    )

    add_hand_distribution(c_axes[2], lateral_values, "Hand-specific lateral\nexcursion (mm)")
    lateral_all = [
        value
        for session_values in lateral_values.values()
        for hand_values in session_values.values()
        for value in hand_values
    ]
    lateral_upper = max([42.0, 1.12 * max(lateral_all)])
    c_axes[2].set_ylim(-0.04 * lateral_upper, lateral_upper)

    add_hand_distribution(c_axes[3], adherence_values, "Guidance-region\nadherence (%)")
    c_axes[3].set_ylim(-5, 112)
    add_distribution(c_axes[4], direction_values, "Directional consistency\n(cosine)", ylim=(-1.05, 1.08))
    add_distribution(c_axes[5], path_difference_values, "L–R path-length\ndifference (mm)")
    path_diff_all = [value for values in path_difference_values.values() for value in values]
    path_diff_upper = max(10.0, 1.12 * max(path_diff_all))
    c_axes[5].set_ylim(-0.04 * path_diff_upper, path_diff_upper)
    add_distribution(c_axes[6], separation_values, "Hand-separation\nerror (mm)", log_scale=True)
    separation_positive = [value for values in separation_values.values() for value in values if value > 0]
    c_axes[6].set_ylim(max(min(separation_positive) * 0.55, 0.0005), max(separation_positive) * 1.55)
    add_distribution(
        c_axes[7],
        bilateral_adherence_values,
        "Bilateral guidance\nadherence (%)",
        ylim=(-5, 108),
    )

    distribution_legend = [
        Line2D([], [], marker="o", linestyle="none", color="#596272", markersize=2.8, alpha=OBSERVATION_ALPHA, label="Observation"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor="#596272", markeredgecolor="white", markersize=4.2, label="Median"),
        Line2D([], [], marker="|", linestyle="none", color=IQR_COLOR, markeredgewidth=IQR_LINE_WIDTH, markersize=8, label="IQR (Q1–Q3)"),
        Line2D([], [], marker="o", linestyle="none", color="#596272", markersize=3.6, label="Left"),
        Line2D([], [], marker="^", linestyle="none", color="#596272", markersize=3.6, label="Right"),
    ]
    c_axes[3].legend(
        handles=distribution_legend,
        loc="upper right",
        bbox_to_anchor=(1.03, 1.39),
        fontsize=5.5,
        ncol=5,
        columnspacing=0.65,
        handletextpad=0.25,
        borderaxespad=0.0,
    )
    panel_label(
        c_axes[0],
        "c",
        "Movement-related accuracy",
        "Basic-eligible bilateral P2 actions only",
        heading_y=1.27,
        subtitle_y=1.105,
    )

    # d. Formal work-stability outputs at between- and within-action levels.
    dgrid = outer[2, :].subgridspec(
        1,
        2,
        width_ratios=[0.60, 0.40],
        wspace=0.34,
    )
    ax_d1 = fig.add_subplot(dgrid[0, 0])
    ax_d2 = fig.add_subplot(dgrid[0, 1])

    between_labels = [
        "Action duration CV",
        "Inter-action gap CV",
        "Forward SD (mm)",
        "Forward CV",
        "Lateral SD (mm)",
        "Lateral CV",
        "Hand separation CV",
        "Directional consistency\nIQR",
    ]
    bilateral_duration_cv: dict[str, float | None] = {}
    bilateral_gap_cv: dict[str, float | None] = {}
    for session in sessions:
        ordered_rows = sorted(
            actions_by_session[session], key=lambda row: parse_time(row["_start_time"])
        )
        bilateral_duration_cv[session] = cv(
            row["duration_s"] for row in ordered_rows if row["bilateral_classified"]
        )
        valid_gaps = [
            elapsed(previous["_end_time"], current["_start_time"])
            for previous, current in zip(ordered_rows[:-1], ordered_rows[1:])
            if previous["bilateral_classified"] and current["bilateral_classified"]
        ]
        bilateral_gap_cv[session] = cv(
            value for value in valid_gaps if value is not None
        )
    between_values = np.asarray(
        [
            [bilateral_duration_cv[s] for s in sessions],
            [bilateral_gap_cv[s] for s in sessions],
            [1000.0 * summary_by_session[s]["p2_mean_bilateral_signed_forward_displacement_sd_m"] for s in sessions],
            [summary_by_session[s]["p2_mean_bilateral_signed_forward_displacement_cv"] for s in sessions],
            [1000.0 * summary_by_session[s]["p2_mean_bilateral_lateral_excursion_sd_m"] for s in sessions],
            [summary_by_session[s]["p2_mean_bilateral_lateral_excursion_cv"] for s in sessions],
            [summary_by_session[s]["p2_mean_hand_separation_cv"] for s in sessions],
            [
                summary_by_session[s]["p2_directional_consistency_q3"]
                - summary_by_session[s]["p2_directional_consistency_q1"]
                for s in sessions
            ],
        ],
        dtype=float,
    )
    between_valid = np.isfinite(between_values)
    display_overrides: dict[tuple[int, int], str] = {}
    for column, session in enumerate(sessions):
        if not summary_by_session[session]["p2_mean_bilateral_signed_forward_displacement_cv_interpretable"]:
            between_valid[3, column] = False
            display_overrides[(3, column)] = "NI†"
        if not summary_by_session[session]["p2_mean_bilateral_lateral_excursion_cv_interpretable"]:
            between_valid[5, column] = False
            display_overrides[(5, column)] = "NA"

    within_keys = [
        "p2_speed_variability_m_s_median",
        "p2_acceleration_rms_m_s2_median",
        "p2_jerk_rms_m_s3_median",
        "p2_direction_change_rate_deg_s_median",
    ]
    within_labels = [
        "Speed var.\n(m s$^{-1}$)",
        "Acceleration\nRMS (m s$^{-2}$)",
        "Jerk RMS\n(m s$^{-3}$)",
        "Dir.-change rate\n(° s$^{-1}$)",
    ]
    within_values = np.asarray(
        [[summary_by_session[s][key] for s in sessions] for key in within_keys],
        dtype=float,
    )

    def draw_stability_matrix(
        ax: plt.Axes,
        values: np.ndarray,
        labels: list[str],
        title: str,
        formatters: list[Any],
        valid_mask: np.ndarray | None = None,
        display_overrides: dict[tuple[int, int], str] | None = None,
        row_separators: tuple[int, ...] = (),
    ) -> None:
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values) if valid_mask is None else np.isfinite(values) & valid_mask
        overrides = display_overrides or {}
        # Mixed-unit stability measures cannot share a quantitative colour
        # scale. Use neutral alternating rows and retain the raw cell values.
        background = np.empty((*values.shape, 4), dtype=float)
        for row in range(values.shape[0]):
            row_color = mpl.colors.to_rgba("#FFFFFF" if row % 2 == 0 else "#F3F5F7")
            background[row, :, :] = row_color
        background[~valid] = mpl.colors.to_rgba("#E6EAEE")
        ax.imshow(background, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(sessions)), sessions)
        ax.set_yticks(range(len(labels)), labels)
        ax.tick_params(axis="x", length=0, pad=4, labelsize=6.1)
        ytick_pad = 1.5 if ax is ax_d2 else 4
        ytick_size = 5.4 if ax is ax_d2 else 5.7
        ax.tick_params(axis="y", length=0, pad=ytick_pad, labelsize=ytick_size)
        for tick, session in zip(ax.get_xticklabels(), sessions):
            tick.set_color(COLORS[session])
            tick.set_fontweight("bold")
        ax.set_xticks(np.arange(-0.5, len(sessions), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                if (row, column) in overrides:
                    display = overrides[(row, column)]
                elif not valid[row, column]:
                    display = "NA"
                else:
                    display = formatters[row](values[row, column])
                ax.text(
                    column,
                    row,
                    display,
                    ha="center",
                    va="center",
                    fontsize=5.9,
                    color="#162338" if valid[row, column] else "#687482",
                    fontweight="bold" if (row, column) in overrides else "normal",
                )
        for separator in row_separators:
            ax.axhline(separator + 0.5, color="#AEB6C0", lw=0.75, zorder=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=7.1, fontweight="bold", loc="left", pad=6)

    draw_stability_matrix(
        ax_d1,
        between_values,
        between_labels,
        "Between-action consistency",
        [
            lambda value: "<0.001" if 0 < value < 0.001 else f"{value:.3f}",
            lambda value: "<0.001" if 0 < value < 0.001 else f"{value:.3f}",
            lambda value: f"{value:.1f}",
            lambda value: "<0.001" if 0 < value < 0.001 else f"{value:.3f}",
            lambda value: f"{value:.1f}",
            lambda value: "<0.001" if 0 < value < 0.001 else f"{value:.3f}",
            lambda value: "<0.001" if 0 < value < 0.001 else f"{value:.3f}",
            lambda value: f"{value:.3f}",
        ],
        valid_mask=between_valid,
        display_overrides=display_overrides,
        row_separators=(1, 3, 5),
    )
    draw_stability_matrix(
        ax_d2,
        within_values,
        within_labels,
        "Within-action smoothness — session median",
        [
            lambda value: f"{value:.3f}",
            lambda value: f"{value:.2f}",
            lambda value: f"{value:.1f}",
            lambda value: f"{value:.0f}",
        ],
    )
    panel_label(
        ax_d1,
        "d",
        "Work stability",
        "Bilateral actions only; cells show raw values in row-specific units",
        heading_y=1.30,
        subtitle_y=1.18,
    )
    ax_d1.text(
        0.0,
        -0.105,
        "† U2 included one bilateral action with negative net forward displacement (1/18 actions). The forward-displacement CV was therefore not interpreted.",
        transform=ax_d1.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color="#5C6470",
        clip_on=False,
    )
    # e. Descriptive within-action movement pattern: all eligible actions contribute.
    egrid = outer[3, :].subgridspec(1, 4, wspace=0.18)
    e_axes = [fig.add_subplot(egrid[0, idx]) for idx in range(4)]
    q1_min = min(float(np.min(lateral_profiles[s]["q1"])) for s in sessions)
    q3_max = max(float(np.max(lateral_profiles[s]["q3"])) for s in sessions)
    span = max(q3_max - q1_min, 1.0)
    y_min = 5.0 * math.floor((q1_min - 0.10 * span) / 5.0)
    y_max = 5.0 * math.ceil((q3_max + 0.10 * span) / 5.0)
    for idx, (ax, session) in enumerate(zip(e_axes, sessions)):
        profile = lateral_profiles[session]
        x_percent = 100.0 * profile["grid"]
        ax.fill_between(
            x_percent,
            profile["q1"],
            profile["q3"],
            color=COLORS[session],
            alpha=0.20,
            linewidth=0,
        )
        ax.plot(x_percent, profile["median"], color=COLORS[session], lw=1.45)
        ax.axhline(0, color="#7A828E", lw=0.65, ls="--", zorder=0)
        ax.set_xlim(0, 100)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([0, 50, 100])
        ax.set_title(
            f"{session}   n={len(profile['traces'])} actions",
            fontsize=7.0,
            color=COLORS[session],
            fontweight="bold",
            pad=4,
        )
        if idx == 0:
            ax.set_ylabel("Lateral displacement from\naction start (mm)")
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(length=2.5, width=0.6)
    panel_label(
        e_axes[0],
        "e",
        "Descriptive within-action trajectory profiles",
        "Bilateral-midpoint lateral displacement; line, median; shading, IQR; not a scored outcome",
        heading_y=1.30,
        subtitle_y=1.205,
    )

    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", color=COLORS[s], markersize=5, label=f"{s}  {profile}")
        for s, _, profile in SESSION_SPECS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.127, 0.982),
        ncol=4,
        fontsize=5.8,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.text(
        0.54,
        0.022,
        "Normalized action time (%)",
        fontsize=7.2,
        ha="center",
        va="center",
    )
    prefix = OUT / FIGURE_STEM
    fig.savefig(prefix.with_suffix(".svg"))
    fig.savefig(prefix.with_suffix(".pdf"))
    fig.savefig(prefix.with_suffix(".png"), dpi=400)
    fig.savefig(
        prefix.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def plot_supplementary_stability(hands: list[dict[str, Any]]) -> None:
    sessions = [item[0] for item in SESSION_SPECS]
    metrics = [
        ("speed_variability_m_s", "Speed variability (m s$^{-1}$)", False),
        ("acceleration_rms_m_s2", "Acceleration RMS (m s$^{-2}$)", True),
        ("jerk_rms_m_s3", "Jerk RMS (m s$^{-3}$)", True),
        ("direction_change_rate_deg_s", "Direction-change rate (° s$^{-1}$)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.86, wspace=0.34, hspace=0.55)
    fig.text(
        0.10,
        0.96,
        "Higher-order within-action movement dynamics",
        fontsize=11,
        fontweight="bold",
        color="#162338",
    )
    fig.text(
        0.10,
        0.925,
        "Eligible bilateral hand–action trajectories contained ≥10 synchronized samples; derivatives were unsmoothed.",
        fontsize=6.7,
        color="#5C6470",
    )
    for letter, ax, (key, ylabel, log_scale) in zip("abcd", axes.flat, metrics):
        values = {
            session: [
                row[key]
                for row in hands
                if row["session"] == session
                and row["higher_order_eligible"]
                and row[key] is not None
            ]
            for session in sessions
        }
        add_distribution(ax, values, ylabel, log_scale=log_scale)
        counts = [len(values[session]) for session in sessions]
        for idx, count in enumerate(counts):
            ax.text(idx, -0.15, f"n={count}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.6, color="#5C6470")
        panel_label(ax, letter, ylabel.split(" (")[0], None)
    fig.text(
        0.10,
        0.025,
        "Points are hand–action trajectories nested within four technical sessions; diamonds and bars show session medians and IQRs. These outputs are descriptive and not validated tremor or clinical measures.",
        fontsize=5.7,
        color="#5C6470",
    )
    prefix = OUT / SUPPLEMENTARY_STEM
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def render_outputs(
    summaries: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    hands: list[dict[str, Any]],
    lateral_profiles: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    figure_stem: str,
    supplementary_stem: str,
    session_specs: list[tuple[str, Path, str]],
    parse_time_fn: Callable,
    elapsed_fn: Callable,
    cv_fn: Callable,
) -> None:
    """Render the unchanged base plots with explicit analysis-run settings.

    The helpers are the original analysis functions, passed explicitly to
    keep a single implementation and avoid circular imports. Plotting does
    not read or rewrite source JSON records or the numerical output tables.
    """
    global OUT, FIGURE_STEM, SUPPLEMENTARY_STEM, SESSION_SPECS
    global parse_time, elapsed, cv
    OUT = Path(output_dir)
    FIGURE_STEM = figure_stem
    SUPPLEMENTARY_STEM = supplementary_stem
    SESSION_SPECS = list(session_specs)
    parse_time = parse_time_fn
    elapsed = elapsed_fn
    cv = cv_fn
    plot_main_figure(summaries, actions, hands, lateral_profiles)
    plot_supplementary_stability(hands)
