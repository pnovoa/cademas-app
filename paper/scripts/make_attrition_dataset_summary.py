import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cademas_ml_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = ROOT / "example_attrition" / "data" / "cases_atttrition.csv"
FIG_DIR = ROOT / "paper" / "figures"
OUT_PDF = FIG_DIR / "attrition_dataset_summary.pdf"
OUT_PNG = FIG_DIR / "attrition_dataset_summary.png"
ATTRITION_ORDER = ["No", "Yes"]
ATTRITION_COLORS = {"No": "#c8d1d8", "Yes": "#e13f40"}


def apply_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="black", linestyle=":", linewidth=0.6, alpha=0.35, zorder=0)


def annotate_vertical_bars(ax, bars, fmt="{:.0f}"):
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def stacked_counts_by_attrition(
    ax,
    df,
    category_col,
    title,
    horizontal=False,
    bar_width=0.8,
    group_spacing=1.0,
    side_pad=None,
):
    counts = pd.crosstab(df[category_col], df["attrition"]).reindex(columns=ATTRITION_ORDER, fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    if horizontal:
        labels = counts.index
        y = np.arange(len(labels))
        left = np.zeros(len(labels))
        for attrition in ATTRITION_ORDER:
            values = counts[attrition].to_numpy()
            bars = ax.barh(
                y,
                values,
                left=left,
                color=ATTRITION_COLORS[attrition],
                edgecolor="black",
                linewidth=0.6,
                label=f"Attrition = {attrition}",
                zorder=3,
            )
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{int(value)}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=7,
                    )
            left += values
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Cases")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        apply_grid(ax, axis="x")
    else:
        labels = [str(label).replace(" & ", "\n& ") for label in counts.index]
        x = np.arange(len(labels)) * group_spacing
        bottom = np.zeros(len(labels))
        for attrition in ATTRITION_ORDER:
            values = counts[attrition].to_numpy()
            bars = ax.bar(
                x,
                values,
                width=bar_width,
                bottom=bottom,
                color=ATTRITION_COLORS[attrition],
                edgecolor="black",
                linewidth=0.6,
                label=f"Attrition = {attrition}",
                zorder=3,
            )
            for bar, value, base in zip(bars, values, bottom):
                if value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        base + value / 2,
                        f"{int(value)}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=7,
                    )
            bottom += values
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", rotation=20)
        if side_pad is None:
            side_pad = bar_width / 2 + 0.08
        ax.set_xlim(x[0] - side_pad, x[-1] + side_pad)
        ax.set_ylabel("Cases")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        apply_grid(ax, axis="y")


def box_by_attrition(ax, df, column, title, ylabel):
    groups = [df.loc[df["attrition"] == label, column].dropna() for label in ATTRITION_ORDER]
    bp = ax.boxplot(groups, tick_labels=ATTRITION_ORDER, patch_artist=True, widths=0.36)
    for patch, attrition in zip(bp["boxes"], ATTRITION_ORDER):
        patch.set_facecolor(ATTRITION_COLORS[attrition])
        patch.set_edgecolor("black")
        patch.set_zorder(3)
    for key in ["medians", "whiskers", "caps"]:
        for item in bp[key]:
            item.set_color("black")
            item.set_zorder(4)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Attrition label")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    apply_grid(ax, axis="y")


def grouped_box_by_attrition(ax, df, columns, labels, title, ylabel):
    positions = []
    data = []
    colors = []
    tick_positions = []
    for idx, column in enumerate(columns, start=1):
        no_pos = idx - 0.18
        yes_pos = idx + 0.18
        positions.extend([no_pos, yes_pos])
        tick_positions.append(idx)
        for attrition in ATTRITION_ORDER:
            data.append(df.loc[df["attrition"] == attrition, column].dropna())
            colors.append(ATTRITION_COLORS[attrition])

    bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_zorder(3)
    for key in ["medians", "whiskers", "caps"]:
        for item in bp[key]:
            item.set_color("black")
            item.set_zorder(4)
    ax.set_xticks(tick_positions, labels)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    apply_grid(ax, axis="y")


def grouped_means_by_attrition(ax, df, columns, labels, title, ylabel, ylim=None):
    means = df.groupby("attrition")[columns].mean().reindex(ATTRITION_ORDER)
    x = np.arange(len(columns))
    width = 0.36
    for offset, attrition in zip([-width / 2, width / 2], ATTRITION_ORDER):
        bars = ax.bar(
            x + offset,
            means.loc[attrition].to_numpy(),
            width=width,
            color=ATTRITION_COLORS[attrition],
            edgecolor="black",
            linewidth=0.6,
            label=f"Attrition = {attrition}",
            zorder=3,
        )
        annotate_vertical_bars(ax, bars, fmt="{:.1f}")
    ax.set_xticks(x, labels)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    apply_grid(ax, axis="y")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, sep=";")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.8), constrained_layout=True)
    axes = axes.ravel()

    stacked_counts_by_attrition(
        axes[0],
        df,
        "department",
        "(a) Department",
        bar_width=0.2,
        group_spacing=0.55,
        side_pad=0.35,
    )
    stacked_counts_by_attrition(axes[1], df, "job_role", "(b) Job role", horizontal=True)
    box_by_attrition(axes[2], df, "age", "(c) Age", "Age")
    box_by_attrition(
        axes[3],
        df,
        "monthly_income",
        "(d) Monthly income",
        "Monthly income",
    )
    grouped_box_by_attrition(
        axes[4],
        df,
        ["years_at_company", "total_working_years"],
        ["Years at\ncompany", "Total working\nyears"],
        "(e) Tenure and experience",
        "Years",
    )
    grouped_means_by_attrition(
        axes[5],
        df,
        ["environment_satisfaction", "job_involvement", "job_satisfaction"],
        ["Environment\nsatisfaction", "Job\ninvolvement", "Job\nsatisfaction"],
        "(f) Satisfaction and involvement",
        "Mean score",
        ylim=(0, 4),
    )
    stacked_counts_by_attrition(
        axes[6],
        df,
        "over_time",
        "(g) Overtime",
        bar_width=0.2,
        group_spacing=0.55,
        side_pad=0.35,
    )
    stacked_counts_by_attrition(
        axes[7],
        df,
        "business_travel",
        "(h) Business travel",
        bar_width=0.2,
        group_spacing=0.55,
        side_pad=0.35,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ATTRITION_COLORS[attrition])
        for attrition in ATTRITION_ORDER
    ]
    fig.legend(
        handles,
        [f"Attrition = {attrition}" for attrition in ATTRITION_ORDER],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
