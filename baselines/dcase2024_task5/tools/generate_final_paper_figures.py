#!/usr/bin/env python3
"""Regenerate the three diagnostic figures used by the final paper.

The script reads the verified three-seed intervention tables and oracle endpoint
summary. It does not run training or the event matcher.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pandas as pd
import seaborn as sns

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK_ROOT.parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from tools.multiseed_results import load_silencing_directories, recording_recurrence

SHARED_FILENAME = "BUK5_20180921_015906a.wav"
CONFIGURATIONS = [
    "baseline", "cife", "phase1_random", "slot_uniform",
    "phase2_energy", "phase3_activation", "afri_direct", "afri_inverse",
]
CONFIG_LABELS = ["Base", "CIFE", "Random", "Slot", "Energy", "Activ.", "AFRI-D", "AFRI-I"]


def save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def rule_check_figure(subset_effects: pd.DataFrame, output_dir: Path) -> None:
    data = subset_effects[
        (subset_effects["evaluator_variant"] == "corrected")
        & (~subset_effects["originally_silent"])
        & (subset_effects["target_filename"] != SHARED_FILENAME)
    ].copy()
    data["precision_r"] = data["file_tp"] / (data["file_tp"] + data["file_fp"])
    data["threshold_margin"] = data["precision_r"] - data["original_subset_f"] / 200.0
    substantive = data[data["file_tp"] > 0]
    false_alarm_only = data[data["file_tp"] == 0]

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.scatter(
        substantive["threshold_margin"], substantive["delta_subset_f"],
        s=9, alpha=0.50, color="#0072B2", edgecolors="none",
        label=r"$\geq$1 correct detection",
    )
    ax.scatter(
        false_alarm_only["threshold_margin"], false_alarm_only["delta_subset_f"],
        s=18, alpha=0.9, color="#D55E00", marker="x", linewidths=0.9,
        label="False alarms only",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=0.5, linscale=0.8)
    ax.set_xlabel(r"Prediction precision $-$ original $F_s/2$", fontsize=8)
    ax.set_ylabel("Change in subset F1 (points)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", handletextpad=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.35)
    save_figure(fig, output_dir, "fig_rule_check")


def recurrence_figure(
    interventions: pd.DataFrame, output_dir: Path, evaluator_variant: str
) -> None:
    data = interventions[
        (interventions["evaluator_variant"] == evaluator_variant)
        & (~interventions["originally_silent"])
        & (interventions["target_tp"] > 0)
        & (interventions["target_filename"] != SHARED_FILENAME)
    ].copy()
    recurrence = recording_recurrence(
        data, evaluator_variant, require_true_positive=True
    )
    universal = recurrence[
        recurrence["n_aggregate_improvements"] == recurrence["n_interventions"]
    ].sort_values(["mean_delta", "max_delta"], ascending=False)
    filenames = list(universal.index)
    seeds = sorted(data["seed"].unique())
    seed_names = {seed: f"Seed {chr(ord('A') + index)}" for index, seed in enumerate(seeds)}
    data["column_key"] = data.apply(
        lambda row: f"{int(row['seed'])}|{row['configuration']}", axis=1
    )
    columns = [f"{seed}|{config}" for seed in seeds for config in CONFIGURATIONS]
    matrix = data.pivot_table(
        index="target_filename", columns="column_key",
        values="delta_fmeasure_harmonic", aggfunc="first",
    ).reindex(index=filenames, columns=columns)

    # A wide, two-column heatmap. Values are printed in the cells, so a
    # separate colorbar would duplicate information and consume useful width.
    fig, ax = plt.subplots(figsize=(7.15, 3.55))
    ax.set_facecolor("#d9d9d9")
    green_map = LinearSegmentedColormap.from_list(
        "white_to_green", ["#ffffff", "#d9f0d3", "#78c679", "#238443"]
    )
    vmax = float(matrix.max().max())
    sns.heatmap(
        matrix, mask=matrix.isna(), cmap=green_map, vmin=0, vmax=vmax,
        linewidths=0.45, linecolor="#bdbdbd", cbar=False, ax=ax,
    )
    normalizer = Normalize(vmin=0, vmax=vmax)
    for row_index, filename in enumerate(matrix.index):
        for column_index, column in enumerate(matrix.columns):
            value = matrix.loc[filename, column]
            if pd.isna(value):
                continue
            text_color = "white" if normalizer(value) > 0.58 else "#1a1a1a"
            ax.text(
                column_index + 0.5, row_index + 0.5, f"{value:.2f}",
                ha="center", va="center", fontsize=6.7, color=text_color,
                fontweight="bold" if value >= 1.0 else "normal",
            )
    ax.set_xticks([position + 0.5 for position in range(len(columns))])
    ax.set_xticklabels(
        CONFIG_LABELS * len(seeds), rotation=42, ha="right",
        rotation_mode="anchor", fontsize=7.2,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xlabel("Configuration within seed", fontsize=8.5)
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0)
    for boundary in range(1, len(seeds)):
        ax.axvline(boundary * len(CONFIGURATIONS), color="black", linewidth=1.35)
    for index, seed in enumerate(seeds):
        center = (index * len(CONFIGURATIONS) + len(CONFIGURATIONS) / 2) / len(columns)
        ax.text(
            center, 1.025, seed_names[seed], transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    fig.tight_layout(pad=0.45, rect=(0, 0, 1, 0.965))
    save_figure(fig, output_dir, "fig_tpos_recurrence_heatmap")


def score_hit_figure(endpoints: pd.DataFrame, output_dir: Path) -> None:
    coverage_columns = (
        ("original_coverage", "oracle_coverage")
        if "original_coverage" in endpoints.columns
        else ("original_hit_rate", "oracle_hit_rate")
    )
    original_coverage, oracle_coverage = coverage_columns
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.25), sharex=True)
    x_values = [0, 1]
    for _, row in endpoints.iterrows():
        axes[0].plot(
            x_values, [row["original_f"], row["oracle_f"]],
            color="#7AA6C2", alpha=0.55, linewidth=0.65,
        )
        axes[1].plot(
            x_values, [row[original_coverage], row[oracle_coverage]],
            color="#7AA6C2", alpha=0.55, linewidth=0.65,
        )
    means = endpoints[
        ["original_f", "oracle_f", original_coverage, oracle_coverage]
    ].mean()
    axes[0].plot(
        x_values, [means["original_f"], means["oracle_f"]],
        "-o", color="#D55E00", linewidth=2.2, markersize=4, label="Mean",
    )
    axes[1].plot(
        x_values, [means[original_coverage], means[oracle_coverage]],
        "-o", color="#D55E00", linewidth=2.2, markersize=4,
    )
    axes[0].set_ylabel(r"Official $F_{\mathrm{harm}}$", fontsize=8)
    axes[1].set_ylabel("Recording coverage (%)", fontsize=8)
    axes[1].set_xticks(x_values, ["Original", "After label-informed\ndeletion"])
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(pad=0.45, h_pad=0.35)
    save_figure(fig, output_dir, "fig_score_hit_tradeoff")


def parse_args() -> argparse.Namespace:
    analysis_root = TASK_ROOT / "analysis"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--silencing-dir", action="append", type=Path,
        default=[analysis_root / "silencing_seeds1234_2026", analysis_root / "silencing_seed3009"],
    )
    parser.add_argument(
        "--oracle-endpoints", type=Path,
        default=analysis_root / "paper_outputs_three_seeds" / "tables" / "oracle_score_coverage_endpoints_released.csv",
    )
    parser.add_argument(
        "--recurrence-variant", choices=("released", "corrected"),
        default="released",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "paper" / "final_framing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    silencing = load_silencing_directories(args.silencing_dir)
    endpoints = pd.read_csv(args.oracle_endpoints)
    sns.set_theme(style="whitegrid", context="paper")
    rule_check_figure(silencing["subset_effects"], args.output_dir)
    recurrence_figure(
        silencing["interventions"], args.output_dir, args.recurrence_variant
    )
    score_hit_figure(endpoints, args.output_dir)
    print(f"Wrote paper figures to {args.output_dir}")


if __name__ == "__main__":
    main()
