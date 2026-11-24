#!/usr/bin/env python3
"""Plot selected official scores across seeds as a paired dot chart."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(TASK_ROOT / "tools"))

from multiseed_results import DISPLAY_NAMES, filter_frame, load_sweep_manifests

DEFAULT_ORDER = [
    "baseline",
    "cife",
    "phase1_random",
    "phase2_energy",
    "phase3_activation",
    "afri_direct",
    "afri_inverse",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=["slot_uniform"])
    return parser.parse_args()


def make_figure(manifests, output_prefix, exclude):
    scores = load_sweep_manifests(manifests)
    scores = filter_frame(scores, exclude_configurations=exclude)
    seeds = sorted(scores["seed"].unique())
    if len(seeds) != 2:
        raise ValueError("This paired figure requires exactly two seeds; found %s" % seeds)

    matrix = scores.pivot(index="configuration", columns="seed", values="fmeasure_harmonic")
    order = [configuration for configuration in DEFAULT_ORDER if configuration in matrix.index]
    order.extend(sorted(set(matrix.index) - set(order)))
    matrix = matrix.loc[order]

    seed_a, seed_b = seeds
    color_a = "#0072B2"  # colorblind-safe blue
    color_b = "#D55E00"  # colorblind-safe vermillion

    fig, ax = plt.subplots(figsize=(7.15, 3.7))
    for y, configuration in enumerate(order):
        first = matrix.loc[configuration, seed_a]
        second = matrix.loc[configuration, seed_b]
        ax.plot([first, second], [y, y], color="#A6A6A6", linewidth=1.4, zorder=1)
        ax.scatter(first, y, color=color_a, marker="o", s=48, zorder=2)
        ax.scatter(second, y, color=color_b, marker="s", s=48, zorder=2)
        ax.text(first, y - 0.18, "%.2f" % first, color=color_a, ha="center", va="bottom", fontsize=7.5)
        ax.text(second, y + 0.18, "%.2f" % second, color=color_b, ha="center", va="top", fontsize=7.5)

    ax.scatter([], [], color=color_a, marker="o", s=48, label="Seed %s" % seed_a)
    ax.scatter([], [], color=color_b, marker="s", s=48, label="Seed %s" % seed_b)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY_NAMES.get(configuration, configuration) for configuration in order])
    ax.invert_yaxis()
    ax.set_xlabel(r"Official $F_{\mathrm{harm}}$ (0--100 scale)")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_title("Configuration performance across two random seeds", fontsize=10)
    fig.tight_layout()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return matrix


def main():
    args = parse_args()
    matrix = make_figure(args.manifest, args.output_prefix, args.exclude)
    print(matrix.to_string())
    print("Wrote %s and %s" % (
        args.output_prefix.with_suffix(".pdf"),
        args.output_prefix.with_suffix(".png"),
    ))


if __name__ == "__main__":
    main()
