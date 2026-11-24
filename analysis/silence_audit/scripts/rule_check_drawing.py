import matplotlib.pyplot as plt
import pandas as pd
SHARED_FILENAME = "BUK5_20180921_015906a.wav"

def save_figure(*args):
    raise ValueError("Install the formatting save callback before drawing")

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
