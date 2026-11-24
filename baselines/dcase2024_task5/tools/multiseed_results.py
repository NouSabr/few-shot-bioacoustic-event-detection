"""Reusable loaders and descriptive summaries for multi-seed paper results."""

from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_CONFIGURATION_ORDER = [
    "baseline",
    "cife",
    "phase1_random",
    "slot_uniform",
    "phase2_energy",
    "phase3_activation",
    "afri_direct",
    "afri_inverse",
]

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "cife": "CIFE-inspired",
    "phase1_random": "Random",
    "slot_uniform": "Slot uniform",
    "phase2_energy": "Energy",
    "phase3_activation": "Activation",
    "afri_direct": "AFRI direct",
    "afri_inverse": "AFRI inverse",
}

BOOLEAN_COLUMNS = {
    "originally_silent",
    "improves_any_subset",
    "improves_all_affected_subsets",
    "improves_official",
    "forfeits_true_positives",
    "observed_subset_improvement",
    "condition_holds",
    "condition_matches_observation",
}


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().eq("true")


def _deduplicate_or_raise(frame: pd.DataFrame, keys: Sequence[str], label: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    duplicated = frame.duplicated(list(keys), keep=False)
    if not duplicated.any():
        return frame

    conflicts = []
    retained = []
    for _, group in frame[duplicated].groupby(list(keys), dropna=False):
        comparison = group.drop(columns=["source_path"], errors="ignore").drop_duplicates()
        if len(comparison) > 1:
            conflicts.append(group[list(keys)].iloc[0].to_dict())
        retained.append(group.iloc[[0]])
    if conflicts:
        raise ValueError("Conflicting duplicate %s rows: %s" % (label, conflicts[:10]))

    unique = frame[~duplicated]
    return pd.concat([unique] + retained, ignore_index=True)


def load_sweep_manifests(paths: Sequence[Path]) -> pd.DataFrame:
    """Load conventional selected scores from one or more collector manifests."""
    rows: List[Dict] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        for run in manifest.get("runs", []):
            if not str(run.get("artifact_status", "")).startswith("complete_"):
                continue
            scores = run.get("logged_scores") or {}
            selection = run.get("selection") or {}
            rows.append(
                {
                    "configuration": run.get("configuration"),
                    "seed": run.get("seed"),
                    "artifact_status": run.get("artifact_status"),
                    "fmeasure_harmonic": scores.get("fmeasure_harmonic"),
                    "fmeasure_arithmetic": scores.get("fmeasure_arithmetic"),
                    "precision_harmonic": _percent(scores.get("precision_harmonic")),
                    "recall_harmonic": _percent(scores.get("recall_harmonic")),
                    "detection_threshold": selection.get("detection_threshold"),
                    "postprocessing_type": selection.get("postprocessing_type"),
                    "postprocessing_threshold": selection.get("postprocessing_threshold"),
                    "source_path": str(path),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No complete runs found in sweep manifests: %s" % list(paths))
    frame = _deduplicate_or_raise(frame, ["configuration", "seed"], "sweep")
    return add_display_names(frame)


def _percent(value):
    if value is None:
        return None
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def load_silencing_directories(paths: Sequence[Path]) -> Dict[str, pd.DataFrame]:
    """Load and concatenate counterfactual result directories."""
    names = {
        "interventions": "silencing_interventions.csv",
        "subset_effects": "silencing_subset_effects.csv",
        "summary": "silencing_summary.csv",
        "skipped": "skipped_runs.csv",
    }
    loaded: Dict[str, List[pd.DataFrame]] = {key: [] for key in names}
    for value in paths:
        directory = Path(value).expanduser().resolve()
        for key, filename in names.items():
            path = directory / filename
            if not path.is_file():
                if key == "skipped":
                    continue
                raise FileNotFoundError("Missing counterfactual output: %s" % path)
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                # A successful analysis has no skipped rows; older counterfactual
                # outputs represented that as a zero-byte/blank CSV. Treat it as
                # an empty optional table rather than failing notebook loading.
                frame = pd.DataFrame()
            if not frame.empty:
                frame["source_path"] = str(path)
            loaded[key].append(frame)

    frames: Dict[str, pd.DataFrame] = {}
    for key, parts in loaded.items():
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for column in BOOLEAN_COLUMNS & set(frame.columns):
            frame[column] = as_bool(frame[column])
        frames[key] = add_display_names(frame)

    # Current counterfactual outputs store target counts directly. Backfill
    # legacy outputs from the per-subset table so existing completed sweeps can
    # be stratified without rerunning the evaluator.
    interventions = frames["interventions"]
    subset_effects = frames["subset_effects"]
    if (
        not interventions.empty
        and "target_tp" not in interventions
        and not subset_effects.empty
        and {"file_tp", "file_fp", "file_fn"}.issubset(subset_effects.columns)
    ):
        keys = ["configuration", "seed", "evaluator_variant", "target_filename"]
        counts = subset_effects[keys + ["file_tp", "file_fp", "file_fn"]].drop_duplicates()
        conflict = counts.duplicated(keys, keep=False)
        if conflict.any():
            grouped = counts[conflict].groupby(keys, dropna=False)
            bad = [key for key, group in grouped if len(group.drop_duplicates()) > 1]
            if bad:
                raise ValueError("Conflicting legacy target counts: %s" % bad[:10])
            counts = counts.drop_duplicates(keys)
        counts = counts.rename(
            columns={"file_tp": "target_tp", "file_fp": "target_fp", "file_fn": "target_fn"}
        )
        interventions = interventions.merge(counts, on=keys, how="left", validate="one_to_one")
        if interventions[["target_tp", "target_fp", "target_fn"]].isna().any().any():
            raise ValueError("Could not backfill target counts for every intervention")
        interventions["forfeits_true_positives"] = interventions["target_tp"] > 0
        denominator = interventions["target_tp"] + interventions["target_fp"]
        interventions["target_prediction_precision"] = (
            interventions["target_tp"] / denominator.where(denominator != 0, 1)
        )
        frames["interventions"] = interventions

    frames["interventions"] = _deduplicate_or_raise(
        frames["interventions"],
        ["configuration", "seed", "evaluator_variant", "target_filename"],
        "intervention",
    )
    frames["subset_effects"] = _deduplicate_or_raise(
        frames["subset_effects"],
        ["configuration", "seed", "evaluator_variant", "target_filename", "subset"],
        "subset-effect",
    )
    return frames


def add_display_names(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "configuration" not in frame:
        return frame
    result = frame.copy()
    result["configuration_label"] = result["configuration"].map(DISPLAY_NAMES).fillna(
        result["configuration"]
    )
    return result


def filter_frame(
    frame: pd.DataFrame,
    include_configurations: Optional[Sequence[str]] = None,
    exclude_configurations: Optional[Sequence[str]] = None,
    include_seeds: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    result = frame.copy()
    if include_configurations:
        result = result[result["configuration"].isin(include_configurations)]
    if exclude_configurations:
        result = result[~result["configuration"].isin(exclude_configurations)]
    if include_seeds:
        result = result[result["seed"].isin(include_seeds)]
    return result.reset_index(drop=True)


def filter_all(
    scores: pd.DataFrame,
    silencing: Mapping[str, pd.DataFrame],
    include_configurations: Optional[Sequence[str]] = None,
    exclude_configurations: Optional[Sequence[str]] = None,
    include_seeds: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    filtered_scores = filter_frame(
        scores, include_configurations, exclude_configurations, include_seeds
    )
    filtered_silencing = {}
    for key, frame in silencing.items():
        if frame.empty or "configuration" not in frame:
            filtered_silencing[key] = frame.copy()
        else:
            filtered_silencing[key] = filter_frame(
                frame, include_configurations, exclude_configurations, include_seeds
            )
    return filtered_scores, filtered_silencing


def configuration_order(frame: pd.DataFrame, include_baseline: bool = True) -> List[str]:
    present = set(frame["configuration"].dropna().unique())
    ordered = [value for value in DEFAULT_CONFIGURATION_ORDER if value in present]
    ordered.extend(sorted(present - set(ordered)))
    if not include_baseline:
        ordered = [value for value in ordered if value != "baseline"]
    return ordered


def paired_deltas(
    scores: pd.DataFrame,
    value_column: str = "fmeasure_harmonic",
    baseline: str = "baseline",
) -> pd.DataFrame:
    matrix = scores.pivot(index="seed", columns="configuration", values=value_column)
    if baseline not in matrix:
        raise ValueError("Baseline configuration %r is absent after filtering" % baseline)
    return matrix.subtract(matrix[baseline], axis=0).drop(columns=[baseline])


def paired_delta_summary(delta: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean_delta": delta.mean(),
            "sd_delta": delta.std(ddof=1),
            "min_delta": delta.min(),
            "max_delta": delta.max(),
            "positive_seeds": (delta > 0).sum(),
            "negative_seeds": (delta < 0).sum(),
            "n_seeds": delta.notna().sum(),
        }
    ).sort_values("mean_delta", ascending=False)


def conventional_summary(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["configuration", "configuration_label"], as_index=False)
        .agg(
            mean_fmeasure_harmonic=("fmeasure_harmonic", "mean"),
            sd_fmeasure_harmonic=("fmeasure_harmonic", "std"),
            min_fmeasure_harmonic=("fmeasure_harmonic", "min"),
            max_fmeasure_harmonic=("fmeasure_harmonic", "max"),
            mean_fmeasure_arithmetic=("fmeasure_arithmetic", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("mean_fmeasure_harmonic", ascending=False)
    )


def intervention_prevalence(interventions: pd.DataFrame) -> pd.DataFrame:
    eligible = interventions[~interventions["originally_silent"]].copy()
    grouped = (
        eligible.groupby(
            ["configuration", "configuration_label", "seed", "evaluator_variant"],
            as_index=False,
        )
        .agg(
            eligible_interventions=("target_filename", "size"),
            subset_improvements=("improves_any_subset", "sum"),
            aggregate_improvements=("improves_official", "sum"),
            mean_delta_harmonic=("delta_fmeasure_harmonic", "mean"),
            max_delta_harmonic=("delta_fmeasure_harmonic", "max"),
        )
    )
    grouped["subset_improvement_rate"] = (
        grouped["subset_improvements"] / grouped["eligible_interventions"]
    )
    grouped["aggregate_improvement_rate"] = (
        grouped["aggregate_improvements"] / grouped["eligible_interventions"]
    )
    return grouped


def stratified_intervention_summary(interventions: pd.DataFrame) -> pd.DataFrame:
    """Summarize eligible interventions by evaluator and whether TP are forfeited.

    ``t > 0`` is the substantive recording-abandonment case: emptying the target
    discards at least one matched event. ``t = 0`` removes only evaluator-counted
    false positives and is reported separately.
    """
    required = {"target_tp", "originally_silent", "evaluator_variant"}
    missing = required - set(interventions.columns)
    if missing:
        raise ValueError(
            "Intervention table lacks TP-stratification columns: %s. "
            "Regenerate it with the current silencing_counterfactual.py."
            % sorted(missing)
        )
    eligible = interventions[~interventions["originally_silent"]].copy()
    eligible["target_status"] = eligible["target_tp"].fillna(0).gt(0).map(
        {True: "t > 0", False: "t = 0"}
    )
    grouped = (
        eligible.groupby(["evaluator_variant", "target_status"], as_index=False)
        .agg(
            eligible_interventions=("target_filename", "size"),
            aggregate_improvements=("improves_official", "sum"),
            mean_delta_harmonic=("delta_fmeasure_harmonic", "mean"),
            max_delta_harmonic=("delta_fmeasure_harmonic", "max"),
        )
    )
    grouped["aggregate_improvement_rate"] = (
        grouped["aggregate_improvements"] / grouped["eligible_interventions"]
    )
    return grouped


def evaluator_summary(interventions: pd.DataFrame) -> pd.DataFrame:
    eligible = interventions[~interventions["originally_silent"]].copy()
    grouped = (
        eligible.groupby("evaluator_variant", as_index=False)
        .agg(
            eligible_interventions=("target_filename", "size"),
            subset_improvements=("improves_any_subset", "sum"),
            aggregate_improvements=("improves_official", "sum"),
            mean_delta_harmonic=("delta_fmeasure_harmonic", "mean"),
            max_delta_harmonic=("delta_fmeasure_harmonic", "max"),
        )
    )
    grouped["subset_improvement_rate"] = (
        grouped["subset_improvements"] / grouped["eligible_interventions"]
    )
    grouped["aggregate_improvement_rate"] = (
        grouped["aggregate_improvements"] / grouped["eligible_interventions"]
    )
    return grouped


def recording_recurrence(
    interventions: pd.DataFrame,
    evaluator_variant: str = "corrected",
    require_true_positive: bool = False,
) -> pd.DataFrame:
    eligible = interventions[
        (~interventions["originally_silent"])
        & (interventions["evaluator_variant"] == evaluator_variant)
    ].copy()
    if require_true_positive:
        if "target_tp" not in eligible:
            raise ValueError(
                "target_tp is required for t > 0 recurrence; regenerate intervention outputs"
            )
        eligible = eligible[eligible["target_tp"] > 0]
    positive = eligible[eligible["improves_official"]]
    base = eligible.groupby("target_filename").agg(
        n_interventions=("target_filename", "size"),
        mean_delta=("delta_fmeasure_harmonic", "mean"),
        max_delta=("delta_fmeasure_harmonic", "max"),
    )
    gains = positive.groupby("target_filename").agg(
        n_aggregate_improvements=("target_filename", "size"),
        positive_seeds=("seed", "nunique"),
        positive_configurations=("configuration", "nunique"),
    )
    return (
        base.join(gains)
        .fillna(0)
        .sort_values(
            ["positive_seeds", "n_aggregate_improvements", "max_delta"],
            ascending=False,
        )
    )


def oracle_removal_bound(
    subset_effects: pd.DataFrame,
    evaluator_variant: str = "corrected",
    min_eval_value: float = 1e-5,
) -> pd.DataFrame:
    """Exact oracle upper bound from jointly silencing any set of recordings.

    Files that belong to multiple subsets share one keep/silence decision. Given
    those coupled decisions, all remaining subset optimizations are independent
    and small enough for exhaustive enumeration on the Task 5 validation split.
    """
    required = {
        "configuration", "seed", "evaluator_variant", "target_filename",
        "subset", "file_tp", "file_fp", "file_fn", "silenced_file_fn",
        "original_subset_f",
    }
    missing = required - set(subset_effects.columns)
    if missing:
        raise ValueError("Subset-effect table lacks oracle-bound columns: %s" % sorted(missing))

    source = subset_effects[subset_effects["evaluator_variant"] == evaluator_variant].copy()
    rows = []

    def fmeasure(counts):
        tp, fp, fn = counts
        return max(2.0 * tp / (2.0 * tp + fp + fn), min_eval_value) if tp else min_eval_value

    def hmean(values):
        values = [max(float(v), min_eval_value) for v in values]
        return len(values) / sum(1.0 / value for value in values)

    for (configuration, seed), run in source.groupby(["configuration", "seed"]):
        memberships = run.groupby("target_filename")["subset"].agg(lambda x: tuple(sorted(set(x))))
        coupled = sorted(name for name, values in memberships.items() if len(values) > 1)
        original_subset = run.groupby("subset")["original_subset_f"].first() / 100.0
        original_h = hmean(original_subset.tolist()) * 100.0

        # One count record per file; duplicate subset rows must agree.
        count_cols = ["file_tp", "file_fp", "file_fn", "silenced_file_fn"]
        file_counts = run.groupby("target_filename")[count_cols].first()

        best_h = -1.0
        best_removed = None
        best_subset_scores = None
        for coupled_flags in itertools.product((False, True), repeat=len(coupled)):
            coupled_remove = dict(zip(coupled, coupled_flags))
            subset_scores = {}
            removed = {name for name, flag in coupled_remove.items() if flag}

            for subset, subset_run in run.groupby("subset"):
                names = sorted(set(subset_run["target_filename"]))
                independent = [name for name in names if name not in coupled_remove]
                fixed = [0, 0, 0]
                for name in names:
                    if name not in coupled_remove:
                        continue
                    values = file_counts.loc[name]
                    if coupled_remove[name]:
                        fixed[2] += int(values["silenced_file_fn"])
                    else:
                        fixed[0] += int(values["file_tp"])
                        fixed[1] += int(values["file_fp"])
                        fixed[2] += int(values["file_fn"])

                best_subset_f = -1.0
                best_subset_removed = set()
                for flags in itertools.product((False, True), repeat=len(independent)):
                    counts = list(fixed)
                    subset_removed = set()
                    for name, remove in zip(independent, flags):
                        values = file_counts.loc[name]
                        if remove:
                            counts[2] += int(values["silenced_file_fn"])
                            subset_removed.add(name)
                        else:
                            counts[0] += int(values["file_tp"])
                            counts[1] += int(values["file_fp"])
                            counts[2] += int(values["file_fn"])
                    value = fmeasure(counts)
                    if value > best_subset_f:
                        best_subset_f = value
                        best_subset_removed = subset_removed
                subset_scores[subset] = best_subset_f
                removed.update(best_subset_removed)

            aggregate = hmean(subset_scores.values()) * 100.0
            if aggregate > best_h:
                best_h = aggregate
                best_removed = sorted(removed)
                best_subset_scores = subset_scores

        rows.append(
            {
                "configuration": configuration,
                "seed": seed,
                "evaluator_variant": evaluator_variant,
                "original_fmeasure_harmonic": original_h,
                "oracle_fmeasure_harmonic": best_h,
                "oracle_delta_harmonic": best_h - original_h,
                "n_removed_recordings": len(best_removed or []),
                "removed_recordings": ";".join(best_removed or []),
                **{
                    "oracle_subset_%s" % subset: score * 100.0
                    for subset, score in sorted((best_subset_scores or {}).items())
                },
            }
        )
    return add_display_names(pd.DataFrame(rows))


def validate_results(
    scores: pd.DataFrame, silencing: Mapping[str, pd.DataFrame]
) -> Dict[str, int]:
    interventions = silencing["interventions"]
    subset_effects = silencing["subset_effects"]
    if scores.empty or interventions.empty or subset_effects.empty:
        raise ValueError("Scores and counterfactual result tables must be non-empty")
    mismatches = int((~subset_effects["condition_matches_observation"]).sum())
    missing_score_runs = set(
        zip(interventions["configuration"], interventions["seed"])
    ) - set(zip(scores["configuration"], scores["seed"]))
    if missing_score_runs:
        raise ValueError("Counterfactual runs absent from sweep scores: %s" % sorted(missing_score_runs))
    containment_violations = 0
    if {"released", "corrected"}.issubset(
        set(interventions["evaluator_variant"].dropna().unique())
    ):
        comparison = interventions.pivot_table(
            index=["configuration", "seed", "target_filename"],
            columns="evaluator_variant",
            values="improves_official",
            aggfunc="first",
        )
        containment_violations = int(
            ((comparison["released"] == True) & (comparison["corrected"] == False)).sum()
        )
        if containment_violations:
            raise ValueError(
                "Released-positive interventions must be corrected-positive; found %d violations"
                % containment_violations
            )
    return {
        "n_score_runs": len(scores),
        "n_interventions": len(interventions),
        "n_subset_effects": len(subset_effects),
        "condition_mismatches": mismatches,
        "released_corrected_containment_violations": containment_violations,
    }
