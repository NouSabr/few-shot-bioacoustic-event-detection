"""Single-recording silencing counterfactual for selected sweep predictions.

For each collected configuration/seed prediction, remove every predicted event
for one recording while holding the operating point and all other recordings
fixed.  The module reports changes under both the released evaluator and a
corrected aggregation that always excludes the five support events, including
when a recording has no prediction rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep official evaluator imports independent of the shell's working directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MIN_EVAL_VALUE = 0.00001
N_SHOTS = 5
SCORE_TOLERANCE = 0.001
DELTA_TOLERANCE = 1e-12
PREDICTION_HEADER = ["Audiofilename", "Starttime", "Endtime"]
SUPPORTED_VARIANTS = ("released", "corrected")
COMPLETE_ARTIFACT_STATUSES = {"complete_verified", "complete_unverified"}


class CounterfactualError(RuntimeError):
    """Expected analysis failure for one run or input artifact."""


@dataclass(frozen=True)
class ReferenceIndex:
    """Validation reference structure needed for count pooling."""

    subsets: Mapping[str, Tuple[str, ...]]
    memberships: Mapping[str, Tuple[str, ...]]
    full_positive_counts: Mapping[str, int]
    query_positive_counts: Mapping[str, int]

    @property
    def filenames(self) -> Tuple[str, ...]:
        return tuple(sorted(self.memberships))


@dataclass(frozen=True)
class PredictionTable:
    header: Tuple[str, ...]
    rows: Tuple[Mapping[str, str], ...]

    @property
    def filenames_with_predictions(self) -> set:
        return {str(row["Audiofilename"]) for row in self.rows}


@dataclass(frozen=True)
class EvaluationSnapshot:
    """Evaluator output normalized for the analysis."""

    overall: Mapping[str, float]
    individual_counts: Mapping[str, Mapping[str, int]]
    subset_scores: Mapping[str, Mapping[str, float]]


Evaluator = Callable[[Path, Path], EvaluationSnapshot]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_fingerprint(reference_dir: Path) -> str:
    """Hash relative paths and bytes of every reference CSV."""
    digest = hashlib.sha256()
    for path in sorted(reference_dir.rglob("*.csv")):
        digest.update(str(path.relative_to(reference_dir)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _positive_column(fieldnames: Sequence[str], path: Path) -> str:
    matches = [name for name in fieldnames if "Q" in name]
    if not matches:
        raise CounterfactualError(f"reference file has no Q column: {path}")
    return matches[0]


def positive_event_counts(path: Path, n_shots: int = N_SHOTS) -> Tuple[int, int]:
    """Return full and post-support POS counts, mirroring remove_shots_from_ref."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise CounterfactualError(f"reference CSV has no header: {path}")
        column = _positive_column(reader.fieldnames, path)
        rows = list(reader)

    positives = [row for row in rows if row.get(column) == "POS"]
    full_count = len(positives)
    if full_count < n_shots:
        return full_count, 0
    try:
        fifth_end = float(positives[n_shots - 1]["Endtime"])
        query_count = sum(
            1
            for row in positives
            if float(row["Endtime"]) > fifth_end
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CounterfactualError(
            f"reference file has invalid Endtime values: {path}"
        ) from exc
    return full_count, query_count


def count_positive_events(path: Path) -> int:
    """Backwards-compatible full POS count helper."""
    return positive_event_counts(path)[0]


def build_reference_index(reference_dir: Path, n_shots: int = N_SHOTS) -> ReferenceIndex:
    """Read subset membership and positive counts from Validation_Set."""
    if not reference_dir.is_dir():
        raise CounterfactualError(f"reference directory unavailable: {reference_dir}")

    subsets: Dict[str, Tuple[str, ...]] = {}
    memberships: Dict[str, List[str]] = {}
    full_counts: Dict[str, int] = {}
    query_counts: Dict[str, int] = {}

    subset_dirs = sorted(path for path in reference_dir.iterdir() if path.is_dir())
    if not subset_dirs:
        raise CounterfactualError(f"reference directory has no subset folders: {reference_dir}")

    for subset_dir in subset_dirs:
        filenames = []
        for ref_csv in sorted(subset_dir.glob("*.csv")):
            filename = ref_csv.with_suffix(".wav").name
            full_count, query_count = positive_event_counts(ref_csv, n_shots)
            if filename in full_counts and (
                full_counts[filename] != full_count
                or query_counts[filename] != query_count
            ):
                raise CounterfactualError(
                    f"duplicate reference {filename} has inconsistent POS counts: "
                    f"full/query {full_counts[filename]}/{query_counts[filename]} "
                    f"versus {full_count}/{query_count}"
                )
            full_counts[filename] = full_count
            query_counts[filename] = query_count
            memberships.setdefault(filename, []).append(subset_dir.name)
            filenames.append(filename)
        subsets[subset_dir.name] = tuple(filenames)

    if not full_counts:
        raise CounterfactualError(f"no reference CSVs found under {reference_dir}")

    return ReferenceIndex(
        subsets=subsets,
        memberships={key: tuple(value) for key, value in memberships.items()},
        full_positive_counts=full_counts,
        query_positive_counts=query_counts,
    )


def load_prediction_table(path: Path) -> PredictionTable:
    if not path.is_file():
        raise CounterfactualError(f"prediction CSV unavailable: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PREDICTION_HEADER:
            raise CounterfactualError(
                f"unexpected prediction header in {path}: {reader.fieldnames}; "
                f"expected {PREDICTION_HEADER}"
            )
        rows = tuple(dict(row) for row in reader)
    return PredictionTable(tuple(PREDICTION_HEADER), rows)


def prediction_removal_stats(table: PredictionTable, filename: str) -> Tuple[int, float]:
    selected = [row for row in table.rows if row["Audiofilename"] == filename]
    duration = 0.0
    for row in selected:
        try:
            duration += max(float(row["Endtime"]) - float(row["Starttime"]), 0.0)
        except (TypeError, ValueError) as exc:
            raise CounterfactualError(
                f"non-numeric onset/offset for {filename}: {row}"
            ) from exc
    return len(selected), duration


def write_silenced_prediction(
    table: PredictionTable, target_filename: str, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table.header))
        writer.writeheader()
        writer.writerows(
            row for row in table.rows if row["Audiofilename"] != target_filename
        )


def _normalise_snapshot(raw: Any) -> EvaluationSnapshot:
    if isinstance(raw, EvaluationSnapshot):
        return raw
    if not isinstance(raw, tuple) or len(raw) != 4:
        raise CounterfactualError("evaluator must return its standard four-value tuple")
    overall, individual, subset_scores, _ = raw
    return EvaluationSnapshot(overall, individual, subset_scores)


def run_released_evaluator(
    prediction_csv: Path,
    reference_dir: Path,
    evaluator=None,
    quiet: bool = True,
) -> EvaluationSnapshot:
    """Run the repository evaluator without leaving report files behind."""
    injected_evaluator = evaluator is not None
    if evaluator is None:
        try:
            from src.utils.evaluation import evaluate as evaluator
        except ImportError as exc:
            raise CounterfactualError(
                f"released evaluator dependencies are unavailable: {exc}"
            ) from exc

    with tempfile.TemporaryDirectory(prefix="silencing-eval-") as output_dir:
        def invoke():
            if injected_evaluator:
                return evaluator(Path(prediction_csv), Path(reference_dir))
            return evaluator(
                str(prediction_csv),
                str(reference_dir),
                "silencing_counterfactual",
                "VAL",
                output_dir,
            )

        if quiet:
            with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
                raw = invoke()
        else:
            raw = invoke()
    return _normalise_snapshot(raw)


def score_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp != 0 else MIN_EVAL_VALUE
    precision = max(precision, MIN_EVAL_VALUE)
    recall = tp / (tp + fn) if tp != 0 else MIN_EVAL_VALUE
    fmeasure = tp / (tp + 0.5 * (fp + fn)) if tp != 0 else MIN_EVAL_VALUE
    return {"precision": precision, "recall": recall, "f-measure": fmeasure}


def harmonic_mean(values: Iterable[float]) -> float:
    values = list(values)
    return len(values) / sum(1.0 / max(float(value), MIN_EVAL_VALUE) for value in values)


def complete_counts(
    individual_counts: Mapping[str, Mapping[str, int]],
    reference: ReferenceIndex,
    corrected_support_handling: bool,
) -> Dict[str, Dict[str, int]]:
    """Backfill files absent from the prediction CSV under one evaluator policy."""
    counts = {filename: dict(value) for filename, value in individual_counts.items()}
    unknown = sorted(set(counts) - set(reference.memberships))
    if unknown:
        raise CounterfactualError(
            f"evaluator returned filenames absent from the reference tree: {unknown}"
        )
    for filename in reference.filenames:
        if filename in counts:
            continue
        positives = (
            reference.query_positive_counts[filename]
            if corrected_support_handling
            else reference.full_positive_counts[filename]
        )
        counts[filename] = {
            "TP": 0,
            "FP": 0,
            "FN": positives,
            "total_n_pos_events": positives,
        }
    return counts


def aggregate_counts(
    counts_by_file: Mapping[str, Mapping[str, int]], reference: ReferenceIndex
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]], Dict[str, float]]:
    subset_counts: Dict[str, Dict[str, int]] = {}
    subset_scores: Dict[str, Dict[str, float]] = {}
    for subset, filenames in reference.subsets.items():
        counts = {
            "TP": sum(int(counts_by_file[name]["TP"]) for name in filenames),
            "FP": sum(int(counts_by_file[name]["FP"]) for name in filenames),
            "FN": sum(int(counts_by_file[name]["FN"]) for name in filenames),
        }
        subset_counts[subset] = counts
        subset_scores[subset] = score_counts(**{key.lower(): value for key, value in counts.items()})

    precision_values = [score["precision"] for score in subset_scores.values()]
    recall_values = [score["recall"] for score in subset_scores.values()]
    f_values = [score["f-measure"] for score in subset_scores.values()]
    overall = {
        "precision": harmonic_mean(precision_values),
        "recall": harmonic_mean(recall_values),
        "fmeasure": round(harmonic_mean(f_values) * 100, 3),
        "precision-avg": sum(precision_values) / len(precision_values),
        "recall-avg": sum(recall_values) / len(recall_values),
        "fmeasure-avg": round(sum(f_values) / len(f_values) * 100, 3),
    }
    return subset_counts, subset_scores, overall


def corrected_snapshot(
    released: EvaluationSnapshot, reference: ReferenceIndex
) -> Tuple[EvaluationSnapshot, Dict[str, Dict[str, int]]]:
    counts = complete_counts(
        released.individual_counts, reference, corrected_support_handling=True
    )
    _, subset_scores, overall = aggregate_counts(counts, reference)
    return EvaluationSnapshot(overall, counts, subset_scores), counts


def released_complete_counts(
    released: EvaluationSnapshot, reference: ReferenceIndex
) -> Dict[str, Dict[str, int]]:
    return complete_counts(
        released.individual_counts, reference, corrected_support_handling=False
    )


def score_value(scores: Mapping[str, float], key: str) -> float:
    aliases = {
        "fmeasure_arithmetic": ("fmeasure-avg", "fmeasureavg"),
        "precision_arithmetic": ("precision-avg", "precisionavg"),
        "recall_arithmetic": ("recall-avg", "recallavg"),
    }
    if key == "fmeasure_harmonic":
        return float(scores["fmeasure"])
    if key == "precision_harmonic":
        return float(scores["precision"]) * 100
    if key == "recall_harmonic":
        return float(scores["recall"]) * 100
    for alias in aliases[key]:
        if alias in scores:
            value = float(scores[alias])
            return value if key == "fmeasure_arithmetic" else value * 100
    raise KeyError(f"score {key} is absent from {scores}")


def subset_f_percent(snapshot: EvaluationSnapshot, subset: str) -> float:
    score = snapshot.subset_scores[subset]
    value = score.get("f-measure", score.get("fmeasure"))
    return float(value) * 100


def exact_fmeasure_aggregate(snapshot: EvaluationSnapshot, harmonic: bool) -> float:
    """Aggregate unrounded subset F-measures on the evaluator's 0--100 scale."""
    values = [
        float(score.get("f-measure", score.get("fmeasure")))
        for score in snapshot.subset_scores.values()
    ]
    aggregate = harmonic_mean(values) if harmonic else sum(values) / len(values)
    return aggregate * 100.0


def analytic_silence_condition(
    other_tp: int,
    other_fp: int,
    other_fn: int,
    file_tp: int,
    file_fp: int,
    file_fn: int,
    silenced_fn: int,
) -> Dict[str, Any]:
    """Evaluate the exact pooled-F condition for replacing one file by silence."""
    support_penalty = silenced_fn - (file_tp + file_fn)
    lhs = other_tp * (file_fp - file_tp)
    rhs = (
        file_tp * (other_fp + other_fn + file_tp + file_fn)
        + support_penalty * (other_tp + file_tp)
    )
    return {
        "support_penalty": support_penalty,
        "condition_lhs": lhs,
        "condition_rhs": rhs,
        "condition_margin": lhs - rhs,
        "condition_holds": lhs > rhs,
    }


def _overall_delta_fields(
    original: EvaluationSnapshot, silenced: EvaluationSnapshot
) -> Dict[str, float]:
    fields = {}
    for name in (
        "fmeasure_harmonic",
        "fmeasure_arithmetic",
        "precision_harmonic",
        "precision_arithmetic",
        "recall_harmonic",
        "recall_arithmetic",
    ):
        if name == "fmeasure_harmonic":
            original_value = exact_fmeasure_aggregate(original, harmonic=True)
            silenced_value = exact_fmeasure_aggregate(silenced, harmonic=True)
        elif name == "fmeasure_arithmetic":
            original_value = exact_fmeasure_aggregate(original, harmonic=False)
            silenced_value = exact_fmeasure_aggregate(silenced, harmonic=False)
        else:
            original_value = score_value(original.overall, name)
            silenced_value = score_value(silenced.overall, name)
        fields[f"original_{name}"] = original_value
        fields[f"silenced_{name}"] = silenced_value
        fields[f"delta_{name}"] = silenced_value - original_value
    return fields


def _variant_snapshot_and_counts(
    variant: str,
    released: EvaluationSnapshot,
    reference: ReferenceIndex,
) -> Tuple[EvaluationSnapshot, Dict[str, Dict[str, int]]]:
    if variant == "released":
        return released, released_complete_counts(released, reference)
    if variant == "corrected":
        return corrected_snapshot(released, reference)
    raise ValueError(f"unsupported evaluator variant: {variant}")


def analyse_run(
    run_record: Mapping[str, Any],
    reference: ReferenceIndex,
    reference_dir: Path,
    variants: Sequence[str] = SUPPORTED_VARIANTS,
    evaluator: Optional[Evaluator] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run every one-file intervention for one configuration/seed artifact."""
    configuration = run_record.get("configuration")
    seed = run_record.get("seed")
    archive = run_record.get("archive") or {}
    source = run_record.get("source") or {}
    prediction_value = archive.get("final_csv") or source.get("final_csv")
    if not prediction_value:
        raise CounterfactualError("manifest run has no selected final CSV")
    prediction_csv = Path(prediction_value)
    table = load_prediction_table(prediction_csv)

    unknown_predictions = sorted(table.filenames_with_predictions - set(reference.memberships))
    if unknown_predictions:
        raise CounterfactualError(
            f"prediction CSV contains filenames absent from references: {unknown_predictions}"
        )

    released_original = run_released_evaluator(
        prediction_csv, reference_dir, evaluator=evaluator
    )
    expected = run_record.get("logged_scores") or {}
    expected_h = expected.get("fmeasure_harmonic")
    expected_a = expected.get("fmeasure_arithmetic")
    if expected_h is not None and not math.isclose(
        score_value(released_original.overall, "fmeasure_harmonic"),
        float(expected_h),
        abs_tol=SCORE_TOLERANCE,
    ):
        raise CounterfactualError(
            "selected final CSV does not reproduce logged harmonic F: "
            f"{released_original.overall.get('fmeasure')} versus {expected_h}"
        )
    if expected_a is not None and not math.isclose(
        score_value(released_original.overall, "fmeasure_arithmetic"),
        float(expected_a),
        abs_tol=SCORE_TOLERANCE,
    ):
        raise CounterfactualError(
            "selected final CSV does not reproduce logged arithmetic F: "
            f"{released_original.overall.get('fmeasure-avg')} versus {expected_a}"
        )

    original_variants = {
        variant: _variant_snapshot_and_counts(variant, released_original, reference)
        for variant in variants
    }
    intervention_rows: List[Dict[str, Any]] = []
    subset_rows: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="silencing-interventions-") as temp_dir:
        temp_root = Path(temp_dir)
        for file_index, filename in enumerate(reference.filenames):
            prediction_count, prediction_duration = prediction_removal_stats(table, filename)
            originally_silent = prediction_count == 0
            modified_csv = temp_root / f"silenced_{file_index:04d}.csv"
            write_silenced_prediction(table, filename, modified_csv)
            released_silenced = run_released_evaluator(
                modified_csv, reference_dir, evaluator=evaluator
            )

            for variant in variants:
                original_snapshot, original_counts = original_variants[variant]
                silenced_snapshot, silenced_counts = _variant_snapshot_and_counts(
                    variant, released_silenced, reference
                )
                file_counts = original_counts[filename]
                file_tp = int(file_counts["TP"])
                file_fp = int(file_counts["FP"])
                file_fn = int(file_counts["FN"])
                target_predictions = file_tp + file_fp
                affected_subsets = reference.memberships[filename]
                overall_fields = _overall_delta_fields(
                    original_snapshot, silenced_snapshot
                )
                per_subset_deltas = [
                    subset_f_percent(silenced_snapshot, subset)
                    - subset_f_percent(original_snapshot, subset)
                    for subset in affected_subsets
                ]
                intervention_rows.append(
                    {
                        "configuration": configuration,
                        "seed": seed,
                        "evaluator_variant": variant,
                        "target_filename": filename,
                        "affected_subsets": ";".join(affected_subsets),
                        "originally_silent": originally_silent,
                        "target_tp": file_tp,
                        "target_fp": file_fp,
                        "target_fn": file_fn,
                        "forfeits_true_positives": file_tp > 0,
                        "target_prediction_precision": (
                            file_tp / target_predictions if target_predictions else 0.0
                        ),
                        "predicted_events_removed": prediction_count,
                        "predicted_duration_removed_seconds": prediction_duration,
                        "max_delta_affected_subset_f": max(per_subset_deltas),
                        "min_delta_affected_subset_f": min(per_subset_deltas),
                        "improves_any_subset": any(
                            delta > DELTA_TOLERANCE for delta in per_subset_deltas
                        ),
                        "improves_all_affected_subsets": all(
                            delta > DELTA_TOLERANCE for delta in per_subset_deltas
                        ),
                        "improves_official": (
                            overall_fields["delta_fmeasure_harmonic"] > DELTA_TOLERANCE
                        ),
                        **overall_fields,
                    }
                )

                for subset in affected_subsets:
                    subset_filenames = reference.subsets[subset]
                    original_subset_counts = {
                        key: sum(int(original_counts[name][key]) for name in subset_filenames)
                        for key in ("TP", "FP", "FN")
                    }
                    other_counts = {
                        key: original_subset_counts[key] - int(file_counts[key])
                        for key in ("TP", "FP", "FN")
                    }
                    silence_counts = silenced_counts[filename]
                    condition = analytic_silence_condition(
                        other_tp=other_counts["TP"],
                        other_fp=other_counts["FP"],
                        other_fn=other_counts["FN"],
                        file_tp=int(file_counts["TP"]),
                        file_fp=int(file_counts["FP"]),
                        file_fn=int(file_counts["FN"]),
                        silenced_fn=int(silence_counts["FN"]),
                    )
                    original_subset_f = subset_f_percent(original_snapshot, subset)
                    silenced_subset_f = subset_f_percent(silenced_snapshot, subset)
                    observed_improvement = (
                        silenced_subset_f - original_subset_f > DELTA_TOLERANCE
                    )
                    subset_rows.append(
                        {
                            "configuration": configuration,
                            "seed": seed,
                            "evaluator_variant": variant,
                            "target_filename": filename,
                            "subset": subset,
                            "originally_silent": originally_silent,
                            "file_tp": int(file_counts["TP"]),
                            "file_fp": int(file_counts["FP"]),
                            "file_fn": int(file_counts["FN"]),
                            "silenced_file_fn": int(silence_counts["FN"]),
                            "other_tp": other_counts["TP"],
                            "other_fp": other_counts["FP"],
                            "other_fn": other_counts["FN"],
                            "original_subset_f": original_subset_f,
                            "silenced_subset_f": silenced_subset_f,
                            "delta_subset_f": silenced_subset_f - original_subset_f,
                            "observed_subset_improvement": observed_improvement,
                            "condition_matches_observation": (
                                condition["condition_holds"] == observed_improvement
                            ),
                            **condition,
                        }
                    )

    run_summary = {
        "configuration": configuration,
        "seed": seed,
        "artifact_status": run_record.get("artifact_status"),
        "prediction_csv": str(prediction_csv),
        "prediction_sha256": sha256_file(prediction_csv),
        "n_reference_files": len(reference.filenames),
        "n_prediction_rows": len(table.rows),
        "variants": list(variants),
    }
    return intervention_rows, subset_rows, run_summary


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def select_manifest_runs(
    manifest: Mapping[str, Any],
    configurations: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Tuple[List[Mapping[str, Any]], List[Dict[str, Any]]]:
    selected = []
    skipped = []
    configuration_filter = set(configurations or [])
    seed_filter = set(seeds or [])
    for run in manifest.get("runs", []):
        configuration = run.get("configuration")
        seed = run.get("seed")
        if configuration_filter and configuration not in configuration_filter:
            continue
        if seed_filter and seed not in seed_filter:
            continue
        if run.get("artifact_status") not in COMPLETE_ARTIFACT_STATUSES:
            skipped.append(
                {
                    "configuration": configuration,
                    "seed": seed,
                    "artifact_status": run.get("artifact_status"),
                    "reason": "run has no complete selected prediction artifact",
                }
            )
            continue
        selected.append(run)
    return selected, skipped


def summarise_interventions(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, Any, Any], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["configuration"], row["seed"], row["evaluator_variant"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (configuration, seed, variant), group in sorted(groups.items()):
        eligible = [row for row in group if not row["originally_silent"]]
        improving = [row for row in eligible if row["improves_official"]]
        best = max(
            eligible,
            key=lambda row: row["delta_fmeasure_harmonic"],
            default=None,
        )
        summary.append(
            {
                "configuration": configuration,
                "seed": seed,
                "evaluator_variant": variant,
                "n_reference_files": len(group),
                "n_originally_silent": len(group) - len(eligible),
                "n_eligible_interventions": len(eligible),
                "n_subset_improvements": sum(
                    bool(row["improves_any_subset"]) for row in eligible
                ),
                "n_official_improvements": len(improving),
                "fraction_official_improvements": (
                    len(improving) / len(eligible) if eligible else 0.0
                ),
                "best_target_filename": best["target_filename"] if best else None,
                "max_delta_fmeasure_harmonic": (
                    best["delta_fmeasure_harmonic"] if best else None
                ),
            }
        )
    return summary


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames: List[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in resolved_fieldnames:
                resolved_fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_state(project_root: Path) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project_root, text=True
            ).strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    expected = (
        "silencing_interventions.csv",
        "silencing_subset_effects.csv",
        "silencing_summary.csv",
        "skipped_runs.csv",
        "analysis_manifest.json",
    )
    existing = [output_dir / name for name in expected if (output_dir / name).exists()]
    if existing and not overwrite:
        raise CounterfactualError(
            "analysis output already exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def _portable_final_csv(run: Mapping[str, Any], manifest_path: Path) -> Optional[Path]:
    archive = run.get("archive") or {}
    source = run.get("source") or {}
    registered = archive.get("final_csv") or source.get("final_csv")
    if registered:
        resolved = _resolve_manifest_path(registered, manifest_path)
        if resolved.is_file():
            return resolved
    configuration = run.get("configuration")
    seed = run.get("seed")
    if configuration is None or seed is None:
        return None
    portable = (
        manifest_path.parent
        / f"seed_{seed}"
        / str(configuration)
        / "selected_final.csv"
    )
    return portable.resolve() if portable.is_file() else None


def run_manifest_analysis(
    manifest_path: Path,
    reference_dir: Path,
    output_dir: Path,
    variants: Sequence[str] = SUPPORTED_VARIANTS,
    configurations: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    evaluator: Optional[Evaluator] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    invalid_variants = sorted(set(variants) - set(SUPPORTED_VARIANTS))
    if invalid_variants:
        raise CounterfactualError(f"unsupported evaluator variants: {invalid_variants}")
    prepare_output_directory(output_dir, overwrite)

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    reference = build_reference_index(reference_dir)
    runs, skipped = select_manifest_runs(manifest, configurations, seeds)
    if not runs and not skipped:
        raise CounterfactualError(
            "no collector-manifest runs matched the requested configuration/seed filters"
        )

    intervention_rows: List[Dict[str, Any]] = []
    subset_rows: List[Dict[str, Any]] = []
    completed_runs = []
    errors = []
    for run in runs:
        # Prefer the registered artifact path, but recover the canonical path
        # relative to a moved collector archive when absolute source paths stale.
        run_copy = dict(run)
        archive = dict(run_copy.get("archive") or {})
        source = dict(run_copy.get("source") or {})
        portable_final = _portable_final_csv(run_copy, manifest_path)
        if portable_final is not None:
            archive["final_csv"] = str(portable_final)
        run_copy["archive"] = archive
        run_copy["source"] = source
        try:
            global_rows, per_subset_rows, run_summary = analyse_run(
                run_copy,
                reference,
                reference_dir,
                variants=variants,
                evaluator=evaluator,
            )
            intervention_rows.extend(global_rows)
            subset_rows.extend(per_subset_rows)
            completed_runs.append(run_summary)
        except Exception as exc:
            errors.append(
                {
                    "configuration": run.get("configuration"),
                    "seed": run.get("seed"),
                    "artifact_status": run.get("artifact_status"),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    skipped.extend(errors)
    summary_rows = summarise_interventions(intervention_rows)
    write_csv(output_dir / "silencing_interventions.csv", intervention_rows)
    write_csv(output_dir / "silencing_subset_effects.csv", subset_rows)
    write_csv(output_dir / "silencing_summary.csv", summary_rows)
    write_csv(
        output_dir / "skipped_runs.csv",
        skipped,
        fieldnames=["configuration", "seed", "artifact_status", "reason"],
    )

    project_root = Path(__file__).resolve().parents[1]
    analysis_manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "collector_manifest": str(manifest_path.resolve()),
        "collector_manifest_sha256": sha256_file(manifest_path),
        "reference_dir": str(reference_dir.resolve()),
        "reference_fingerprint_sha256": reference_fingerprint(reference_dir),
        "evaluator_variants": list(variants),
        "n_completed_runs": len(completed_runs),
        "n_skipped_or_failed_runs": len(skipped),
        "n_interventions": len(intervention_rows),
        "n_subset_effects": len(subset_rows),
        "completed_runs": completed_runs,
        "skipped_or_failed_runs": skipped,
        "provenance": _git_state(project_root),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return analysis_manifest
