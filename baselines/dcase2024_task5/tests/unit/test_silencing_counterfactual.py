"""Tests for the single-recording silencing counterfactual."""

import csv
import json
from pathlib import Path

import pytest

from tools.silencing_counterfactual import (
    CounterfactualError,
    EvaluationSnapshot,
    aggregate_counts,
    analytic_silence_condition,
    build_reference_index,
    complete_counts,
    exact_fmeasure_aggregate,
    load_prediction_table,
    run_manifest_analysis,
    write_silenced_prediction,
)


def _write_reference(path: Path, n_pos=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Starttime", "Endtime", "Q"])
        for index in range(n_pos):
            writer.writerow([float(index), float(index) + 0.5, "POS"])


def _write_predictions(path: Path, filenames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Audiofilename", "Starttime", "Endtime"])
        for index, filename in enumerate(filenames):
            writer.writerow([filename, float(index), float(index) + 0.25])


def _reference_tree(root: Path):
    _write_reference(root / "S1" / "A.csv")
    _write_reference(root / "S1" / "B.csv")
    _write_reference(root / "S2" / "C.csv")
    _write_reference(root / "S2" / "D.csv")
    return build_reference_index(root)


def _fake_evaluator_factory(reference):
    base_counts = {
        "A.wav": {"TP": 1, "FP": 20, "FN": 0, "total_n_pos_events": 1},
        "B.wav": {"TP": 1, "FP": 0, "FN": 0, "total_n_pos_events": 1},
        "C.wav": {"TP": 1, "FP": 0, "FN": 0, "total_n_pos_events": 1},
    }

    def evaluate(prediction_csv, reference_dir):
        table = load_prediction_table(Path(prediction_csv))
        present = table.filenames_with_predictions
        individual = {
            filename: dict(counts)
            for filename, counts in base_counts.items()
            if filename in present
        }
        full = complete_counts(
            individual, reference, corrected_support_handling=False
        )
        _, subset_scores, overall = aggregate_counts(full, reference)
        return EvaluationSnapshot(overall, individual, subset_scores)

    return evaluate


def test_build_reference_index_counts_support_and_membership(tmp_path):
    reference = _reference_tree(tmp_path / "Validation_Set")
    assert reference.full_positive_counts["A.wav"] == 6
    assert reference.query_positive_counts["A.wav"] == 1
    assert reference.memberships["A.wav"] == ("S1",)
    assert reference.subsets["S1"] == ("A.wav", "B.wav")


def test_build_reference_index_rejects_inconsistent_duplicate(tmp_path):
    root = tmp_path / "Validation_Set"
    _write_reference(root / "S1" / "A.csv", n_pos=6)
    _write_reference(root / "S2" / "A.csv", n_pos=7)
    with pytest.raises(CounterfactualError, match="inconsistent POS counts"):
        build_reference_index(root)


def test_write_silenced_prediction_removes_only_target(tmp_path):
    source = tmp_path / "source.csv"
    destination = tmp_path / "silenced.csv"
    _write_predictions(source, ["A.wav", "A.wav", "B.wav"])
    table = load_prediction_table(source)
    write_silenced_prediction(table, "A.wav", destination)
    result = load_prediction_table(destination)
    assert [row["Audiofilename"] for row in result.rows] == ["B.wav"]
    assert len(table.rows) == 3


def test_analytic_condition_matches_known_gain():
    result = analytic_silence_condition(
        other_tp=10,
        other_fp=0,
        other_fn=0,
        file_tp=1,
        file_fp=20,
        file_fn=0,
        silenced_fn=1,
    )
    assert result["support_penalty"] == 0
    assert result["condition_holds"] is True
    assert result["condition_margin"] > 0


def test_official_support_penalty_makes_condition_stricter():
    corrected = analytic_silence_condition(10, 0, 0, 1, 20, 0, 1)
    released = analytic_silence_condition(10, 0, 0, 1, 20, 0, 6)
    assert released["support_penalty"] == 5
    assert released["condition_margin"] < corrected["condition_margin"]



def test_analytic_condition_matches_direct_fmeasure_comparison():
    def fmeasure(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    for other_tp in range(1, 5):
        for other_errors in range(5):
            for file_tp in range(4):
                for file_fp in range(7):
                    for file_fn in range(3):
                        for support_penalty in (0, 5):
                            silenced_fn = file_tp + file_fn + support_penalty
                            condition = analytic_silence_condition(
                                other_tp,
                                other_errors,
                                0,
                                file_tp,
                                file_fp,
                                file_fn,
                                silenced_fn,
                            )
                            original = fmeasure(
                                other_tp + file_tp,
                                file_fp,
                                other_errors + file_fn,
                            )
                            silenced = fmeasure(
                                other_tp,
                                0,
                                other_errors + silenced_fn,
                            )
                            assert condition["condition_holds"] == (silenced > original)
                            if support_penalty == 0 and file_tp + file_fp > 0:
                                target_precision = file_tp / (file_tp + file_fp)
                                assert condition["condition_holds"] == (
                                    target_precision < original / 2
                                )
                            if support_penalty == 5:
                                official_rule = (
                                    file_tp + file_fp > 5
                                    and 2 * file_tp / (file_tp + file_fp - 5) < original
                                )
                                assert condition["condition_holds"] == official_rule

def test_exact_fmeasure_aggregate_ignores_rounded_overall_value():
    subset_scores = {
        "A": {"f-measure": 0.100004},
        "B": {"f-measure": 0.2},
    }
    snapshot = EvaluationSnapshot(
        {"fmeasure": 13.334}, {}, subset_scores
    )
    exact = exact_fmeasure_aggregate(snapshot, harmonic=True)
    assert exact != snapshot.overall["fmeasure"]
    assert exact > 13.333


def test_run_manifest_analysis_writes_single_file_results(tmp_path):
    reference_dir = tmp_path / "Validation_Set"
    reference = _reference_tree(reference_dir)
    prediction = tmp_path / "selected_final.csv"
    _write_predictions(prediction, ["A.wav", "B.wav", "C.wav"])
    evaluator = _fake_evaluator_factory(reference)
    original = evaluator(prediction, reference_dir)

    collector_manifest = {
        "sweep_id": "test-sweep",
        "runs": [
            {
                "configuration": "baseline",
                "seed": 1234,
                "artifact_status": "complete_verified",
                "archive": {"final_csv": str(prediction)},
                "source": {},
                "logged_scores": {
                    "fmeasure_harmonic": original.overall["fmeasure"],
                    "fmeasure_arithmetic": original.overall["fmeasure-avg"],
                },
            },
            {
                "configuration": "phase3_activation",
                "seed": 1234,
                "artifact_status": "log_only",
                "archive": {},
                "source": {},
            },
        ],
    }
    manifest_path = tmp_path / "collector_manifest.json"
    manifest_path.write_text(json.dumps(collector_manifest), encoding="utf-8")
    output = tmp_path / "analysis"

    result = run_manifest_analysis(
        manifest_path,
        reference_dir,
        output,
        evaluator=evaluator,
    )

    assert result["n_completed_runs"] == 1
    assert result["n_skipped_or_failed_runs"] == 1
    assert result["n_interventions"] == 8  # 4 files x 2 evaluator variants
    assert (output / "silencing_interventions.csv").is_file()
    assert (output / "silencing_subset_effects.csv").is_file()
    assert (output / "silencing_summary.csv").is_file()
    assert (output / "skipped_runs.csv").is_file()

    with (output / "silencing_interventions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    a_rows = [row for row in rows if row["target_filename"] == "A.wav"]
    assert len(a_rows) == 2
    assert all(row["improves_official"] == "True" for row in a_rows)
    assert all(int(row["target_tp"]) == 1 for row in a_rows)
    assert all(int(row["target_fp"]) == 20 for row in a_rows)
    assert all(row["forfeits_true_positives"] == "True" for row in a_rows)
    d_rows = [row for row in rows if row["target_filename"] == "D.wav"]
    assert len(d_rows) == 2
    assert all(row["originally_silent"] == "True" for row in d_rows)
    assert all(float(row["delta_fmeasure_harmonic"]) == 0.0 for row in d_rows)

    with (output / "silencing_subset_effects.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        subset_rows = list(csv.DictReader(handle))
    assert all(row["condition_matches_observation"] == "True" for row in subset_rows)


def test_manifest_original_score_mismatch_skips_run(tmp_path):
    reference_dir = tmp_path / "Validation_Set"
    reference = _reference_tree(reference_dir)
    prediction = tmp_path / "selected_final.csv"
    _write_predictions(prediction, ["A.wav", "B.wav", "C.wav"])
    evaluator = _fake_evaluator_factory(reference)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "configuration": "baseline",
                        "seed": 1234,
                        "artifact_status": "complete_verified",
                        "archive": {"final_csv": str(prediction)},
                        "logged_scores": {
                            "fmeasure_harmonic": 99.0,
                            "fmeasure_arithmetic": 99.0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = run_manifest_analysis(
        manifest_path,
        reference_dir,
        tmp_path / "output",
        evaluator=evaluator,
    )
    assert result["n_completed_runs"] == 0
    assert result["n_skipped_or_failed_runs"] == 1
    assert "does not reproduce logged harmonic" in result["skipped_or_failed_runs"][0]["reason"]
