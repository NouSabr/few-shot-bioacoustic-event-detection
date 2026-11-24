"""Tests for tools/sweep_collector.py."""

import csv
import json
from pathlib import Path

import pytest
import yaml

from tools.sweep_collector import (
    STATUS_COMPLETE_UNVERIFIED,
    STATUS_LOG_ONLY,
    STATUS_MISSING_RUN,
    Winner,
    collect_registry,
    locate_artifacts,
    parse_log,
    select_winner,
    verify_with_evaluator,
)


def _scores(fmeasure, arithmetic):
    return {
        "precision": 0.1,
        "recall": 0.2,
        "fmeasure": fmeasure,
        "precision-avg": 0.3,
        "recall-avg": 0.4,
        "fmeasure-avg": arithmetic,
    }


def _write_log(path: Path, best_a=11.0, best_b=12.0):
    score_a = _scores(best_a, 21.0)
    score_b = _scores(best_b, 22.0)
    final = score_b if best_b > best_a else score_a
    path.write_text(
        "\n".join(
            [
                "Before preprocessing:",
                f"Overall_scores: {_scores(10.0, 20.0)}",
                "Threshold 0.2",
                f"Overall_scores: {score_a}",
                "******************BEST RESULT*****************",
                "file.wav {'TP': 1, 'FP': 2, 'FN': 3, 'total_n_pos_events': 4}",
                f"{score_a} 0.2",
                "Threshold length 0.05",
                f"Overall_scores: {score_b}",
                "******************BEST RESULT*****************",
                "file.wav {'TP': 2, 'FP': 1, 'FN': 2, 'total_n_pos_events': 4}",
                f"{score_b} 0.05",
                "Best Result:",
                str(final),
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_prediction(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Audiofilename", "Starttime", "Endtime"])
        writer.writerow(["file.wav", "1.0", "2.0"])


def test_select_winner_uses_global_best_and_preserves_mode():
    parsed = {
        "final": {"fmeasure": 12.0},
        "best_A": [
            {"k": 0.0, "winning_thresh": 0.2, "fmeasure": 11.0, "per_file": {}}
        ],
        "best_B": [
            {"k": 0.0, "winning_thresh": 0.05, "fmeasure": 12.0, "per_file": {}}
        ],
    }
    winner = select_winner(parsed)
    assert winner.mode == "B"
    assert winner.k == 0.0
    assert winner.postprocessing_threshold == 0.05


def test_progress_prefixed_before_marker_keeps_k_alignment(tmp_path):
    log_path = tmp_path / "stdout.log"
    first_a = _scores(10.0, 20.0)
    first_b = _scores(11.0, 21.0)
    second_a = _scores(12.0, 22.0)
    second_b = _scores(13.0, 23.0)
    log_path.write_text(
        "\n".join(
            [
                "Before preprocessing:",
                f"Overall_scores: {_scores(9.0, 19.0)}",
                "Threshold 0.2",
                f"Overall_scores: {first_a}",
                "******************BEST RESULT*****************",
                f"{first_a} 0.2",
                "Threshold length 0.05",
                f"Overall_scores: {first_b}",
                "******************BEST RESULT*****************",
                f"{first_b} 0.05",
                "Testing DataLoader 0: 100%|progress|Before preprocessing:",
                f"Overall_scores: {_scores(11.5, 21.5)}",
                "Threshold 0.2",
                f"Overall_scores: {second_a}",
                "******************BEST RESULT*****************",
                f"{second_a} 0.2",
                "Threshold length 0.05",
                f"Overall_scores: {second_b}",
                "******************BEST RESULT*****************",
                f"{second_b} 0.05",
                "Best Result:",
                str(second_b),
                "",
            ]
        ),
        encoding="utf-8",
    )
    parsed = parse_log(log_path)
    winner = select_winner(parsed)
    assert winner.k == 0.05
    assert winner.mode == "B"


def test_select_winner_matches_training_tie_order():
    parsed = {
        "final": {"fmeasure": 12.0},
        "best_A": [
            {"k": 0.0, "winning_thresh": 0.2, "fmeasure": 12.0, "per_file": {}}
        ],
        "best_B": [
            {"k": 0.0, "winning_thresh": 0.05, "fmeasure": 12.0, "per_file": {}}
        ],
    }
    assert select_winner(parsed).mode == "A"


def test_locate_artifacts_matches_float_names(tmp_path):
    threshold_dir = tmp_path / "0.9500000000000001"
    _write_prediction(threshold_dir / "Eval_raw.csv")
    expected = threshold_dir / (
        "Eval_VAL_threshold_fix_length_postproc_0.15000000000000002.csv"
    )
    _write_prediction(expected)
    winner = Winner(
        mode="B",
        k=0.95,
        postprocessing_threshold=0.15,
        scores={"fmeasure": 1.0},
        per_file={},
    )
    located = locate_artifacts(tmp_path, winner)
    assert located.threshold_dir == threshold_dir
    assert located.final_csv == expected


def test_collect_registry_archives_selected_files_and_manifest(tmp_path):
    log_path = tmp_path / "stdout.log"
    run_dir = tmp_path / "run"
    _write_log(log_path)
    _write_prediction(run_dir / "0.0" / "Eval_raw.csv")
    _write_prediction(
        run_dir / "0.0" / "Eval_VAL_threshold_fix_length_postproc_0.05.csv"
    )
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "sweep_id": "test-sweep",
                "expected_seeds": [1234],
                "expected_configurations": ["baseline"],
                "runs": [
                    {
                        "configuration": "baseline",
                        "seed": 1234,
                        "log": str(log_path),
                        "run_dir": str(run_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "archive"
    manifest = collect_registry(registry_path, output, copy_files=True)
    record = manifest["runs"][0]
    assert record["artifact_status"] == STATUS_COMPLETE_UNVERIFIED
    assert (output / "seed_1234" / "baseline" / "stdout.log").is_file()
    assert (output / "seed_1234" / "baseline" / "selected_raw.csv").is_file()
    assert (output / "seed_1234" / "baseline" / "selected_final.csv").is_file()
    assert (output / "seed_1234" / "baseline" / "selection.json").is_file()
    written = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert written["summary"] == {STATUS_COMPLETE_UNVERIFIED: 1}


def test_log_only_run_is_recorded_without_crashing(tmp_path):
    log_path = tmp_path / "stdout.log"
    _write_log(log_path)
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "sweep_id": "test-sweep",
                "runs": [
                    {
                        "configuration": "phase3_activation",
                        "seed": 1234,
                        "log": str(log_path),
                        "run_dir": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "archive"
    manifest = collect_registry(registry_path, output, copy_files=True)
    record = manifest["runs"][0]
    assert record["artifact_status"] == STATUS_LOG_ONLY
    assert record["selection"]["postprocessing_type"] == "fixed_length"
    assert record["winning_per_file_counts"]["file.wav"]["TP"] == 2
    assert (output / "seed_1234" / "phase3_activation" / "stdout.log").is_file()
    assert (output / "seed_1234" / "phase3_activation" / "selection.json").is_file()


def test_expected_but_unregistered_run_is_reported(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "sweep_id": "test-sweep",
                "expected_seeds": [1234],
                "expected_configurations": ["baseline", "cife"],
                "runs": [
                    {
                        "configuration": "baseline",
                        "seed": 1234,
                        "log": str(tmp_path / "missing.log"),
                        "run_dir": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = collect_registry(registry_path, tmp_path / "archive")
    missing = [r for r in manifest["runs"] if r["artifact_status"] == STATUS_MISSING_RUN]
    assert [(r["configuration"], r["seed"]) for r in missing] == [("cife", 1234)]


def test_duplicate_registry_entries_fail_loudly(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    run = {"configuration": "baseline", "seed": 1234, "log": "x", "run_dir": None}
    registry_path.write_text(
        yaml.safe_dump({"runs": [run, dict(run)]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate configuration-seed"):
        collect_registry(registry_path, tmp_path / "archive")


def test_verify_with_evaluator_accepts_matching_scores(tmp_path):
    prediction = tmp_path / "prediction.csv"
    reference = tmp_path / "reference"
    _write_prediction(prediction)
    reference.mkdir()

    def fake_evaluate(*args, **kwargs):
        return (
            {"fmeasure": 12.0, "fmeasure-avg": 22.0},
            {},
            {},
            {},
        )

    verification = verify_with_evaluator(
        prediction,
        reference,
        {"fmeasure": 12.0, "fmeasureavg": 22.0},
        evaluator=fake_evaluate,
    )
    assert verification["passed"] is True
    assert verification["delta_fmeasure_harmonic"] == 0.0


def test_verify_with_evaluator_rejects_score_mismatch(tmp_path):
    prediction = tmp_path / "prediction.csv"
    reference = tmp_path / "reference"
    _write_prediction(prediction)
    reference.mkdir()

    def fake_evaluate(*args, **kwargs):
        return (
            {"fmeasure": 11.5, "fmeasure-avg": 22.0},
            {},
            {},
            {},
        )

    with pytest.raises(RuntimeError, match="does not reproduce"):
        verify_with_evaluator(
            prediction,
            reference,
            {"fmeasure": 12.0, "fmeasureavg": 22.0},
            evaluator=fake_evaluate,
        )
