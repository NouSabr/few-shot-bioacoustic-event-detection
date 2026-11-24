"""Tests for multi-seed result loading and summaries."""

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.multiseed_results import (
    evaluator_summary,
    filter_all,
    intervention_prevalence,
    load_silencing_directories,
    load_sweep_manifests,
    paired_delta_summary,
    paired_deltas,
    recording_recurrence,
    stratified_intervention_summary,
    validate_results,
)


def _run(configuration, seed, score):
    return {
        "configuration": configuration,
        "seed": seed,
        "artifact_status": "complete_verified",
        "logged_scores": {
            "fmeasure_harmonic": score,
            "fmeasure_arithmetic": score + 10,
            "precision_harmonic": 0.2,
            "recall_harmonic": 0.4,
        },
        "selection": {
            "detection_threshold": 0.9,
            "postprocessing_type": "fixed_length",
            "postprocessing_threshold": 0.1,
        },
    }


def _write_silencing(directory: Path):
    directory.mkdir(parents=True)
    interventions = pd.DataFrame(
        [
            {
                "configuration": configuration,
                "seed": seed,
                "evaluator_variant": variant,
                "target_filename": filename,
                "originally_silent": False,
                "target_tp": 1 if filename == "A.wav" else 0,
                "target_fp": 4 if filename == "A.wav" else 2,
                "target_fn": 0,
                "forfeits_true_positives": filename == "A.wav",
                "improves_any_subset": gain > 0,
                "improves_all_affected_subsets": gain > 0,
                "improves_official": gain > 0,
                "delta_fmeasure_harmonic": gain,
            }
            for configuration in ("baseline", "phase1_random")
            for seed in (1, 2)
            for variant in ("released", "corrected")
            for filename, gain in (("A.wav", 0.2), ("B.wav", -0.1))
        ]
    )
    subset = interventions.assign(
        subset="S",
        file_tp=interventions["target_tp"],
        file_fp=interventions["target_fp"],
        file_fn=interventions["target_fn"],
        observed_subset_improvement=interventions["improves_any_subset"],
        condition_holds=interventions["improves_any_subset"],
        condition_matches_observation=True,
    )
    summary = pd.DataFrame(
        [{"configuration": "baseline", "seed": 1, "evaluator_variant": "corrected"}]
    )
    skipped = pd.DataFrame(columns=["configuration", "seed", "reason"])
    interventions.to_csv(directory / "silencing_interventions.csv", index=False)
    subset.to_csv(directory / "silencing_subset_effects.csv", index=False)
    summary.to_csv(directory / "silencing_summary.csv", index=False)
    skipped.to_csv(directory / "skipped_runs.csv", index=False)


def test_load_filter_and_paired_summaries(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    _run("baseline", 1, 10.0),
                    _run("phase1_random", 1, 11.0),
                    _run("baseline", 2, 12.0),
                    _run("phase1_random", 2, 11.5),
                ]
            }
        ),
        encoding="utf-8",
    )
    silence_dir = tmp_path / "silencing"
    _write_silencing(silence_dir)

    scores = load_sweep_manifests([manifest])
    silencing = load_silencing_directories([silence_dir])
    report = validate_results(scores, silencing)
    assert report["condition_mismatches"] == 0

    delta = paired_deltas(scores)
    assert delta.loc[1, "phase1_random"] == 1.0
    assert delta.loc[2, "phase1_random"] == -0.5
    summary = paired_delta_summary(delta)
    assert summary.loc["phase1_random", "positive_seeds"] == 1
    assert summary.loc["phase1_random", "negative_seeds"] == 1

    filtered_scores, filtered_silencing = filter_all(
        scores, silencing, include_configurations=["baseline"], include_seeds=[2]
    )
    assert set(filtered_scores["configuration"]) == {"baseline"}
    assert set(filtered_scores["seed"]) == {2}
    assert set(filtered_silencing["interventions"]["configuration"]) == {"baseline"}


def test_silencing_summaries_and_recurrence(tmp_path):
    silence_dir = tmp_path / "silencing"
    _write_silencing(silence_dir)
    silencing = load_silencing_directories([silence_dir])
    interventions = silencing["interventions"]

    prevalence = intervention_prevalence(interventions)
    corrected = prevalence[prevalence["evaluator_variant"] == "corrected"]
    assert (corrected["aggregate_improvements"] == 1).all()
    assert (corrected["eligible_interventions"] == 2).all()

    overall = evaluator_summary(interventions)
    corrected_overall = overall[overall["evaluator_variant"] == "corrected"].iloc[0]
    assert corrected_overall["aggregate_improvements"] == 4
    assert corrected_overall["eligible_interventions"] == 8

    stratified = stratified_intervention_summary(interventions)
    corrected_strata = stratified[stratified["evaluator_variant"] == "corrected"]
    t_positive = corrected_strata[corrected_strata["target_status"] == "t > 0"].iloc[0]
    t_zero = corrected_strata[corrected_strata["target_status"] == "t = 0"].iloc[0]
    assert t_positive["aggregate_improvements"] == 4
    assert t_positive["eligible_interventions"] == 4
    assert t_zero["aggregate_improvements"] == 0
    assert t_zero["eligible_interventions"] == 4

    recurrence = recording_recurrence(interventions, "corrected")
    assert recurrence.loc["A.wav", "positive_seeds"] == 2
    assert recurrence.loc["A.wav", "positive_configurations"] == 2
    assert recurrence.loc["B.wav", "n_aggregate_improvements"] == 0
    substantive = recording_recurrence(
        interventions, "corrected", require_true_positive=True
    )
    assert list(substantive.index) == ["A.wav"]


def test_empty_legacy_skipped_csv_is_accepted(tmp_path):
    silence_dir = tmp_path / "silencing"
    _write_silencing(silence_dir)
    (silence_dir / "skipped_runs.csv").write_text("", encoding="utf-8")
    loaded = load_silencing_directories([silence_dir])
    assert loaded["skipped"].empty


def test_conflicting_duplicate_manifest_rows_fail(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"runs": [_run("baseline", 1, 10.0)]}), encoding="utf-8")
    second.write_text(json.dumps({"runs": [_run("baseline", 1, 11.0)]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        load_sweep_manifests([first, second])


def test_legacy_interventions_backfill_target_counts(tmp_path):
    silence_dir = tmp_path / "silencing"
    _write_silencing(silence_dir)
    path = silence_dir / "silencing_interventions.csv"
    frame = pd.read_csv(path).drop(
        columns=["target_tp", "target_fp", "target_fn", "forfeits_true_positives"]
    )
    frame.to_csv(path, index=False)

    loaded = load_silencing_directories([silence_dir])
    interventions = loaded["interventions"]
    assert {"target_tp", "target_fp", "target_fn"}.issubset(interventions.columns)
    assert interventions.loc[interventions["target_filename"] == "A.wav", "target_tp"].eq(1).all()
    assert interventions.loc[interventions["target_filename"] == "B.wav", "target_tp"].eq(0).all()


def test_oracle_removal_bound_handles_shared_file():
    from tools.multiseed_results import oracle_removal_bound

    rows = []
    # A is low-precision and shared across S1/S2; B and C preserve TP.
    specs = {
        "S1": {"A.wav": (1, 20, 0, 1), "B.wav": (10, 0, 0, 10)},
        "S2": {"A.wav": (1, 20, 0, 1), "C.wav": (8, 0, 0, 8)},
    }
    for subset, files in specs.items():
        tp = sum(v[0] for v in files.values())
        fp = sum(v[1] for v in files.values())
        fn = sum(v[2] for v in files.values())
        original_f = 100 * 2 * tp / (2 * tp + fp + fn)
        for name, (file_tp, file_fp, file_fn, silent_fn) in files.items():
            rows.append(
                {
                    "configuration": "baseline",
                    "seed": 1,
                    "evaluator_variant": "corrected",
                    "target_filename": name,
                    "subset": subset,
                    "file_tp": file_tp,
                    "file_fp": file_fp,
                    "file_fn": file_fn,
                    "silenced_file_fn": silent_fn,
                    "original_subset_f": original_f,
                }
            )
    result = oracle_removal_bound(pd.DataFrame(rows)).iloc[0]
    assert result["oracle_fmeasure_harmonic"] > result["original_fmeasure_harmonic"]
    assert "A.wav" in result["removed_recordings"]
    assert result["n_removed_recordings"] == 1
