"""Collect the selected prediction artifacts from an ablation sweep.

The training pipeline writes one directory per detection threshold and does not
copy the globally selected CSV to a stable location.  This module combines the
existing notebook log parser with explicit run metadata to identify, validate,
and archive that selected artifact without guessing from timestamps.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
# The CLI may be launched from the repository root or any other directory. Add
# the task root explicitly so the released evaluator's ``src.*`` imports do not
# depend on the caller's current working directory.
for import_root in (PROJECT_ROOT, NOTEBOOKS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from parse_logs import parse_log  # noqa: E402

SCORE_TOLERANCE = 0.001
FLOAT_TOLERANCE = 1e-8
CSV_HEADER = ["Audiofilename", "Starttime", "Endtime"]

STATUS_COMPLETE_VERIFIED = "complete_verified"
STATUS_COMPLETE_UNVERIFIED = "complete_unverified"
STATUS_LOG_ONLY = "log_only"
STATUS_PREDICTIONS_MISSING = "predictions_missing"
STATUS_SCORE_MISMATCH = "score_mismatch"
STATUS_AMBIGUOUS_RUN = "ambiguous_run"
STATUS_LOG_PARSE_FAILURE = "log_parse_failure"
STATUS_LOG_MISSING = "log_missing"
STATUS_MISSING_RUN = "missing_run"
STATUS_INVALID_CSV = "invalid_csv"
STATUS_DESTINATION_CONFLICT = "destination_conflict"
STATUS_VERIFICATION_UNAVAILABLE = "verification_unavailable"

INCOMPLETE_STATUSES = {
    STATUS_LOG_ONLY,
    STATUS_PREDICTIONS_MISSING,
    STATUS_SCORE_MISMATCH,
    STATUS_AMBIGUOUS_RUN,
    STATUS_LOG_PARSE_FAILURE,
    STATUS_LOG_MISSING,
    STATUS_MISSING_RUN,
    STATUS_INVALID_CSV,
    STATUS_DESTINATION_CONFLICT,
    STATUS_VERIFICATION_UNAVAILABLE,
}


class CollectorError(RuntimeError):
    """Expected collection failure for one run."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Winner:
    """The operating point selected by the original evaluation loop."""

    mode: str
    k: float
    postprocessing_threshold: float
    scores: Mapping[str, Any]
    per_file: Mapping[str, Any]

    @property
    def postprocessing_name(self) -> str:
        return "adaptive" if self.mode == "A" else "fixed_length"


@dataclass(frozen=True)
class LocatedArtifacts:
    """Source artifacts corresponding to one Winner."""

    threshold_dir: Path
    raw_csv: Path
    final_csv: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_scores(scores: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "fmeasure_harmonic": scores.get("fmeasure"),
        "fmeasure_arithmetic": scores.get("fmeasureavg"),
        "precision_harmonic": scores.get("precision"),
        "recall_harmonic": scores.get("recall"),
        "precision_arithmetic": scores.get("precisionavg"),
        "recall_arithmetic": scores.get("recallavg"),
    }


def _ordered_candidates(parsed: Mapping[str, Any]) -> List[Tuple[str, Mapping[str, Any]]]:
    """Return candidates in the same order as test_epoch_end evaluates them.

    For each increasing detection threshold, type A is considered before type B.
    The training code replaces the global winner only on a strict improvement,
    so preserving this order also reproduces its tie behaviour.
    """
    candidates: List[Tuple[str, Mapping[str, Any]]] = []
    by_mode = {
        "A": list(parsed.get("best_A", [])),
        "B": list(parsed.get("best_B", [])),
    }
    ks = sorted(
        {
            float(entry["k"])
            for entries in by_mode.values()
            for entry in entries
            if entry.get("k") is not None
        }
    )
    for k in ks:
        for mode in ("A", "B"):
            matches = [
                entry
                for entry in by_mode[mode]
                if entry.get("k") is not None
                and math.isclose(float(entry["k"]), k, abs_tol=FLOAT_TOLERANCE)
            ]
            if len(matches) > 1:
                raise CollectorError(
                    STATUS_LOG_PARSE_FAILURE,
                    f"log contains multiple type-{mode} winners for k={k}",
                )
            if matches:
                candidates.append((mode, matches[0]))
    return candidates


def select_winner(parsed: Mapping[str, Any]) -> Winner:
    final = parsed.get("final", {})
    final_f = final.get("fmeasure")
    if final_f is None:
        raise CollectorError(STATUS_LOG_PARSE_FAILURE, "log has no final F-measure")

    selected: Optional[Tuple[str, Mapping[str, Any]]] = None
    best_f = float("-inf")
    for mode, entry in _ordered_candidates(parsed):
        fmeasure = entry.get("fmeasure")
        if fmeasure is None:
            continue
        if float(fmeasure) > best_f:
            selected = (mode, entry)
            best_f = float(fmeasure)

    if selected is None:
        raise CollectorError(
            STATUS_LOG_PARSE_FAILURE,
            "log has no complete post-processing BEST RESULT blocks",
        )

    if not math.isclose(best_f, float(final_f), abs_tol=SCORE_TOLERANCE):
        raise CollectorError(
            STATUS_LOG_PARSE_FAILURE,
            f"recovered winner F={best_f:.3f} does not match final F={float(final_f):.3f}",
        )

    mode, entry = selected
    if entry.get("k") is None or entry.get("winning_thresh") is None:
        raise CollectorError(
            STATUS_LOG_PARSE_FAILURE,
            "winning block is missing k or post-processing threshold",
        )
    return Winner(
        mode=mode,
        k=float(entry["k"]),
        postprocessing_threshold=float(entry["winning_thresh"]),
        scores=entry,
        per_file=entry.get("per_file", {}),
    )


def _parse_float_name(path: Path) -> Optional[float]:
    try:
        return float(path.name)
    except ValueError:
        return None


def _threshold_from_filename(path: Path) -> Optional[float]:
    try:
        return float(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return None


def _unique_close_match(
    candidates: Iterable[Tuple[Optional[float], Path]],
    expected: float,
    description: str,
) -> Path:
    matches = [
        path
        for value, path in candidates
        if value is not None and math.isclose(value, expected, abs_tol=FLOAT_TOLERANCE)
    ]
    if not matches:
        raise CollectorError(
            STATUS_PREDICTIONS_MISSING,
            f"no {description} matches {expected}",
        )
    if len(matches) > 1:
        raise CollectorError(
            STATUS_AMBIGUOUS_RUN,
            f"multiple {description}s match {expected}: {matches}",
        )
    return matches[0]


def locate_artifacts(run_dir: Path, winner: Winner) -> LocatedArtifacts:
    if not run_dir.exists() or not run_dir.is_dir():
        raise CollectorError(STATUS_LOG_ONLY, f"run directory unavailable: {run_dir}")

    threshold_dir = _unique_close_match(
        ((_parse_float_name(path), path) for path in run_dir.iterdir() if path.is_dir()),
        winner.k,
        "detection-threshold directory",
    )
    raw_csv = threshold_dir / "Eval_raw.csv"
    if not raw_csv.is_file():
        raise CollectorError(
            STATUS_PREDICTIONS_MISSING,
            f"selected raw CSV is missing: {raw_csv}",
        )

    if winner.mode == "A":
        pattern = "Eval_VAL_threshold_ada_postproc_*.csv"
    else:
        pattern = "Eval_VAL_threshold_fix_length_postproc_*.csv"
    final_csv = _unique_close_match(
        ((_threshold_from_filename(path), path) for path in threshold_dir.glob(pattern)),
        winner.postprocessing_threshold,
        "post-processing CSV",
    )
    return LocatedArtifacts(threshold_dir, raw_csv, final_csv)


def validate_prediction_header(path: Path) -> None:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), None)
    except OSError as exc:
        raise CollectorError(STATUS_INVALID_CSV, f"cannot read {path}: {exc}") from exc
    if header != CSV_HEADER:
        raise CollectorError(
            STATUS_INVALID_CSV,
            f"unexpected CSV header in {path}: {header}; expected {CSV_HEADER}",
        )


def _count_differences(
    expected_counts: Mapping[str, Mapping[str, Any]],
    actual_counts: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    """Compact per-file TP/FP/FN differences for verification failures."""
    differences = []
    expected_files = set(expected_counts)
    actual_files = set(actual_counts)
    for filename in sorted(expected_files - actual_files):
        differences.append(f"{filename}: present in log, absent from evaluator output")
    for filename in sorted(actual_files - expected_files):
        differences.append(f"{filename}: absent from log, present in evaluator output")
    for filename in sorted(expected_files & actual_files):
        expected = expected_counts[filename]
        actual = actual_counts[filename]
        changed = []
        for key in ("TP", "FP", "FN", "total_n_pos_events"):
            if expected.get(key) != actual.get(key):
                changed.append(f"{key} {expected.get(key)}->{actual.get(key)}")
        if changed:
            differences.append(f"{filename}: " + ", ".join(changed))
    return differences


def verify_with_evaluator(
    prediction_csv: Path,
    reference_dir: Path,
    expected_scores: Mapping[str, Any],
    tolerance: float = SCORE_TOLERANCE,
    evaluator=None,
    expected_counts: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the repository evaluator in a temporary output directory."""
    if not reference_dir.is_dir():
        raise CollectorError(
            STATUS_SCORE_MISMATCH,
            f"reference directory unavailable: {reference_dir}",
        )
    if evaluator is None:
        try:
            from src.utils.evaluation import evaluate as evaluator
        except ImportError as exc:
            raise CollectorError(
                STATUS_VERIFICATION_UNAVAILABLE,
                f"official evaluator dependencies are unavailable: {exc}",
            ) from exc

    with tempfile.TemporaryDirectory(prefix="sweep-collector-eval-") as temp_dir:
        overall, individual_counts, _, _ = evaluator(
            str(prediction_csv),
            str(reference_dir),
            "sweep_collector_verification",
            "VAL",
            temp_dir,
        )

    expected_h = expected_scores.get("fmeasure")
    expected_a = expected_scores.get("fmeasureavg")
    got_h = overall.get("fmeasure")
    got_a = overall.get("fmeasure-avg")
    harmonic_ok = expected_h is not None and math.isclose(
        float(got_h), float(expected_h), abs_tol=tolerance
    )
    arithmetic_ok = expected_a is not None and math.isclose(
        float(got_a), float(expected_a), abs_tol=tolerance
    )
    verification = {
        "performed": True,
        "tolerance": tolerance,
        "passed": harmonic_ok and arithmetic_ok,
        "recomputed_fmeasure_harmonic": got_h,
        "recomputed_fmeasure_arithmetic": got_a,
        "delta_fmeasure_harmonic": (
            float(got_h) - float(expected_h) if expected_h is not None else None
        ),
        "delta_fmeasure_arithmetic": (
            float(got_a) - float(expected_a) if expected_a is not None else None
        ),
    }
    if not verification["passed"]:
        message = (
            "selected CSV does not reproduce the logged score: "
            f"harmonic {got_h} vs {expected_h}; arithmetic {got_a} vs {expected_a}"
        )
        if expected_counts is not None:
            differences = _count_differences(expected_counts, individual_counts)
            if differences:
                message += "; per-file differences: " + " | ".join(differences[:10])
                if len(differences) > 10:
                    message += f" | ... and {len(differences) - 10} more"
            else:
                message += (
                    "; per-file counts match the log, so the discrepancy is in "
                    "subset/reference aggregation (check duplicated filenames and "
                    "the exact Validation_Set tree)"
                )
        raise CollectorError(STATUS_SCORE_MISMATCH, message)
    return verification


def _resolve_path(value: Optional[str], registry_dir: Path) -> Optional[Path]:
    if value in (None, ""):
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = registry_dir / expanded
    return expanded.resolve()


def load_registry(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}
    if not isinstance(registry, dict):
        raise ValueError("registry root must be a mapping")
    runs = registry.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("registry 'runs' must be a list")
    return registry


def _git_state() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
            ).strip()
        )
        return {"collector_git_commit": commit, "collector_git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"collector_git_commit": None, "collector_git_dirty": None}


def _base_record(run: Mapping[str, Any], sweep_id: str) -> Dict[str, Any]:
    return {
        "sweep_id": sweep_id,
        "configuration": run.get("configuration"),
        "seed": run.get("seed"),
        "artifact_status": None,
        "message": None,
        "note": run.get("note"),
        "source": {},
        "selection": None,
        "logged_scores": None,
        "verification": {"performed": False, "passed": None},
        "checksums": {},
        "winning_per_file_counts": None,
        "provenance": {"source_git_commit": run.get("git_commit")},
    }


def _copy_idempotent(source: Path, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) == sha256_file(destination):
            return
        if not overwrite:
            raise CollectorError(
                STATUS_DESTINATION_CONFLICT,
                f"destination exists with different content: {destination}",
            )
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _archive_all_raw(run_dir: Path, destination: Path, overwrite: bool) -> List[Dict[str, Any]]:
    archived = []
    numeric_dirs = []
    for path in run_dir.iterdir():
        if not path.is_dir():
            continue
        value = _parse_float_name(path)
        if value is not None:
            numeric_dirs.append((value, path))
    numeric_dirs.sort(key=lambda item: item[0])
    for value, threshold_dir in numeric_dirs:
        source = threshold_dir / "Eval_raw.csv"
        if not source.is_file():
            continue
        target = destination / threshold_dir.name / "Eval_raw.csv"
        _copy_idempotent(source, target, overwrite)
        archived.append(
            {
                "k": value,
                "source": str(source),
                "archived": str(target),
                "sha256": sha256_file(target),
            }
        )
    return archived


def collect_run(
    run: Mapping[str, Any],
    registry_dir: Path,
    output_dir: Path,
    sweep_id: str,
    copy_files: bool,
    overwrite: bool,
    reference_dir: Optional[Path],
    all_raw: bool,
) -> Dict[str, Any]:
    record = _base_record(run, sweep_id)
    log_path = _resolve_path(run.get("log"), registry_dir)
    run_dir = _resolve_path(run.get("run_dir"), registry_dir)
    record["source"] = {
        "log": str(log_path) if log_path else None,
        "run_directory": str(run_dir) if run_dir else None,
    }

    try:
        if log_path is None or not log_path.is_file():
            raise CollectorError(STATUS_LOG_MISSING, f"log unavailable: {log_path}")
        record["checksums"]["log_sha256"] = sha256_file(log_path)
        destination = output_dir / f"seed_{run['seed']}" / str(run["configuration"])
        if copy_files:
            log_target = destination / "stdout.log"
            _copy_idempotent(log_path, log_target, overwrite)
            record.setdefault("archive", {})["log"] = str(log_target)

        parsed = parse_log(str(log_path))
        winner = select_winner(parsed)
        record["selection"] = {
            "postprocessing_type": winner.postprocessing_name,
            "postprocessing_mode": winner.mode,
            "detection_threshold": winner.k,
            "postprocessing_threshold": winner.postprocessing_threshold,
        }
        record["logged_scores"] = _normalise_scores(parsed["final"])
        record["winning_per_file_counts"] = winner.per_file

        if run_dir is None:
            raise CollectorError(
                STATUS_LOG_ONLY,
                "log parsed, but no prediction run directory is registered",
            )
        artifacts = locate_artifacts(run_dir, winner)
        validate_prediction_header(artifacts.raw_csv)
        validate_prediction_header(artifacts.final_csv)
        record["source"].update(
            {
                "threshold_directory": str(artifacts.threshold_dir),
                "raw_csv": str(artifacts.raw_csv),
                "final_csv": str(artifacts.final_csv),
            }
        )
        record["checksums"].update(
            {
                "raw_csv_sha256": sha256_file(artifacts.raw_csv),
                "final_csv_sha256": sha256_file(artifacts.final_csv),
            }
        )

        if reference_dir is not None:
            record["verification"] = verify_with_evaluator(
                artifacts.final_csv,
                reference_dir,
                parsed["final"],
                expected_counts=winner.per_file,
            )
            status = STATUS_COMPLETE_VERIFIED
        else:
            status = STATUS_COMPLETE_UNVERIFIED

        if copy_files:
            raw_target = destination / "selected_raw.csv"
            final_target = destination / "selected_final.csv"
            _copy_idempotent(artifacts.raw_csv, raw_target, overwrite)
            _copy_idempotent(artifacts.final_csv, final_target, overwrite)
            record["archive"] = {
                "raw_csv": str(raw_target),
                "final_csv": str(final_target),
            }
            if all_raw:
                record["archive"]["all_raw"] = _archive_all_raw(
                    run_dir, destination / "all_raw", overwrite
                )
        record["artifact_status"] = status
        record["message"] = "selected prediction artifacts located"
    except CollectorError as exc:
        record["artifact_status"] = exc.status
        record["message"] = str(exc)
    except Exception as exc:  # keep one bad run from hiding the rest of the matrix
        record["artifact_status"] = STATUS_LOG_PARSE_FAILURE
        record["message"] = f"unexpected collection error: {type(exc).__name__}: {exc}"
    return record


def _expected_missing_records(
    registry: Mapping[str, Any], existing: Sequence[Mapping[str, Any]], sweep_id: str
) -> List[Dict[str, Any]]:
    configurations = registry.get("expected_configurations", []) or []
    seeds = registry.get("expected_seeds", []) or []
    present = {(item.get("configuration"), item.get("seed")) for item in existing}
    missing = []
    for seed in seeds:
        for configuration in configurations:
            if (configuration, seed) in present:
                continue
            record = _base_record(
                {"configuration": configuration, "seed": seed}, sweep_id
            )
            record["artifact_status"] = STATUS_MISSING_RUN
            record["message"] = "expected configuration-seed pair is absent from registry"
            missing.append(record)
    return missing


def _check_duplicate_runs(runs: Sequence[Mapping[str, Any]]) -> None:
    seen = set()
    duplicates = set()
    for run in runs:
        key = (run.get("configuration"), run.get("seed"))
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise ValueError(
            "registry has duplicate configuration-seed entries; select one explicitly: "
            + ", ".join(f"{config}/seed={seed}" for config, seed in sorted(duplicates))
        )


def collect_registry(
    registry_path: Path,
    output_dir: Path,
    copy_files: bool = False,
    overwrite: bool = False,
    reference_dir_override: Optional[Path] = None,
    all_raw: bool = False,
) -> Dict[str, Any]:
    registry_path = registry_path.resolve()
    registry = load_registry(registry_path)
    runs = registry.get("runs", [])
    _check_duplicate_runs(runs)
    sweep_id = str(registry.get("sweep_id", registry_path.stem))
    registry_dir = registry_path.parent
    reference_dir = reference_dir_override or _resolve_path(
        registry.get("reference_dir"), registry_dir
    )

    records = [
        collect_run(
            run=run,
            registry_dir=registry_dir,
            output_dir=output_dir,
            sweep_id=sweep_id,
            copy_files=copy_files,
            overwrite=overwrite,
            reference_dir=reference_dir,
            all_raw=all_raw,
        )
        for run in runs
    ]
    records.extend(_expected_missing_records(registry, runs, sweep_id))
    summary: Dict[str, int] = {}
    for record in records:
        status = record["artifact_status"]
        summary[status] = summary.get(status, 0) + 1

    manifest = {
        "schema_version": 1,
        "sweep_id": sweep_id,
        "collected_at": utc_now(),
        "registry": str(registry_path),
        "reference_dir": str(reference_dir) if reference_dir else None,
        "copy_files": copy_files,
        "all_raw": all_raw,
        "summary": summary,
        "collector_provenance": _git_state(),
        "runs": records,
    }
    if copy_files:
        write_outputs(output_dir, manifest, overwrite=overwrite)
    return manifest


def _manifest_csv_rows(manifest: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    for record in manifest["runs"]:
        selection = record.get("selection") or {}
        scores = record.get("logged_scores") or {}
        verification = record.get("verification") or {}
        source = record.get("source") or {}
        yield {
            "sweep_id": record.get("sweep_id"),
            "configuration": record.get("configuration"),
            "seed": record.get("seed"),
            "artifact_status": record.get("artifact_status"),
            "message": record.get("message"),
            "postprocessing_type": selection.get("postprocessing_type"),
            "detection_threshold": selection.get("detection_threshold"),
            "postprocessing_threshold": selection.get("postprocessing_threshold"),
            "fmeasure_harmonic": scores.get("fmeasure_harmonic"),
            "fmeasure_arithmetic": scores.get("fmeasure_arithmetic"),
            "verification_passed": verification.get("passed"),
            "log": source.get("log"),
            "run_directory": source.get("run_directory"),
            "raw_csv": source.get("raw_csv"),
            "final_csv": source.get("final_csv"),
            "note": record.get("note"),
        }


def write_outputs(output_dir: Path, manifest: Mapping[str, Any], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # One self-contained metadata file per expected run makes log-only and later
    # recovered runs inspectable without loading the sweep-wide manifest. These
    # files are metadata, not source predictions, so they are refreshed on each
    # collection as artifact status evolves.
    for record in manifest["runs"]:
        configuration = record.get("configuration")
        seed = record.get("seed")
        if configuration is None or seed is None:
            continue
        run_output = output_dir / f"seed_{seed}" / str(configuration)
        run_output.mkdir(parents=True, exist_ok=True)
        selection_path = run_output / "selection.json"
        record.setdefault("archive", {})["selection"] = str(selection_path)
        selection_path.write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )

    manifest_path = output_dir / "manifest.json"
    manifest_csv = output_dir / "manifest.csv"
    report_path = output_dir / "collection_report.txt"
    missing_path = output_dir / "missing_artifacts.csv"

    if manifest_path.exists() and not overwrite:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        # Idempotent reruns are allowed; a newly collected timestamp alone should
        # not force --overwrite. Artifact copying already guards content changes.
        if existing is not None:
            pass

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows = list(_manifest_csv_rows(manifest))
    fieldnames = list(rows[0].keys()) if rows else []
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    incomplete = [row for row in rows if row["artifact_status"] in INCOMPLETE_STATUSES]
    with missing_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(incomplete)

    lines = [
        f"Sweep collector report: {manifest['sweep_id']}",
        f"Collected: {manifest['collected_at']}",
        "",
    ]
    for row in rows:
        marker = "OK" if row["artifact_status"].startswith("complete_") else "MISSING"
        lines.append(
            f"[{marker:7}] {str(row['configuration']):22} seed={row['seed']} "
            f"status={row['artifact_status']}"
        )
        if row["detection_threshold"] is not None:
            lines.append(
                "          "
                f"k={row['detection_threshold']} "
                f"post={row['postprocessing_type']} "
                f"threshold={row['postprocessing_threshold']} "
                f"F={row['fmeasure_harmonic']}"
            )
        if row["message"]:
            lines.append(f"          {row['message']}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_report(manifest: Mapping[str, Any]) -> str:
    lines = []
    for record in manifest["runs"]:
        status = record["artifact_status"]
        marker = "OK" if status.startswith("complete_") else "MISSING"
        lines.append(
            f"[{marker:7}] {str(record.get('configuration')):22} "
            f"seed={record.get('seed')} status={status}"
        )
        selection = record.get("selection") or {}
        scores = record.get("logged_scores") or {}
        if selection:
            lines.append(
                "          "
                f"k={selection.get('detection_threshold')} "
                f"post={selection.get('postprocessing_type')} "
                f"threshold={selection.get('postprocessing_threshold')} "
                f"F={scores.get('fmeasure_harmonic')}"
            )
        if record.get("message"):
            lines.append(f"          {record['message']}")
    return "\n".join(lines)


def has_incomplete_runs(manifest: Mapping[str, Any]) -> bool:
    return any(
        record.get("artifact_status") in INCOMPLETE_STATUSES
        for record in manifest.get("runs", [])
    )
