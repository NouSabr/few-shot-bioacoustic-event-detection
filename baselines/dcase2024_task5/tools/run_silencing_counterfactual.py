#!/usr/bin/env python3
"""Run one-recording-at-a-time silencing on collected sweep predictions."""

import argparse
from pathlib import Path

from silencing_counterfactual import (
    SUPPORTED_VARIANTS,
    CounterfactualError,
    run_manifest_analysis,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Remove all predictions for each validation recording in turn and "
            "measure the subset and aggregate score changes."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Collector manifest.json")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Validation_Set root containing the evaluator subset directories",
    )
    parser.add_argument("--output", type=Path, required=True, help="Analysis output directory")
    parser.add_argument(
        "--variant",
        action="append",
        choices=SUPPORTED_VARIANTS,
        dest="variants",
        help="Evaluator variant; repeatable. Default: released and corrected.",
    )
    parser.add_argument(
        "--configuration",
        action="append",
        help="Only analyse this configuration; repeatable.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Only analyse this seed; repeatable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any selected collector run is skipped or fails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        result = run_manifest_analysis(
            manifest_path=args.manifest.resolve(),
            reference_dir=args.reference_dir.resolve(),
            output_dir=args.output.resolve(),
            variants=tuple(args.variants or SUPPORTED_VARIANTS),
            configurations=args.configuration,
            seeds=args.seed,
            overwrite=args.overwrite,
        )
    except CounterfactualError as exc:
        raise SystemExit(f"Silencing counterfactual failed: {exc}") from exc

    print("Silencing counterfactual complete")
    print(f"  completed runs: {result['n_completed_runs']}")
    print(f"  skipped/failed runs: {result['n_skipped_or_failed_runs']}")
    print(f"  global intervention rows: {result['n_interventions']}")
    print(f"  subset-effect rows: {result['n_subset_effects']}")
    print(f"  output: {args.output.resolve()}")
    if args.strict and result["n_skipped_or_failed_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
