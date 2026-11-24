#!/usr/bin/env python3
"""CLI for collecting the selected prediction CSV from each sweep run."""

import argparse
from pathlib import Path

from sweep_collector import collect_registry, format_report, has_incomplete_runs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Locate the globally selected prediction CSV for each registered "
            "configuration/seed run and archive it with a provenance manifest."
        )
    )
    parser.add_argument("--registry", type=Path, required=True, help="Sweep registry YAML")
    parser.add_argument("--output", type=Path, required=True, help="Archive output directory")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy artifacts and write manifests. Without this flag the command is a dry run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace archived files when their content differs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any expected run is incomplete.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Validation_Set root; enables official-evaluator score verification.",
    )
    parser.add_argument(
        "--all-raw",
        action="store_true",
        help="Also archive Eval_raw.csv for every available detection threshold.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = collect_registry(
        registry_path=args.registry,
        output_dir=args.output.resolve(),
        copy_files=args.copy,
        overwrite=args.overwrite,
        reference_dir_override=(args.reference_dir.resolve() if args.reference_dir else None),
        all_raw=args.all_raw,
    )
    mode = "COPY" if args.copy else "DRY RUN"
    print(f"Sweep collector: {mode}\n")
    print(format_report(manifest))
    print("\nSummary:")
    for status, count in sorted(manifest["summary"].items()):
        print(f"  {status}: {count}")
    if not args.copy:
        print("\nNo files were written. Re-run with --copy after reviewing the report.")
    if args.strict and has_incomplete_runs(manifest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
