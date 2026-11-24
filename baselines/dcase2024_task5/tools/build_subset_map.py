#!/usr/bin/env python3
"""
build_subset_map.py
===================
Build the authoritative ``filename -> (subset, n_pos_events)`` map from the
DCASE 2024 Task 5 reference tree, and write it to JSON.

Why this exists
---------------
The result notebooks reconstruct per-subset scores by pooling the per-file
TP/FP/FN printed in each run's stdout log.  That reconstruction needs two
things the logs cannot supply:

1.  **The real subset of every file.**  ``parse_logs.py`` used to infer it from
    the filename prefix, which merged the evaluator's ``PB`` and ``PB24``
    folders into one group (all their files start with ``BUK``).  Five groups
    instead of six is the single largest source of disagreement with the
    official score -- around 1.1 to 1.9 F-measure points.

2.  **The positive-event count of files the log never printed.**  The evaluator
    only prints files that had at least one predicted event
    (``evaluation.py`` populates ``individual_file_result`` inside the loop over
    ``pred_events_by_audiofile``).  Files with *no* predictions are backfilled
    as ``TP=0, FP=0, FN=<all positive events>`` and never appear in the log, so
    a log-only reconstruction silently loses their FN and scores too high.

Both come straight from the reference tree, so this script only needs the
dataset -- no model, no re-running the evaluator.

Note on the 5 shots
-------------------
The evaluator is deliberately reproduced here, including an inconsistency:
for files *with* predictions it strips the first ``N_SHOTS`` events from the
reference (``remove_shots_from_ref``), but the backfill path counts **all**
positive events without stripping them.  ``n_pos_events`` below therefore
counts every positive event, matching the backfill.  Do not "fix" this --
it would stop the reconstruction from reproducing the official score.

Usage
-----
Run on a machine where the Validation_Set is mounted::

    python tools/build_subset_map.py \
        --val-dir /media/helios/Data/Uni/Masters/Dataset/Development_Set/Validation_Set \
        --out notebooks/file_to_subset.json

The default ``--val-dir`` is read from ``configs/train.yaml``
(``path.eval_dir``).  Commit the resulting JSON -- it is small and it makes the
notebooks reproducible on machines without the dataset attached.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Import the evaluator's own helpers so the positive-event semantics cannot
# drift from the official implementation.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import pandas as pd  # noqa: E402  (after sys.path fix)

from src.utils.evaluation import POS_VALUE, select_events_with_value  # noqa: E402

DEFAULT_VAL_DIR = (
    "/media/helios/Data/Uni/Masters/Dataset/Development_Set/Validation_Set"
)


def build_map(val_dir: str) -> dict:
    """Walk the reference tree and return {filename.wav: {subset, n_pos_events}}."""
    if not os.path.isdir(val_dir):
        raise SystemExit(
            f"Validation set not found: {val_dir}\n"
            "Pass the correct path with --val-dir (see path.eval_dir in configs/train.yaml)."
        )

    mapping: dict[str, dict] = {}
    subsets = sorted(
        d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))
    )
    if not subsets:
        raise SystemExit(f"No subset folders inside {val_dir}")

    for subset in subsets:
        csvs = sorted(glob.glob(os.path.join(val_dir, subset, "*.csv")))
        for csv_path in csvs:
            ref = pd.read_csv(csv_path, dtype=str)
            # Count ALL positive events (no shot removal) -- see module docstring.
            n_pos = len(select_events_with_value(ref, value=POS_VALUE))
            wav = os.path.basename(csv_path)[: -len(".csv")] + ".wav"
            if wav in mapping:
                raise SystemExit(
                    f"Duplicate filename across subsets: {wav} "
                    f"({mapping[wav]['subset']} and {subset}). "
                    "The reconstruction assumes filenames are unique."
                )
            mapping[wav] = {"subset": subset, "n_pos_events": int(n_pos)}
        print(f"  {subset:8s} {len(csvs):3d} files")

    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--val-dir", default=DEFAULT_VAL_DIR, help="path to Validation_Set")
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(_HERE), "notebooks", "file_to_subset.json"),
        help="output JSON path",
    )
    args = ap.parse_args()

    print(f"Reading reference tree: {args.val_dir}")
    mapping = build_map(args.val_dir)

    by_subset: dict[str, int] = {}
    for meta in mapping.values():
        by_subset[meta["subset"]] = by_subset.get(meta["subset"], 0) + 1

    with open(args.out, "w") as fh:
        json.dump(mapping, fh, indent=1, sort_keys=True)
        fh.write("\n")

    print(
        f"\nWrote {len(mapping)} files across {len(by_subset)} subsets -> {args.out}\n"
        f"  subsets: {', '.join(f'{k}={v}' for k, v in sorted(by_subset.items()))}\n\n"
        "Next: python tools/reconcile_subset_scores.py --logs paper/run_logs"
    )


if __name__ == "__main__":
    main()
