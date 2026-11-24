"""
Reproduce the DCASE evaluator's per-subset aggregation from an existing
stdout.log, using the actual folder partition of the validation set.

Usage (on helios):
    python audit_subset_partition.py \
        --log /home/helios/Downloads/run_logs/baseline.log \
        --ref /path/to/Development_Set/Validation_Set

Expected output: per-folder F, harmonic mean (should match reported micro),
arithmetic mean (should match reported macro), and a duplicate-file report.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import parse_logs as pl

MIN_EVAL_VALUE = 1e-8


def per_file_counts_from_log(log_path):
    """Return {filename: {'TP':..,'FP':..,'FN':..}} for the winning block."""
    result = pl.parse_log(log_path)
    block = result.get("winning_block", {})
    if not block:
        raise RuntimeError(
            f"No winning_block found in {log_path}. Check parse_logs output."
        )
    return block


def file_to_folders(ref_dir):
    """Return {filename: [folder, ...]} from the reference directory."""
    ref_dir = Path(ref_dir)
    mapping = defaultdict(list)
    for folder in sorted(p for p in ref_dir.iterdir() if p.is_dir()):
        for wav in folder.glob("*.wav"):
            mapping[wav.name].append(folder.name)
    return dict(mapping)


def f_from_counts(tp, fp, fn):
    if tp == 0:
        return MIN_EVAL_VALUE
    return tp / (tp + 0.5 * (fp + fn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to stdout.log")
    ap.add_argument("--ref", required=True, help="Validation_Set directory")
    args = ap.parse_args()

    per_file = per_file_counts_from_log(args.log)
    file_folder = file_to_folders(args.ref)

    # Duplicate-file report
    dupes = {f: fs for f, fs in file_folder.items() if len(fs) > 1}
    print(f"\n{'='*60}")
    print(f"Reference directory: {args.ref}")
    print(f"Folders: {sorted({f for fs in file_folder.values() for f in fs})}")
    print(f"Files in reference:  {len(file_folder)}")
    print(f"Files in log block:  {len(per_file)}")
    if dupes:
        print(f"\nDuplicate-file report ({len(dupes)} filenames in >1 folder):")
        for f, fs in dupes.items():
            print(f"  {f:40s}  in {fs}")
    else:
        print("\nNo duplicate filenames across folders.")

    # Pool counts per folder (double-counting duplicates -- matches evaluator)
    folder_counts = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    unmatched = []
    for fname, counts in per_file.items():
        folders = file_folder.get(fname)
        if not folders:
            unmatched.append(fname)
            continue
        for folder in folders:
            for k in ("TP", "FP", "FN"):
                folder_counts[folder][k] += counts.get(k, 0)

    if unmatched:
        print(f"\nWARN: {len(unmatched)} files in log had no folder match:")
        for f in unmatched[:5]:
            print(f"  {f}")
        if len(unmatched) > 5:
            print(f"  ... and {len(unmatched) - 5} more")

    # Compute F per folder
    print(f"\n{'='*60}")
    print(f"{'folder':<12s} {'TP':>8s} {'FP':>8s} {'FN':>8s} {'F (%)':>10s}")
    print("-" * 60)
    f_values = []
    for folder in sorted(folder_counts):
        c = folder_counts[folder]
        f = f_from_counts(c["TP"], c["FP"], c["FN"])
        f_values.append(f)
        print(f"{folder:<12s} {c['TP']:>8d} {c['FP']:>8d} {c['FN']:>8d} "
              f"{f*100:>10.4f}")

    # Aggregate
    micro = stats.hmean(f_values) * 100
    macro = np.mean(f_values) * 100
    print("-" * 60)
    print(f"Micro (harmonic mean across {len(f_values)} folders): {micro:.4f}")
    print(f"Macro (arithmetic mean across {len(f_values)} folders): {macro:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
