#!/usr/bin/env python3
"""
reconcile_subset_scores.py
==========================
Rebuild the official DCASE per-subset F-measures from the run logs and check
them against the score the evaluator itself printed.

This is the pass/fail gate for the per-subset numbers used in the paper.  It
reproduces ``evaluation.py``'s aggregation exactly:

    per file    counts come from the winning BEST RESULT block in the log
    backfill    files the log never printed get TP=0, FP=0, FN=n_pos_events
                (evaluation.py:273-291)
    per subset  pool TP/FP/FN, then F = TP / (TP + 0.5*(FP + FN))
                with the MIN_EVAL_VALUE floor (evaluation.py:132-148)
    overall     fmeasure     = harmonic mean of the per-subset F  (official)
                fmeasure-avg = arithmetic mean of the per-subset F

If ``file_to_subset.json`` is present the six real subsets are used and the
result must match the logged score.  Without it the script falls back to the
old filename-prefix grouping, which merges ``PB`` and ``PB24`` into five
groups, and reports how far off that is -- run
``tools/build_subset_map.py`` to get the map.

Pure standard library, so it runs anywhere the logs do.

Usage
-----
    python tools/reconcile_subset_scores.py --logs paper/run_logs
    python tools/reconcile_subset_scores.py --logs paper/run_logs --map notebooks/file_to_subset.json

Exit status is 1 if any run fails to reconcile (only enforced when a map is
supplied), so it works as a CI / pre-submission check.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys

# Mirrors src/utils/evaluation.py -- keep in sync.
MIN_EVAL_VALUE = 0.00001

_RE_SEP = re.compile(r"\*{10,}BEST RESULT\*{10,}")
_RE_FILE = re.compile(r"^(.+?\.wav)\s+(\{[^}]+\})")
_RE_SUMMARY = re.compile(r"^(\{[^}]+\})\s+([\d.eE+\-]+)\s*$")
_RE_FINAL = re.compile(r"^Best Result:\s*$")
_RE_DICT = re.compile(r"^(\{[^}]+\})\s*$")

# The pre-map grouping, kept only so the script still says something useful
# before the reference tree has been walked.  Merges PB and PB24.
_PREFIX_RULES = (
    ("buk", "PB"),
    ("me", "ME"),
    ("r4", "HB"),
    ("file_", "HB"),
    ("rd", "RD"),
    ("pw", "PW"),
)


def prefix_subset(filename: str) -> str:
    low = filename.lower()
    for prefix, subset in _PREFIX_RULES:
        if low.startswith(prefix):
            return subset
    raise KeyError(f"no prefix rule matches {filename!r}")


def parse_log(path: str) -> tuple[dict, dict]:
    """Return (final_scores, per_file_counts_of_the_winning_block)."""
    with open(path, errors="replace") as fh:
        lines = fh.read().split("\n")

    final = None
    for i, line in enumerate(lines):
        if _RE_FINAL.match(line):
            for j in range(i + 1, min(i + 4, len(lines))):
                m = _RE_DICT.match(lines[j].strip())
                if m:
                    final = ast.literal_eval(m.group(1))
                    break
    if final is None:
        raise SystemExit(f"{path}: no 'Best Result:' block found")

    blocks, current = [], {}
    for line in lines:
        if _RE_SEP.search(line):
            current = {}
            continue
        m_file = _RE_FILE.match(line)
        if m_file:
            current[m_file.group(1)] = ast.literal_eval(m_file.group(2))
            continue
        m_summary = _RE_SUMMARY.match(line.strip())
        if m_summary and current:
            blocks.append((ast.literal_eval(m_summary.group(1)), current))
            current = {}

    winning = [b for b in blocks if abs(b[0]["fmeasure"] - final["fmeasure"]) < 1e-9]
    if not winning:
        raise SystemExit(
            f"{path}: no BEST RESULT block matches the final score "
            f"{final['fmeasure']} -- log may be truncated"
        )
    return final, winning[0][1]


def scores_from_counts(tp: int, fp: int, fn: int) -> dict:
    """Byte-for-byte equivalent of evaluation.py:compute_scores_from_counts."""
    precision = tp / (tp + fp) if tp + fp != 0 else MIN_EVAL_VALUE
    if precision < MIN_EVAL_VALUE:
        precision = MIN_EVAL_VALUE
    recall = tp / (fn + tp) if tp != 0 else MIN_EVAL_VALUE
    fmeasure = tp / (tp + 0.5 * (fp + fn)) if tp != 0 else MIN_EVAL_VALUE
    return {"precision": precision, "recall": recall, "f-measure": fmeasure}


def hmean(values: list[float]) -> float:
    return len(values) / sum(1.0 / v for v in values)


def reconcile(final: dict, per_file: dict, subset_map: dict | None) -> dict:
    """Pool into subsets and compare with the evaluator's own numbers."""
    counts = {name: dict(c) for name, c in per_file.items()}
    backfilled = []

    if subset_map is not None:
        # evaluation.py:273-291 -- files with no predictions at all.
        for name, meta in subset_map.items():
            if name not in counts:
                n_pos = meta["n_pos_events"]
                counts[name] = {"TP": 0, "FP": 0, "FN": n_pos,
                                "total_n_pos_events": n_pos}
                backfilled.append(name)
        unknown = sorted(set(counts) - set(subset_map))
        if unknown:
            raise SystemExit(
                "Files in the log are absent from the subset map -- it is stale:\n  "
                + "\n  ".join(unknown)
            )
        subset_of = lambda name: subset_map[name]["subset"]  # noqa: E731
    else:
        subset_of = prefix_subset

    pooled: dict[str, list[int]] = {}
    for name, c in counts.items():
        acc = pooled.setdefault(subset_of(name), [0, 0, 0])
        acc[0] += c["TP"]
        acc[1] += c["FP"]
        acc[2] += c["FN"]

    per_subset = {s: scores_from_counts(*v) for s, v in sorted(pooled.items())}
    f_values = [s["f-measure"] for s in per_subset.values()]

    got_h = round(hmean(f_values) * 100, 3)
    got_a = round(sum(f_values) / len(f_values) * 100, 3)

    return {
        "per_subset": per_subset,
        "n_files": len(counts),
        "backfilled": backfilled,
        "harmonic": got_h,
        "arithmetic": got_a,
        "official_harmonic": final["fmeasure"],
        "official_arithmetic": final["fmeasure-avg"],
        "d_harmonic": got_h - final["fmeasure"],
        "d_arithmetic": got_a - final["fmeasure-avg"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconcile per-subset scores with the official evaluator")
    ap.add_argument("--logs", required=True, help="directory of *.log files (one per configuration)")
    ap.add_argument("--map", default=None, help="file_to_subset.json from build_subset_map.py")
    ap.add_argument("--tol", type=float, default=0.002, help="tolerance in F-measure points")
    args = ap.parse_args()

    subset_map = None
    if args.map:
        if not os.path.exists(args.map):
            raise SystemExit(f"Subset map not found: {args.map}\nRun tools/build_subset_map.py first.")
        with open(args.map) as fh:
            subset_map = json.load(fh)

    log_paths = sorted(glob.glob(os.path.join(args.logs, "*.log")))
    if not log_paths:
        raise SystemExit(f"No *.log files in {args.logs}")

    mode = (
        f"six-subset map ({len(subset_map)} files)"
        if subset_map
        else "filename-prefix fallback (PB and PB24 MERGED -- expect a gap)"
    )
    print(f"Grouping: {mode}\ntolerance: +/-{args.tol} F-measure points\n")

    header = f"{'run':18s} {'files':>5s} {'bf':>3s} {'grp':>3s} {'harmonic':>9s} {'official':>9s} {'delta':>7s}  {'arith':>7s} {'official':>8s} {'delta':>7s}  ok"
    print(header)
    print("-" * len(header))

    failures = []
    for path in log_paths:
        final, per_file = parse_log(path)
        r = reconcile(final, per_file, subset_map)
        ok = abs(r["d_harmonic"]) < args.tol and abs(r["d_arithmetic"]) < args.tol
        if not ok:
            failures.append((os.path.basename(path), r))
        print(
            f"{os.path.basename(path):18s} {r['n_files']:5d} {len(r['backfilled']):3d} "
            f"{len(r['per_subset']):3d} {r['harmonic']:9.3f} {r['official_harmonic']:9.3f} "
            f"{r['d_harmonic']:+7.3f}  {r['arithmetic']:7.3f} {r['official_arithmetic']:8.3f} "
            f"{r['d_arithmetic']:+7.3f}  {'YES' if ok else 'NO'}"
        )

    # Per-subset detail for the first run, so the actual numbers are visible.
    first, per_file = parse_log(log_paths[0])
    detail = reconcile(first, per_file, subset_map)
    print(f"\nper-subset F-measure ({os.path.basename(log_paths[0])}):")
    for subset, s in detail["per_subset"].items():
        print(f"   {subset:8s} F={s['f-measure'] * 100:6.2f}  P={s['precision'] * 100:6.2f}  R={s['recall'] * 100:6.2f}")
    if detail["backfilled"]:
        print(f"   backfilled (absent from log): {', '.join(detail['backfilled'])}")

    if subset_map is None:
        print(
            "\nNo subset map supplied, so this is the five-group reconstruction.\n"
            "The gap above is expected and is dominated by PB/PB24 being merged.\n"
            "Run tools/build_subset_map.py on a machine with the dataset mounted,\n"
            "then re-run with --map notebooks/file_to_subset.json."
        )
        return 0

    if failures:
        print(f"\nFAILED to reconcile {len(failures)} of {len(log_paths)} runs.")
        print("The per-subset table is NOT the official decomposition -- do not publish it.")
        return 1

    print(f"\nAll {len(log_paths)} runs reconcile within {args.tol} points.")
    print("The per-subset scores are the official ones; leverage figures can be recomputed from them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
