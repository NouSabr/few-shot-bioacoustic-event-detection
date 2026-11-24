"""Reconstruct per-subset scores from the run logs and print the paper's tables.

The official evaluator prints aggregate scores plus per-file TP/FP/FN, but never
per-subset scores, so the six per-subset F-measures have to be pooled from the
per-file counts.  Two details make or break the agreement with the official
numbers (see the checks printed at the end):

  * A file for which a configuration predicts nothing produces no log line at
    all.  Those files must be counted as all-false-negative, not dropped --
    dropping them puts the baseline's F_harm at 16.95 against the official
    16.74, while imputing gives 16.77.
  * ``n_pos_events`` in ``file_to_subset.json`` counts every annotated event,
    whereas the evaluator's ``total_n_pos_events`` excludes the five support
    events.  The offset is exactly 5 on all 42 files, so imputed FN is
    ``n_pos_events - 5``.

Pooling is micro (sum counts, then compute F) -- macro-averaging per-file F
within a subset puts the baseline's arithmetic mean at 31.63 against 28.71.

Usage:  python3 baselines/dcase2024_task5/tools/six_subset_tables.py
"""

import ast
import re
from pathlib import Path
import json

import numpy as np

REPO = Path(__file__).resolve().parents[3]
LOG_DIR = REPO / "paper" / "run_logs"
SUBSET_MAP = REPO / "baselines/dcase2024_task5/notebooks/file_to_subset.json"

SUPPORT_EVENTS = 5  # the 5-shot support events the evaluator excludes
SUBSETS = ["HB", "ME", "PB", "PB24", "PW", "RD"]
RUNS = [
    ("Baseline", "baseline"),
    ("CIFE", "CIFE"),
    ("Phase 1 (Random)", "phase_1"),
    ("Phase 2 (Energy)", "phase_2"),
    ("Phase 3 (Activation)", "phase_3"),
    ("AFRI (Direct)", "afri_direct"),
    ("AFRI (Inverse)", "afri_inverse"),
]

RE_FILE = re.compile(r"^(.+?\.wav)\s+(\{[^}]+\})")
RE_SCORES = re.compile(r"^(\{'precision'.*\})\s*([\d.e+\-]*)\s*$")


def best_block(path):
    """Return (overall_scores, per_file_counts) for the highest-F_harm block."""
    blocks, current = [], {}
    for line in path.read_text().splitlines():
        m = RE_FILE.match(line)
        if m:
            current[m.group(1)] = ast.literal_eval(m.group(2))
            continue
        m = RE_SCORES.match(line)
        if m and current:
            try:
                blocks.append((ast.literal_eval(m.group(1)), dict(current)))
            except (ValueError, SyntaxError):
                pass
            current = {}
    return max(blocks, key=lambda b: b[0]["fmeasure"])


def fmeasure(tp, fp, fn):
    return 100 * tp / (tp + 0.5 * (fp + fn)) if tp else 0.0


def hmean(values):
    return len(values) / sum(1.0 / max(v, 1e-5) for v in values)


def per_subset(counts, subset_map):
    """Micro-pool per-file counts into per-subset F, P, R."""
    full = dict(counts)
    for name, meta in subset_map.items():
        if name not in full:
            full[name] = {
                "TP": 0,
                "FP": 0,
                "FN": meta["n_pos_events"] - SUPPORT_EVENTS,
            }
    F, P, R = {}, {}, {}
    for s in SUBSETS:
        files = [f for f in full if subset_map[f]["subset"] == s]
        tp = sum(full[f]["TP"] for f in files)
        fp = sum(full[f]["FP"] for f in files)
        fn = sum(full[f]["FN"] for f in files)
        F[s] = fmeasure(tp, fp, fn)
        P[s] = 100 * tp / (tp + fp) if tp + fp else 0.0
        R[s] = 100 * tp / (tp + fn) if tp + fn else 0.0
    return F, P, R


def main():
    subset_map = json.loads(SUBSET_MAP.read_text())
    res = {}
    for label, run in RUNS:
        overall, counts = best_block(LOG_DIR / f"{run}.log")
        res[label] = (per_subset(counts, subset_map), overall, counts)

    print("PER-SUBSET F-MEASURE (Table: tab:per_species_fmeasure)")
    print(f"{'Configuration':22s}" + "".join(f"{s:>8s}" for s in SUBSETS)
          + f"{'arith':>8s}{'off_a':>8s}{'harm':>8s}{'off_h':>8s}")
    for label, _ in RUNS:
        (F, _, _), o, _ = res[label]
        vals = [F[s] for s in SUBSETS]
        print(f"{label:22s}" + "".join(f"{F[s]:8.2f}" for s in SUBSETS)
              + f"{np.mean(vals):8.2f}{o['fmeasure-avg']:8.2f}"
              + f"{hmean(vals):8.2f}{o['fmeasure']:8.2f}")

    err_a = max(abs(np.mean([res[l][0][0][s] for s in SUBSETS]) - res[l][1]["fmeasure-avg"])
                for l, _ in RUNS)
    err_h = max(abs(hmean([res[l][0][0][s] for s in SUBSETS]) - res[l][1]["fmeasure"])
                for l, _ in RUNS)
    print(f"\nagreement with official:  F_arith within {err_a:.3f}, F_harm within {err_h:.3f}")

    Fb = res["Baseline"][0][0]
    order = sorted(SUBSETS, key=lambda s: Fb[s])
    strong = order[-1]
    print(f"\nMETRIC SENSITIVITY, (F_strong/F_s)^2 relative to {strong}={Fb[strong]:.2f}"
          "  (Fig: fig_metric_sensitivity)")
    for s in order:
        print(f"  {s:5s} F={Fb[s]:7.4f}  weight {(Fb[strong] / Fb[s]) ** 2:8.3f}x")

    Fc = res["CIFE"][0][0]
    base_h = hmean([Fb[s] for s in SUBSETS])
    print(f"\nCIFE SUBSTITUTED INTO BASELINE, one subset at a time "
          f"(baseline pooled F_harm={base_h:.3f})")
    for s in SUBSETS:
        v = [Fc[s] if x == s else Fb[x] for x in SUBSETS]
        print(f"  {s:5s} {Fb[s]:6.2f} -> {Fc[s]:6.2f} ({Fc[s] - Fb[s]:+6.2f})"
              f"   F_harm {hmean(v):7.3f} ({hmean(v) - base_h:+.3f})")

    print(f"\nMASKING DELTAS vs baseline\n{'':24s}"
          + "".join(f"{s:>8s}" for s in SUBSETS))
    for label, _ in RUNS[2:]:
        F = res[label][0][0]
        print(f"  {label:22s}" + "".join(f"{F[s] - Fb[s]:+8.2f}" for s in SUBSETS))
    print("  range over the five masking variants:")
    for s in SUBSETS:
        lo = min(res[l][0][0][s] - Fb[s] for l, _ in RUNS[2:])
        hi = max(res[l][0][0][s] - Fb[s] for l, _ in RUNS[2:])
        print(f"    {s:5s} {lo:+6.2f} .. {hi:+6.2f}")

    print("\nPER-SUBSET PRECISION / RECALL")
    for label, _ in RUNS:
        (_, P, R) = res[label][0]
        print(f"  {label:22s} P " + "".join(f"{P[s]:7.2f}" for s in SUBSETS)
              + "  | R " + "".join(f"{R[s]:7.2f}" for s in SUBSETS))

    print("\nME FILE-LEVEL (Section: ssec:case_me)")
    for label, _ in RUNS:
        counts = res[label][2]
        cells = [f" {f}={fmeasure(counts[f]['TP'], counts[f]['FP'], counts[f]['FN']):6.2f}"
                 for f in ("ME1.wav", "ME2.wav") if f in counts]
        print(f"  {label:22s}" + "".join(cells))


if __name__ == "__main__":
    main()
