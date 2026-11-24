"""
parse_logs.py
=============
Shared log-parsing utility for the DCASE 2024 Task 5 result notebooks.

All three notebooks import this module.  It reads stdout.log files produced
by train.py / test.py (the _Tee redirect) and returns structured data.

Log structure recap
-------------------
For each detection threshold k (0.00, 0.05, …, 0.95):

    post_process(alpha=k)  →  post_proc_A
    ┌─ "Before preprocessing:"
    │  "Overall_scores: {...}"          ← no-postproc score at this k
    ├─ "Threshold 0.2"
    │  "Overall_scores: {...}"
    ├─ "Threshold 0.3…"  (×4 total: 0.2, 0.3, 0.4, 0.5)
    │  "Overall_scores: {...}"
    └─ "******************BEST RESULT*****************"
       "<file>.wav {'TP': N, …}"        ← per-file counts
       "{'precision': …} <threshold>"  ← winning scores + threshold

    post_process_new(alpha=k)  →  post_proc_B
    ┌─ "Threshold length 0.05"
    │  "Overall_scores: {...}"
    ├─ … (×4 total: 0.05, 0.10, 0.15, 0.20)
    └─ "******************BEST RESULT*****************"
       "<file>.wav {'TP': N, …}"
       "{'precision': …} <threshold>"

Final global result (once per run):
    "Best Result: "
    "{'precision': …, 'fmeasure': X, …}"
"""

import ast
import json
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except ImportError:  # only the DataFrame helpers need it
    pd = None


def _require_pandas() -> None:
    """Guard for the DataFrame helpers.

    ``parse_log``, ``counts_with_backfill`` and ``load_subset_map`` are pure
    stdlib, so bootstrap_ci can import this module in environments where pandas
    is not installed.
    """
    if pd is None:
        raise ImportError(
            "pandas is required for the DataFrame helpers in parse_logs; "
            "parse_log() and counts_with_backfill() work without it."
        )

# ─────────────────────────────────────────────────────────────────────────────
# File → subset mapping
# ─────────────────────────────────────────────────────────────────────────────
# The evaluator derives subsets from the reference tree's folder names, giving
# six of them on the Validation split (HB, ME, PB, PB24, PW, RD).  Filenames
# cannot distinguish PB from PB24 -- both are BUK* -- so the mapping has to come
# from the dataset.  Generate it with tools/build_subset_map.py; without it we
# fall back to the prefix rules below, which merge PB and PB24 into five groups
# and land 1.1-1.9 F-measure points above the official score.

SUBSET_MAP_PATH = Path(__file__).with_name("file_to_subset.json")

# Filenames that appear in more than one Validation_Set folder.
#
# The evaluator lists each folder's files independently (gt_file_structure) and
# then aggregates per folder, so a file present in two folders is pooled into
# BOTH subsets -- its TP/FP/FN are counted twice.  file_to_subset.json can only
# record one subset per filename, so pooling from the map alone under-counts the
# duplicate and every affected subset scores too high.
#
# Established by audit_subset_partition.py against the reference tree and
# confirmed by notebooks/snr/per_file_snr.csv, which lists this file under both
# PB and PB24.  Reproducing the double-count makes all 40 BEST RESULT blocks in
# each of the seven run logs reconcile exactly with the evaluator's own printed
# harmonic and arithmetic means; without it every block is 0.03-0.5 points high.
SUBSET_DUPLICATES = {
    "BUK5_20180921_015906a.wav": ("PB", "PB24"),
}

_PREFIX_RULES = (
    ("buk", "PB"),  # merges PB and PB24 -- the reason the map exists
    ("me", "ME"),
    ("r4", "HB"),
    ("file_", "HB"),
    ("rd", "RD"),
    ("pw", "PW"),
)


@lru_cache(maxsize=1)
def load_subset_map() -> Optional[dict]:
    """Return {filename.wav: {'subset', 'n_pos_events'}}, or None if absent."""
    if not SUBSET_MAP_PATH.exists():
        warnings.warn(
            f"{SUBSET_MAP_PATH.name} not found -- falling back to filename prefixes, "
            "which merge PB and PB24. Per-subset scores will NOT match the official "
            "evaluator. Run tools/build_subset_map.py to fix.",
            stacklevel=2,
        )
        return None
    with open(SUBSET_MAP_PATH) as fh:
        return json.load(fh)


def _prefix_subset(filename: str) -> str:
    low = filename.lower()
    for prefix, subset in _PREFIX_RULES:
        if low.startswith(prefix):
            return subset
    raise KeyError(f"no subset rule matches {filename!r}")

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────────────────────

# Matches:  Overall_scores: {'precision': 0.04, ...}
_RE_OVERALL = re.compile(r"Overall_scores:\s*(\{[^}]+\})")

# Matches:  Before preprocessing:
_RE_BEFORE = re.compile(r"Before preprocessing:")

# Matches:  Threshold 0.3  (post_proc_A adaptive)
_RE_THRESH_A = re.compile(r"^Threshold\s+([\d.e+\-]+)\s*$")

# Matches:  Threshold length 0.15  (post_proc_B length)
_RE_THRESH_B = re.compile(r"^Threshold length\s+([\d.e+\-]+)\s*$")

# Matches the separator line before a BEST RESULT block
_RE_BEST_SEP = re.compile(r"\*{10,}BEST RESULT\*{10,}")

# Matches a per-file line:  filename.wav {'TP': 18, 'FP': 1, 'FN': 1, ...}
# Filename may contain spaces (e.g. "R4_cleaned recording_13-10-17.wav"), so use
# a lazy match up to the first ".wav" followed by whitespace and an opening brace.
_RE_FILE = re.compile(r"^(.+?\.wav)\s+(\{[^}]+\})")

# Matches the summary dict + winning threshold at end of a BEST RESULT block:
#   {'precision': 0.08, ...} 0.15000000000000002
_RE_BEST_SUMMARY = re.compile(r"(\{[^}]+\})\s+([\d.e+\-]+)\s*$")

# Matches the final "Best Result:" line
_RE_FINAL = re.compile(r"^Best Result:\s*$")

# Detects a dict-like line (used to capture the line after "Best Result:")
_RE_DICT = re.compile(r"^\{[^}]+\}\s*$")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dict(text: str) -> dict:
    """Safely parse a Python-literal dict string."""
    try:
        return ast.literal_eval(text.strip())
    except Exception:
        return {}


def _scores_from_dict(d: dict) -> dict:
    """Normalise a scores dict — handle both 'f-measure' and 'fmeasure' keys."""
    out = {}
    for k, v in d.items():
        key = k.replace("-", "")  # 'f-measure' → 'fmeasure'
        out[key] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(log_path: str) -> dict:
    """
    Parse a single stdout.log file.

    Returns
    -------
    dict with keys:
        'final'           : dict  – final Best Result scores
        'no_postproc'     : list[dict]  – no-postproc scores per detection-threshold k
                            each entry: {'k': float, **scores}
        'sweep_A'         : list[dict]  – scores for every (k, thresh_A) combo
                            each entry: {'k': float, 'thresh_a': float, **scores}
        'sweep_B'         : list[dict]  – scores for every (k, thresh_B) combo
                            each entry: {'k': float, 'thresh_b': float, **scores}
        'best_A'          : list[dict]  – per-k best from post_proc_A
                            each entry: {'k': float, 'winning_thresh': float, **scores,
                                         'per_file': {filename: {TP,FP,FN,total_n_pos_events}}}
        'best_B'          : list[dict]  – per-k best from post_proc_B (same shape)
        'winning_block'   : dict  – per-file TP/FP/FN of the globally winning BEST RESULT block
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    result = {
        "final": {},
        "no_postproc": [],
        "sweep_A": [],
        "sweep_B": [],
        "best_A": [],
        "best_B": [],
        "winning_block": {},
    }

    # State machine
    # We track which "section" we are in as we scan lines top-to-bottom.
    current_k: Optional[float] = None   # current detection threshold
    in_proc_A: bool = False             # inside a post_process() call
    in_proc_B: bool = False             # inside a post_process_new() call
    in_best_block: bool = False         # inside a BEST RESULT block
    current_best_proc: str = ""         # "A" or "B"
    current_best_files: dict = {}
    current_thresh_a: Optional[float] = None
    current_thresh_b: Optional[float] = None

    # We detect k by counting "Before preprocessing:" lines: the n-th occurrence
    # corresponds to detection threshold index n in [0.00, 0.05, …].
    # But the raw threshold value is never printed — we infer it by counting.
    before_count: int = 0

    # Detection thresholds from evaluate_prototypes:  arange(0, 1, 0.05) → 20 values
    _det_thresholds = [round(i * 0.05, 2) for i in range(20)]

    # Track whether we are past the last BEST RESULT block of a proc_A section
    # (used to detect the transition to proc_B)
    last_best_proc_A_index: int = -1

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # ── Final global result ──────────────────────────────────────────────
        if _RE_FINAL.match(line):
            # Next non-empty line is the dict
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                result["final"] = _scores_from_dict(_parse_dict(lines[j]))
            i = j + 1
            continue

        # ── BEST RESULT separator ────────────────────────────────────────────
        if _RE_BEST_SEP.search(line):
            in_best_block = True
            current_best_proc = "A" if in_proc_A else "B"
            current_best_files = {}
            i += 1
            continue

        # ── Inside a BEST RESULT block ───────────────────────────────────────
        if in_best_block:
            # Per-file line
            m = _RE_FILE.match(line)
            if m:
                fname = m.group(1)
                fdata = _parse_dict(m.group(2))
                current_best_files[fname] = fdata
                i += 1
                continue

            # Summary line: dict + winning threshold
            m = _RE_BEST_SUMMARY.match(line)
            if m:
                scores = _scores_from_dict(_parse_dict(m.group(1)))
                winning_thresh = float(m.group(2))
                entry = {
                    "k": current_k,
                    "winning_thresh": winning_thresh,
                    "per_file": current_best_files,
                    **scores,
                }
                if current_best_proc == "A":
                    result["best_A"].append(entry)
                    in_proc_A = False
                else:
                    result["best_B"].append(entry)
                    in_proc_B = False
                in_best_block = False
                current_best_files = {}
                i += 1
                continue

            # If line is empty or starts a new section, leave best block
            if not line.strip() or _RE_BEFORE.search(line) or _RE_THRESH_B.match(line):
                in_best_block = False
                # Do NOT increment — let the outer loop reprocess this line
                continue

            i += 1
            continue

        # ── "Before preprocessing:" → no-postproc score, start proc_A ───────
        if _RE_BEFORE.search(line):
            in_proc_A = True
            in_proc_B = False
            # The Overall_scores line immediately follows
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                m = _RE_OVERALL.search(lines[j])
                if m:
                    scores = _scores_from_dict(_parse_dict(m.group(1)))
                    # Infer current_k: each "Before preprocessing:" corresponds to
                    # a new detection threshold.  before_count gives the index.
                    current_k = _det_thresholds[before_count] if before_count < len(_det_thresholds) else None
                    before_count += 1
                    result["no_postproc"].append({"k": current_k, **scores})
            i = j + 1
            continue

        # ── Threshold X.X  (post_proc_A adaptive threshold sweep) ───────────
        if in_proc_A and not in_best_block:
            m = _RE_THRESH_A.match(line)
            if m:
                current_thresh_a = float(m.group(1))
                # Next non-empty line should be Overall_scores
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    om = _RE_OVERALL.search(lines[j])
                    if om:
                        scores = _scores_from_dict(_parse_dict(om.group(1)))
                        result["sweep_A"].append({
                            "k": current_k,
                            "thresh_a": current_thresh_a,
                            **scores,
                        })
                i = j + 1
                continue

        # ── Threshold length X.X  (post_proc_B length threshold sweep) ───────
        m = _RE_THRESH_B.match(line)
        if m:
            in_proc_B = True
            in_proc_A = False
            current_thresh_b = float(m.group(1))
            # Next non-empty line should be Overall_scores
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                om = _RE_OVERALL.search(lines[j])
                if om:
                    scores = _scores_from_dict(_parse_dict(om.group(1)))
                    result["sweep_B"].append({
                        "k": current_k,
                        "thresh_b": current_thresh_b,
                        **scores,
                    })
            i = j + 1
            continue

        i += 1

    # ── Derive winning_block: the per-file breakdown of the globally best k ──
    # The globally best result (result["final"]) came from either best_A or best_B.
    # Find the entry whose scores match the final fmeasure.
    final_fm = result["final"].get("fmeasure")
    for entry in result["best_A"] + result["best_B"]:
        if entry.get("fmeasure") == final_fm:
            result["winning_block"] = entry.get("per_file", {})
            break

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Multi-experiment loader
# ─────────────────────────────────────────────────────────────────────────────

def load_experiments(experiment_map: dict) -> dict:
    """
    Load and parse multiple log files.

    Parameters
    ----------
    experiment_map : dict
        Keys = experiment label (str), values = path to stdout.log (str).
        Example:
            {
                "Baseline":     "logs/.../baseline/.../stdout.log",
                "CIFE":         "logs/.../cife/.../stdout.log",
                "Phase 1":      "logs/.../phase1/.../stdout.log",
            }

    Returns
    -------
    dict  {label: parsed_result_dict}
    """
    results = {}
    for label, path in experiment_map.items():
        try:
            results[label] = parse_log(path)
            print(f"  ✓ {label:30s}  fmeasure={results[label]['final'].get('fmeasure', 'N/A')}")
        except FileNotFoundError as e:
            print(f"  ✗ {label:30s}  {e}")
            results[label] = None
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def final_results_df(experiments: dict) -> pd.DataFrame:
    """Return a DataFrame with one row per experiment, final Best Result scores."""
    _require_pandas()
    rows = []
    for label, data in experiments.items():
        if data is None:
            continue
        row = {"experiment": label}
        row.update(data["final"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("experiment")


def no_postproc_df(experiments: dict) -> pd.DataFrame:
    """Return a DataFrame with (experiment, k, scores) for no-postproc results."""
    _require_pandas()
    rows = []
    for label, data in experiments.items():
        if data is None:
            continue
        for entry in data["no_postproc"]:
            row = {"experiment": label}
            row.update(entry)
            rows.append(row)
    return pd.DataFrame(rows)


def sweep_A_df(experiments: dict) -> pd.DataFrame:
    """Return a DataFrame with (experiment, k, thresh_a, scores) for post_proc_A sweep."""
    _require_pandas()
    rows = []
    for label, data in experiments.items():
        if data is None:
            continue
        for entry in data["sweep_A"]:
            row = {"experiment": label}
            row.update(entry)
            rows.append(row)
    return pd.DataFrame(rows)


def sweep_B_df(experiments: dict) -> pd.DataFrame:
    """Return a DataFrame with (experiment, k, thresh_b, scores) for post_proc_B sweep."""
    _require_pandas()
    rows = []
    for label, data in experiments.items():
        if data is None:
            continue
        for entry in data["sweep_B"]:
            row = {"experiment": label}
            row.update(entry)
            rows.append(row)
    return pd.DataFrame(rows)


def counts_with_backfill(data: dict) -> tuple:
    """
    Return ``(counts_by_file, backfilled)`` for one parsed run's winning block.

    The log only prints files that had at least one predicted event, so files the
    evaluator scored but never printed are missing.  When file_to_subset.json is
    available they are backfilled with TP=0, FP=0, FN=n_pos_events, exactly as
    evaluation.py:273-291 does; without the backfill their FN is dropped and
    every affected subset scores too high.

    Pure stdlib -- shared by ``per_file_df`` and bootstrap_ci so the backfill
    rule lives in exactly one place.
    """
    subset_map = load_subset_map()
    counts_by_file = {k: dict(v) for k, v in data["winning_block"].items()}
    backfilled = set()
    if subset_map is not None:
        unknown = sorted(set(counts_by_file) - set(subset_map))
        if unknown:
            raise KeyError(
                f"{SUBSET_MAP_PATH.name} is stale -- these files appear in the "
                f"logs but not in the map: {unknown}"
            )
        for fname, meta in subset_map.items():
            if fname not in counts_by_file:
                n_pos = meta["n_pos_events"]
                counts_by_file[fname] = {
                    "TP": 0, "FP": 0, "FN": n_pos, "total_n_pos_events": n_pos,
                }
                backfilled.add(fname)
    return counts_by_file, backfilled


def subset_of(fname: str) -> str:
    """Primary subset for one filename, from the map when present else prefixes.

    For display only.  Use ``subsets_of`` when pooling -- a duplicated file
    contributes to more than one subset.
    """
    subset_map = load_subset_map()
    if subset_map is not None:
        return subset_map[fname]["subset"]
    return _prefix_subset(fname)


def subsets_of(fname: str) -> tuple:
    """Every subset one file is pooled into, honouring ``SUBSET_DUPLICATES``.

    This is the grouping the evaluator actually aggregates over, so it is what
    per-subset pooling must use.
    """
    if fname in SUBSET_DUPLICATES:
        return SUBSET_DUPLICATES[fname]
    return (subset_of(fname),)


def per_file_df(experiments: dict) -> pd.DataFrame:
    """
    Return a DataFrame with per-file TP/FP/FN from the globally winning block.

    Missing files are backfilled by ``counts_with_backfill``; those rows are
    flagged via ``from_backfill``.
    """
    _require_pandas()
    rows = []
    for label, data in experiments.items():
        if data is None:
            continue

        counts_by_file, backfilled = counts_with_backfill(data)

        for fname, counts in counts_by_file.items():
            tp = counts.get("TP", 0)
            fp = counts.get("FP", 0)
            fn = counts.get("FN", 0)
            total = counts.get("total_n_pos_events", tp + fn)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fm   = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0.0
            subset = subset_of(fname)
            rows.append({
                "from_backfill": fname in backfilled,
                "experiment": label,
                "filename": fname,
                "subset": subset,
                "TP": tp, "FP": fp, "FN": fn,
                "total_n_pos_events": total,
                "precision": prec,
                "recall": rec,
                "fmeasure": fm,
            })
    return pd.DataFrame(rows)


# Mirrors src/utils/evaluation.py -- keep in sync.
MIN_EVAL_VALUE = 0.00001


def subset_scores_df(df_pf: pd.DataFrame) -> pd.DataFrame:
    """
    Pool per-file counts into per-subset scores, the way the evaluator does.

    Counts are summed within each subset and the score computed on the pooled
    totals (evaluation.py:132-148, including the MIN_EVAL_VALUE floor), so these
    are the same per-subset values the official metric is built from.  Returns
    F / precision / recall on a 0-100 scale, indexed by (experiment, subset).
    """
    exploded = df_pf.assign(
        subset=[subsets_of(f) for f in df_pf["filename"]]
    ).explode("subset")
    pooled = exploded.groupby(["experiment", "subset"])[["TP", "FP", "FN"]].sum()
    tp, fp, fn = pooled["TP"], pooled["FP"], pooled["FN"]

    precision = (tp / (tp + fp)).where(tp + fp != 0, MIN_EVAL_VALUE)
    precision = precision.clip(lower=MIN_EVAL_VALUE)
    recall = (tp / (tp + fn)).where(tp != 0, MIN_EVAL_VALUE)
    fmeasure = (tp / (tp + 0.5 * (fp + fn))).where(tp != 0, MIN_EVAL_VALUE)

    out = pooled.copy()
    out["precision"] = precision * 100
    out["recall"] = recall * 100
    out["fmeasure"] = fmeasure * 100
    return out


def verify_against_evaluator(
    df_pf: pd.DataFrame, experiments: dict, tol: float = 0.002
) -> pd.DataFrame:
    """
    Check that the pooled per-subset scores reproduce the logged official score.

    The harmonic mean of the per-subset F-measures is the official DCASE metric
    and the arithmetic mean is its ``fmeasure-avg`` companion, so both must match
    what the evaluator printed.  A mismatch means the reconstruction is wrong --
    most likely a missing or stale file_to_subset.json, which merges PB into one
    group and lands 1.1-1.9 points high.

    Returns a per-experiment comparison and raises AssertionError on mismatch.
    """
    from scipy import stats

    subset_scores = subset_scores_df(df_pf)
    rows, bad = [], []
    for label, data in experiments.items():
        if data is None:
            continue
        f = subset_scores.loc[label, "fmeasure"]
        got_h = round(float(stats.hmean(f.clip(lower=MIN_EVAL_VALUE))), 3)
        got_a = round(float(f.mean()), 3)
        want_h, want_a = data["final"]["fmeasure"], data["final"]["fmeasureavg"]
        ok = abs(got_h - want_h) < tol and abs(got_a - want_a) < tol
        rows.append({
            "experiment": label, "n_subsets": len(f),
            "harmonic": got_h, "official_harmonic": want_h, "d_harmonic": got_h - want_h,
            "arithmetic": got_a, "official_arithmetic": want_a, "d_arithmetic": got_a - want_a,
            "reconciles": ok,
        })
        if not ok:
            bad.append(f"{label}: harmonic {got_h} vs {want_h}, arithmetic {got_a} vs {want_a}")

    report = pd.DataFrame(rows)
    if bad:
        raise AssertionError(
            "Per-subset reconstruction does not match the official evaluator:\n  "
            + "\n  ".join(bad)
            + f"\nGrouping used: {len(subset_scores.index.get_level_values('subset').unique())} subsets."
            + ("" if SUBSET_MAP_PATH.exists() else f"\n{SUBSET_MAP_PATH.name} is missing -- run tools/build_subset_map.py.")
        )
    return report
