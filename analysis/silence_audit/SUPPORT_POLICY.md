# Official versus support-consistent accounting

The official evaluator excludes the first five POS support examples when a
recording has prediction rows. In its missing-output branch it instead counts
all positive annotations as misses. The unmodified original matcher is used
before arithmetic interventions; no new event matching is needed when an entire
recording's predictions are removed and other recordings are held fixed.

For a recording with TP=t, FP=f, and FN=n:
- Official deletion: TP'=FP'=0; FN'=all positive reference events, including support.
- Support-consistent deletion: TP'=FP'=0; FN'=query-positive events only.
- Already empty official outputs already include support misses; do not add another five.
- Recording-level F1/recall always use fixed query references. They are numerically
  identical to ordinary official per-recording F1/recall here because the support
  difference occurs only when TP=0. Do not generalize that to arbitrary policies.

The actual supplied implementation is in `scripts/fresh_counts.py`/`counts.py`.
The original helper is also archived in `provenance/historical_count_policy.py`.
These should not be confused with changing event matching, lowering an IoU
threshold, removing false alarms selectively, or choosing deletion without labels.

The additional unified patch under `patches/` changes only the official
empty-output branch, reusing its existing support-removal routine. That branch
originally reads string timestamps, so the reference patch explicitly reads
numeric Starttime/Endtime before removal to avoid lexicographic comparisons.
The patch is a new reference artifact, not a historical patch or a fresh-matching
validation result. The default optional matcher preserves the historical
unmodified-official-then-backfill path and does not apply this patch.
