# Recording evidence

A descriptive follow-up to the single-recording silence audit. It asks which
recordings contribute official-score gains and what their original detection
counts look like, not why their audio is difficult.

From the parent `silence_audit/` directory:

```bash
python3 -E -S -B scripts/profile_recordings.py --output-dir results/recordings01
```

Python 3.9+; standard library only. Choose a new output directory. Inputs are the
bundled matched counts, reference populations/subset memberships, and comparison
tables. No model, matching, raw annotation, audio, or notebook execution is needed.
The command uses the audit's existing `counts.py` and `fresh_counts.py` helpers,
then checks all 2,016 deletion cases before producing profiles.

## Cohort

The primary cohort uses official (`released`) scoring, originally nonempty
predictions, one affected subset, and at least one original TP. It has 932 eligible
cases, of which 255 gain, involving 13 distinct recording names. It is not the
union of the joint-deletion choices or a cohort of false-alarm-only outputs.

Profiles report gains/eligible outputs, original TP/FP/FN and precision/recall
ranges, and recurrence across configurations/seeds. Ranges labelled `gaining_*`
refer only to gaining outputs. All 24 outputs for each of the 13 names are also
exported so nongaining and TP-zero cases remain visible. Reused recordings make
these dependent observations, not 255 independent examples.

## Outputs

| File | Contents |
| --- | --- |
| `recording_profiles.csv` | 13 recording-level profiles |
| `recording_output_evidence.csv` | 13 × 24 original-output rows |
| `gaining_cases.csv` | 255 primary gaining rows |
| `subset_summary.csv` | Six subsets and their eligible denominators |
| `reconciliation.csv` | Deletion/profile/subset comparison checks |
| `summary.json`, `report.md` | Machine-readable and readable summaries |
| `manifest.json`, `sources/profile_recordings.py` | Input/output hashes, Python version, exact producer |

## Scope of the public adaptation

The original workspace check also re-read annotations for event-duration
summaries and inspected exploratory cached SNR metadata. Those steps are
intentionally excluded here: they are not needed for the count profiles and the
cache has no verified audio/runtime binding. Annotation hashes in the population
manifest identify prior reference copies; this command does not re-verify their
bytes or regenerate population counts from annotations.

`expected/recording_profiles.csv` preserves the original profile values after
removing its three annotation-duration columns; `expected/subset_summary.csv` is
byte-preserved. They are comparison targets only. Source hashes and the extraction
are recorded in the package source map.

No noise type, species, habitat, listening result, or causal explanation is
inferred. This is not an acoustic silence detector. On 2026-11-24, two relocated
runs with Python 3.14.4 produced byte-identical outputs, passed all 2,035
deletion/profile/subset comparisons, and refused existing output paths without
modifying them. Profiles were run before the audit in each copy. See the
[verification record](../provenance/counts-verification.json).

The same checks were repeated after the selective v03 figure refresh in two
new relocated copies. All nine profile files were byte-identical, all 2,035
comparisons passed, and the populated-directory, empty-directory and existing-file
refusals preserved their targets. Profiles again ran before the audit in each copy.
The [refreshed-package record](../provenance/runtime-verification-v03.json) pins
the unchanged code/scientific inputs; the earlier record remains historical evidence.
