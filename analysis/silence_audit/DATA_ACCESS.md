# Data and reference access

## Included inputs

| Input | Role |
| --- | --- |
| `inputs/fresh_counts.csv` | Matched per-recording TP/FP/FN and prediction presence, for both policies |
| `inputs/reference_populations.csv` | Subset memberships, full/query POS counts, and annotation hashes |
| `inputs/predictions/`, `inputs/prediction_manifest.csv` | 24 selected prediction CSVs and their SHA-256 hashes |
| `inputs/joint_deletions.csv` | Archived joint choices; the audit uses their set membership |
| `inputs/archived_endpoints.csv` | Independent endpoint comparison targets |
| `inputs/plotting/` | Archived aggregate counts for the final rule-check figure |
| `expected/` | Audit comparison targets, never substituted for arithmetic |
| `recording_evidence/expected/` | Comparison targets for the descriptive follow-up |

These derived files contain no audio or model weights. Their publication/use
permissions still require author review. A code MIT notice is not a data license.

## External annotations for optional matching

Raw annotations and audio are not bundled. Counts-only replay and recording
profiles do not need either. Prediction-to-count replay needs the exact CSVs named
and hashed in `inputs/reference_populations.csv`.

Dataset access is described on the DCASE 2024 Task 5 page:
https://dcase.community/challenge2024/task-few-shot-bioacoustic-event-detection

The linked development-set record is DOI `10.5281/zenodo.10829604`, version v1
(published 2024-03-18):
https://zenodo.org/records/10829604
As checked on 2026-11-24, its description states CC BY 4.0 and provides
`Development_set_annotations.zip` (372.3 kB), containing annotation CSVs without
audio. The record advertises MD5 `893e656b957ebec36954cfc8ec2b2149` for that ZIP.
This is an access/version lead, not a verified replacement for the package's
per-reference SHA-256 targets: the archive was not downloaded or compared in
this review. Retain the dataset citation and applicable attribution; do not
relabel dataset material as project-MIT code. Exact annotation-byte equivalence
and publication approval for this package's derived outputs remain open.

Pass a directory containing the reference CSVs to `--references-root`. The matching
adapter searches recursively by CSV basename **and SHA-256**, then builds the
required subset hierarchy under its new output directory without changing bytes.
You can retain the normal `Validation_Set/<subset>/*.csv` layout. There are 43
subset memberships but 42 unique recording names: the shared PB/PB24 name uses
identical reference bytes and is copied into both subset directories.

The manifest's `source` column is historical provenance, not a required local
path. Do not flatten away subset membership, rename files, strip BOMs, reorder
rows, relabel annotations, or change times to make a hash/score agree. Support
selection depends on stored annotation order. See [support policy](SUPPORT_POLICY.md).

A current dataset download is not assumed byte-identical to the replay references.
If hashes differ, stop and resolve the version/provenance. Exact copies may need
to be obtained from the authors subject to data permissions; no automatic download
or right to redistribute them is asserted here.

## Limits and generated data

The hashes identify the references used for the frozen counts, not a complete
reconstruction of the original historical reference tree or training environment.
Reproducing arithmetic does not independently validate event matching.

Use output paths below `results/`, which is ignored. Optional matching copies
external annotations into its output directory; do not commit or publish those
results indiscriminately. No weights or raw annotations should be added to this
package as part of routine cleanup.
