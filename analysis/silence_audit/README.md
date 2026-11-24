# Silence audit

Reproduce the scoring audit from frozen predictions and matched counts, separately
from model training. **Silencing means removing all prediction rows for a recording;
it does not mean editing or muting audio.** Other predictions, the selected
threshold, and post-processing stay fixed.

This directory is self-contained: it can be copied outside this repository. It
needs no files from the local manuscript, thesis, or seminar workspaces.

**Release candidate.** Publication permissions for predictions, counts and metadata,
and licensing for newly authored code, remain on the
[release checklist](RELEASE_CHECKLIST.md).

The approved vertical workflow diagram and its rendering targets come from
`bundle_v04`. The other four PDFs and the audit/profile code are unchanged.
On 2026-11-24, one relocated copy passed counts/profile replay, both figure
stages and all five final PDF hash/font checks using existing environments.
Counts and stable profile outputs are byte-identical to the v03 replay.
See [verification](provenance/README.md#vertical-workflow-refresh-v04) for
the v04 results and historical two-copy v03 checks.

## 1. Reproduce from counts

From this directory, with Python 3.9+ and a new output path:

```bash
python3 -E -S -B scripts/run_audit.py --output-dir results/audit01 --verify-maxima
```

Only the standard library is needed. No GPU, audio, annotations, weights, network,
or event matching is required. Omit `--verify-maxima` to replay the archived joint
choices without enumerating their maximum scores. Existing output directories are
never overwritten. Input/code hashes are checked against `manifest.json`.

The audit reconstructs 2,016 single-recording deletion cases and checks the frozen
comparison tables. The primary official TP-positive cohort has **255 gains / 932
eligible cases**. For the archived joint deletions, mean official score rises from
17.10 to 23.39 while recording-macro F1 falls from 33.51 to 30.38 and mean recording
recall falls from 64.22% to 41.61%. These are descriptive results for fixed outputs,
not held-out generalization or a deployable abstention rule.

Key outputs: `single_deletions.csv`, `deletion_summary.csv`, `subset_summary.csv`,
`joint_endpoints.csv`, `recording_metrics.csv`, `summary.csv`, `reconciliation.csv`,
and `audit_report.json`. The report records checks and output hashes.

## 2. Characterize gain-contributing recordings

```bash
python3 -E -S -B scripts/profile_recordings.py --output-dir results/recordings01
```

This counts-only follow-up uses the same arithmetic helpers and reconciles against
saved comparison tables. It profiles the 13 recording names behind the 255 primary
gains, with explicit eligible denominators. It needs neither raw annotations nor a
prior audit run. See [recording evidence](recording_evidence/README.md) for scope and
limitations. It makes no acoustic, ecological, or causal claim.

## 3. Optional: replay prediction matching

Obtain the exact annotation CSVs described in [data access](DATA_ACCESS.md), and
prepare the separately recorded runtime in `requirements-matching.txt`. Then:

```bash
python3 -B scripts/match_predictions.py \
  --references-root /path/to/reference_csvs --output-dir results/matching01
```

The 24 selected prediction CSVs are included and hash-pinned. This runs the
unmodified historical evaluator/matcher on original predictions and checks counts;
it does not run a model. `--limit 1` is a partial preflight, not full verification.
The pins describe an observed replay runtime, **not the training environment**.
No dependencies are installed automatically. Matching has not been rerun for this
candidate; do not describe it as newly validated.

## Figures

All five current paper PDFs are supplied in `paper_figures/`. Reproduction has two
stages; the dependency-free stage alone does **not** produce the final paper PDFs.

First, after the count audit, generate the intermediate SVG/vector drawings:

```bash
python3 -E -S -B scripts/plot_paper.py \
  --audit-dir results/audit01 --output-dir results/svg01
```

This uses only the standard library. Its three PDFs are intermediate exports;
their comparison hashes are in `expected/legacy_paper_figure_sha256.json`.
Optional `--png` needs an existing Pillow install and an invocation without `-S`.

Then use the recorded Python 3.14.4 renderer environment and the observed versions
in [requirements-plotting.txt](requirements-plotting.txt):

```bash
python3 -B scripts/polish_paper.py \
  --svg-dir results/svg01 --output-dir results/figures01
```

Nothing installs automatically. This stage uses Matplotlib, NumPy, pandas, seaborn
and Pillow, separately from the optional matching environment. It replays the
three SVG drawings with bundled Liberation Sans and regenerates the rule-check
plot from archived aggregate counts in `inputs/plotting/`. All 971 points and the
symlog transform/limits are retained; the sparse tick labels match the refreshed
export. The recurrence heatmap is copied unchanged, not redrawn from historical
plotting code. No raw annotations, audio, weights or GPU are needed.

The renderer checks SVG drawing signatures with coordinates normalized to 1e-9
point, plotting-input/font hashes, and all five final PDF hashes in
`expected/paper_figure_sha256.json`. Exact PDF bytes are tied to the recorded
renderer environment, not promised across arbitrary Python/library/font versions.
Choose new output directories for both stages.

For optional PDF hash/font inspection, use an environment containing the dependency
in [requirements-figure-checks.txt](requirements-figure-checks.txt):

```bash
python3 -B scripts/verify_paper_figures.py --figure-dir results/figures01
```

This can also inspect the supplied PDFs with `--figure-dir paper_figures`.
Rendering and font checks passed in one relocated copy; see the
[v04 runtime record](provenance/runtime-verification-v04.json). Historical
v03 and staging checks are retained separately.

## Evaluation conventions

- Eight configurations × three seeds yield 24 outputs, reusing recordings. Cases
  are dependent; seed SDs and boxplots are descriptive, not confidence intervals.
- There are 42 distinct names and 43 subset memberships. The shared PB/PB24 name
  is deleted jointly and counted once in recording averages. Shared-name and
  originally empty cases are exported but excluded from the primary prevalence.
- `released` preserves the official empty-output support-miss behavior and the
  zero-TP subset floor. `corrected` means support-consistent count accounting,
  **not a different event matcher**. See [support policy](SUPPORT_POLICY.md).
- Recording F1/recall retain fixed query references excluding five support
  events, including for deleted/empty recordings. Zero-query references are
  rejected. Do not silently drop recordings to improve an average.
- Deletion choices use labels. Joint verification proves the maximum score,
  not uniqueness of the archived set or its coverage.
- Exact-count gains and rounded-report maxima are separate. A known displayed
  maximum differs (1.896 versus 1.897) under those precision policies.
- The optional direct-evaluator patch is a reference artifact, not the historical
  replay implementation; it has not had native matching validation.

## Files and provenance

`inputs/` contains arithmetic inputs and predictions; `expected/` contains
comparison targets, not substitutes for computation. `vendor/dcase2024/` pins the
historical scorer separately from the repository's other evaluator directory.
`results/` is ignored, including annotation copies made by optional matching.

[Provenance](provenance/README.md) distinguishes unchanged staged artifacts from
public-facing adaptations, earlier staging checks, and refreshed-copy runtime results.
[Third-party notices](THIRD_PARTY_NOTICES.md) and [data access](DATA_ACCESS.md) cover
attribution and unresolved distribution checks. This audit does not establish
historical checkpoint, training-source, or full environment identity.
