# Package provenance

This package was prepared on 2026-11-23 from `bundle_v02` and selectively
refreshed from `bundle_v03` on 2026-11-24. Both bundles' original manifests and
source maps are retained as `bundle_v02_*` and `bundle_v03_*` records. Historical
source paths identify origins; public commands do not load the private workspace.

The workflow-only v04 promotion on 2026-11-24 is recorded below. The v03
descriptions and receipts are historical; their workflow hashes are not current.

The [refresh record](refresh_v03.json) identifies the selected files and unchanged
public entry points. V03's fonts, plotting inputs, figure checks, five drawing/export
helpers, and current PDFs were promoted without copying its private-workspace
`fresh_counts.py` loader or replacing the cleaned `plot_results.py` CLI. All seven
existing public scripts, recording-profile comparisons, and original audit inputs
remain byte-unchanged. Common source-map entries now identify v03; v02 snapshots
and the earlier counts-verification receipt retain their original provenance.

The package-root `SOURCE_MAP.json` maps the current files to their sources and
hashes. `manifest.json` inventories every current package file except itself;
its predecessor's hashes must not be mistaken for hashes of adapted files.

## Artifacts preserved by the workflow-only v04 refresh

- Matched counts, reference-population/membership metadata, selected predictions,
  archived deletion choices, and comparison tables from the staged bundle.
- The main count helper, audit runner, optional matching wrapper, paper-figure renderer,
  evaluator/matcher snapshots, reference patch, runtime pins, and license notices.
- The staged support-policy explanation, font notices, and four non-workflow PDFs.
  Only the approved workflow drawing/PDF and its rendering targets changed.

## Public-facing adaptations

- README, data-access instructions, release checklist, and attribution wording.
- `fresh_counts.py` retains its three used function bodies unchanged, dropping
  the unused private-workspace replay loader and its supporting helpers/imports.
- The standalone `plot_results.py` CLI now takes explicit input/output directories
  without locating a manuscript workspace. Its drawing functions are unchanged.
- `scripts/profile_recordings.py` adapts the original recording-evidence aggregation
  to bundled counts/populations and the existing arithmetic helpers. It omits raw
  annotation durations, cached acoustic metadata, and manuscript protection paths.
- Profile comparison data omit the three duration columns; subset comparison data
  are copied unchanged. Neither table replaces profile computation.

The renderer retains the historical PDF `Creator` string
`paper_revision/tooling/polish_figures.py` for the unchanged figures. The vertical
workflow uses `generate_vertical_workflow.py` to match its approved preview.
These are fixed metadata strings, not paths read or executed at runtime.

## Verification status

### Earlier staging evidence

The earlier staging record reports isolated standard-library count reproduction,
6,683 comparison checks, 2,016 deletion cases, joint-maximum verification, and
byte-identical selected figure PDFs. The preserved
[bundle v03 verification](bundle_v03_verification.json) separately reports 6,683
count checks, 24 maxima, exact approved hashes for all five PDFs, and embedded-font
checks. Its archive SHA-256 was matched to the source archive, and every archived
file hash was matched to the v03 source directory during this refresh. These are
staging results, not execution claims for this adapted public package.

### Pre-refresh public counts-only verification

On 2026-11-24, before the v03 refresh, the public counts-only commands were executed with Python
3.14.4, using `-E -S -B`, a minimal environment, and a working directory outside
the repository. Only inventoried package files were copied to two temporary
locations; neither copy could rely on the manuscript workspace through a
relative path. This was directory/interpreter isolation, not a container or
operating-system filesystem/network isolation.

- Each audit run passed 6,683 comparisons over 2,016 single-deletion cases and
  verified all 24 joint maxima. The primary cohort remained 255 gains / 932 cases.
- Each profile run passed 2,035 deletion/profile/subset comparisons, yielding
  13 names, 312 recording/output rows, 255 gaining rows, and 10 recurrent names.
  Profiles ran before the audit in each copy, without prior audit outputs.
- All 12 audit files and all nine profile files were byte-identical between
  relocated runs. Generated input/output hash manifests were checked.
- Each command refused a populated directory, an empty directory, and an
  existing file without modifying the target: six refusal checks in total.
- Package inputs, protected original sources, branch, index, and existing
  working-tree state were unchanged by execution.

The [machine-readable record](counts-verification.json) is retained unchanged.
It pins the executed scripts and scientific inputs/targets and records stable
output hashes. Those pinned files still match, but its full-package profile
manifest and repeatability claim belong to the pre-refresh inventory.

### Selective refresh checks

The selective refresh itself used static syntax, path, inventory,
source-equivalence, and link checks only. Its original [record](refresh_v03.json)
is preserved unchanged; it does not include the subsequent runtime verification.

### Historical v03 public-package runtime verification

On 2026-11-24, the refreshed package was copied to two new temporary locations
using its hash inventory. Both copies ran from an external working directory
with a minimal environment and Python 3.14.4. Count/profile and intermediate-vector
stages used `-E -S -B`; final rendering and PDF inspection used `-E -B`, existing
pinned environments and temporary caches. This was not a container, a fresh
installation, or OS-level filesystem/network isolation.

- Both audits passed 6,683 checks over 2,016 cases and all 24 joint maxima;
  the primary cohort remained 255 gains / 932 eligible cases.
- Both profiles passed 2,035 comparisons: 13 names, 312 recording/output rows,
  255 gains and 10 recurrent names. Profiles ran before either copy's audit.
- All 12 audit, nine profile, seven intermediate-vector and 10 final-figure files
  were byte-identical across relocated repeats. Output manifests were checked.
- Four final PDFs regenerated exactly; the heatmap was supplied unchanged, not
  redrawn. All five final PDFs matched their expected SHA-256 hashes. Both
  generated and supplied PDFs passed one-page and embedded-font inspection.
- The rule-check plot retained all 971 points, its point-array hashes, symlog
  transform and limits; its sparse tick labels did not overlap. SVG signatures
  passed at 1e-9-point coordinate normalization. The intermediate paired SVG
  differed from the legacy raw-byte target, as recorded; all final PDFs matched.
- Audit, profiles, intermediate drawing and final rendering each refused a
  populated directory, an empty directory and an existing file without changing
  the target: 12 refusal checks. Source copies and 145 protected workspace files
  were unchanged; branch, HEAD, index and existing working-tree state were preserved.

The [runtime record](runtime-verification-v03.json) pins code, inputs, targets,
fonts and runtime declarations and records stable output hashes. It excludes
verification documentation and inventory-dependent profile-manifest hashes to
avoid self-reference. Those manifests were checked during execution; only
documentation, the new receipt and status/inventory metadata changed afterward.
All executable/scientific artifacts remain byte-unchanged. Earlier staging and
counts receipts are retained separately, not rewritten as refreshed-copy results.

Python 3.9 syntax was checked statically; runtime checks used only Python 3.14.4,
not a version matrix. No event matching, training, inference, dependency
installation, raw-annotation/audio access or project test suite was run. Optional
native matching and the additional direct-evaluator patch remain unvalidated here.

### Notice-only release-readiness follow-up

On 2026-11-24, the renderer-distributed DejaVu notice was added for fonts already
identified in the rule-check and supplied heatmap PDFs. Official dataset
DOI/version, stated terms and an annotations-only download lead were documented;
no annotation archive was downloaded or compared. The release checklist separates
these source/notice checks from author licensing/publication decisions and
whole-repository AFRI/QSFormer clearance. Code, scientific inputs/targets, figures
and the completed runtime receipt remain byte-unchanged; this follow-up used
static inventory/source/link checks, not another runtime replay.

No source under the manuscript, defense, or model-implementation directories was
changed. Publishing this candidate, clearing data/code permissions, attaching a
release tag, and claiming public availability are separate steps.


### Vertical workflow refresh v04

On 2026-11-24, six files were promoted from `bundle_v04`: the approved
single-column workflow PDF, its drawing code and export metadata, and three
rendering target files. Public `fresh_counts.py`, `plot_results.py`, recording
profiles, audit/matching/evaluator code, predictions, counts, notices and the
other four PDFs retain their previous bytes. The source map identifies each
promoted file.

The [refresh record](refresh_v04.json) pins the prior inventory and the selected
paths. The v04 staging manifest, source map and verification are preserved as
`bundle_v04_*`; [vertical_workflow_refresh.json](vertical_workflow_refresh.json)
is an unchanged staging snapshot. Its `public_mirror_updated: false` flag
records the state before the public promotion in `refresh_v04.json`.

The [v04 runtime record](runtime-verification-v04.json) documents one relocated
public-package replay using existing environments and an external working
directory. Profiles ran independently before the audit. All 6,683 audit checks,
2,016 deletion cases, 24 joint maxima and 2,035 profile comparisons passed. All
12 audit outputs and eight stable profile outputs are byte-identical to v03;
the separately checked profile manifest reflects the new inventory.
Both figure stages passed: four PDFs regenerated exactly, the unchanged heatmap
was supplied, and all five generated/supplied hashes and embedded-font checks
passed. The other figure bytes and the 971-point rule-plot invariants are
unchanged. Each of the four output-producing commands refused its populated
output without changing it.

A wording pass after the replay shortened documentation and comments in
`plot_paper.py`. Its Python AST matches the replayed source. Runtime and
promotion receipts retain their original hashes; `SOURCE_MAP.json` and
`manifest.json` record the edited files. Workflows were not rerun for this pass.
No fresh installation, event matching, patch validation, model run, raw
annotation/audio access, project test suite, publication or permissions
clearance is claimed. Earlier receipts are retained unchanged.
