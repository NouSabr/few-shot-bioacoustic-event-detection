# Attribution and scope of included notices

Original audit software and documentation are licensed under MIT.
`LICENSE` retains the Centre for Digital Music (2022) notice. The included DCASE
evaluator/matcher snapshots come from that MIT-licensed project; their hashes
are pinned in the bundle manifest.
Upstream baseline source/notice reference:
https://github.com/c4dm/dcase-few-shot-bioacoustic
https://github.com/c4dm/dcase-few-shot-bioacoustic/blob/main/LICENSE

The files under `vendor/dcase2024/` retain the evaluator and matcher bytes used
by the recorded replay. The generated support-consistent reference variant is
explicitly named as modified and accompanied by a unified diff. It is not called
an official evaluator release or the historical method of obtaining corrected counts.

The optional matching dependencies (NumPy, Pandas, SciPy, mir-eval, future, six)
are not vendored. They have their own licenses. Their requirements pins record an
observed replay runtime, not a claim that package versions establish historical
training equivalence. No environment was installed by this packaging task.

AFRI source is not included. Its authors' repository recommends the JASA paper
DOI 10.1121/10.0037080 and states non-commercial use conditions:
https://github.com/kaoshenglun/AFRI
Do not assume the project MIT notice relicenses that upstream implementation or
all derived training modules in the larger workspace. Publication review must cover
anything else the author decides to release.

Prediction/annotation-derived files are not automatically licensed by the code
notice. See `DATA_ACCESS.md` and the author release checklist. This document
records sources/terms for review; it does not settle redistribution rights.

## Figure font and optional plotting tools

`assets/fonts/LiberationSans-Regular.ttf` is the unchanged Liberation Sans font
supplied by bundle v03. Its copyright, SIL Open Font License 1.1 text, and
separate distribution packaging notice are preserved in
[assets/fonts/LICENSE.txt](assets/fonts/LICENSE.txt). The font is not relicensed
under the project MIT notice.

The rule-check and supplied recurrence-heatmap PDFs also embed DejaVu Sans
subsets, as recorded in `provenance/runtime-verification-v03.json`. The renderer's
unchanged [DejaVu notice](assets/fonts/DejaVu-LICENSE.txt) is included for those
fonts; it preserves the Bitstream and Arev notices and the statement about DejaVu
changes. No standalone DejaVu font file is bundled. The notice was copied from
the recorded Matplotlib 3.11.2 environment, not extracted from a historical
heatmap-font installation. Upstream notice reference:
https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/mpl-data/fonts/ttf/LICENSE_DEJAVU

Matplotlib, NumPy, pandas, seaborn, Pillow and pypdf are optional, unvendored
rendering/inspection dependencies with their own notices. Their observed
versions are recorded in `requirements-plotting.txt` and
`requirements-figure-checks.txt`, separately from native matching and training.
No dependency was installed by this refresh. Data-use approvals remain separate
from the MIT declaration for original software and documentation.

## Release boundary reviewed on 2026-11-24

This review predates the author's MIT selection for original contributions.

The upstream DCASE MIT notice, AFRI README use conditions, official dataset
record and Matplotlib DejaVu notice were checked against their primary sources.
This is a source/notice review, not a new license grant or legal clearance.
The local MIT notices and existing Liberation notice remain unchanged.

This package contains no AFRI or QSFormer training source. The parent repository
still retains AFRI adaptations and the QSFormer-derived CIFE module; its separate
release checks remain open. QSFormer source reference:
https://github.com/SissiW/QSFormer
The removed local QSFormer snapshot had no standalone license file identified
in the parent repository's provenance review. Do not infer permission from a
citation or treat this audit-package review as whole-repository clearance.

New audit wrappers, drawings and documentation still need the author's intended
license/copyright declaration. Selected predictions, counts, metadata, aggregate
plotting tables and figure outputs still need author publication approval and
appropriate data attribution. `DATA_ACCESS.md` identifies the official dataset
record without asserting that its download matches the frozen annotation hashes.


### Heatmap seed-label overlay

The replacement headers use DejaVuSans-Bold.ttf from the recorded Matplotlib
renderer; its exact font hash is pinned in `expected/heatmap_seed_labels.json`.
The existing `assets/fonts/DejaVu-LICENSE.txt` is retained. The base heatmap PDF
is the previously supplied historical drawing; only its three seed-header text
blocks are changed. No license grant for unrelated code/data is inferred.
