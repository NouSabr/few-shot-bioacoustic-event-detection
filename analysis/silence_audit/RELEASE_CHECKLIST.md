# Release checklist

This checklist tracks publication permissions, licensing, verification and
release identity. The package is prepared in the cleanup working tree;
publication, clean-install validation and availability updates remain pending.

Current figure-text candidate: plots/captions have been simplified without
changing numerical inputs or plotted observations. The heatmap's seed headers
are updated; its historical matrix is retained. See
`provenance/figure_text_cleanup.json` and `provenance/runtime-verification-v05.json`.
Earlier v03/v04 runtime records below remain historical. This refresh does not
clear any outstanding author, data, code-license, URL/tag or publication decision.

Before committing/publishing the package:

- [ ] Approve the selected prediction, count, profile, and recording-metadata files
      for publication, including their data-use terms.
- [x] Identify the official annotation access route and stated dataset terms;
      DOI/version, annotations-only archive and advertised checksum are in `DATA_ACCESS.md`.
- [ ] Verify downloaded annotation bytes against all frozen reference SHA-256 hashes
      before optional matching. No archive was downloaded in the readiness review;
      raw annotations remain external.
- [ ] Review code attribution/use terms, including the newly authored wrappers and
      plots. The existing project MIT notice does not relicense unrelated work.
- [x] Inspect the current package inventory/source map for scope and hash coverage;
      public data/code approval remains separate.
- [x] Historical v03 counts-only audit and profiles (2026-11-24):
      two relocated copies, 6,683 audit / 2,035 profile checks, all 24 joint maxima,
      byte-identical outputs and existing-output refusal.
- [x] Check aggregate plotting-table provenance and font notices. Liberation's
      notice is retained; the renderer-distributed DejaVu notice is now included
      for the fonts embedded in the rule-check and supplied heatmap PDFs.
- [x] Historical v03: both figure stages and final five-PDF hash/font inspection
      from two relocated copies (2026-11-24), using the recorded existing runtimes:
      four PDFs regenerated, heatmap supplied unchanged; outputs byte-identical.
- [x] Promote the approved vertical workflow from v04 and replay one relocated
      public-package copy (2026-11-24): 6,683 audit / 2,035 profile checks,
      24 maxima, four regenerated PDFs and five-PDF hash/font inspection.
      Counts, stable profiles and the other four PDFs retain their v03 bytes;
      see `provenance/runtime-verification-v04.json`.
- [ ] If promising prediction-to-count reproduction, approve/run optional matching
      against exact annotations and inspect all 24 outputs' comparisons.
- [ ] Validate the additional direct-evaluator reference patch separately before
      asserting fresh-matching equivalence. It was not the historical replay path.
- [ ] Confirm the intended public URL, immutable commit/tag, and citation metadata.
- [ ] Update availability wording only after the release actually exists.

The [provenance records](provenance/README.md) distinguish staging checks,
historical v03 replays and the v04 workflow refresh. Replays used existing
environments. No training, inference, matching, dependency installation or
project test suite was run. Publication permissions remain unresolved.

## Historical notice-only release-readiness review — 2026-11-24

Documentation/notice review only; no new audit/profile/figure execution, matching,
annotation/audio download, dependency installation or model run. The prior runtime
receipt is unchanged. Current scripts, scientific inputs, targets and PDFs retain
their verified hashes; the added notice and documentation are inventoried statically.

| Remaining decision | Evidence / required action |
| --- | --- |
| License for new audit code/docs | Confirm the intended license and copyright holders; do not infer a new grant from the inherited DCASE notice. |
| Publish selected derived artifacts | Approve predictions, counts, profiles, metadata, aggregate tables and PDFs; record their distribution/attribution terms. |
| Exact annotation access | Official v1 annotations-only ZIP identified; compare its CSV bytes to the frozen hashes before claiming availability of exact references. |
| Matching claim | Run optional matching only after separate approval and exact-reference/runtime preflight; otherwise describe this release as verified counts/figure replay. |
| Release identity | Confirm repository URL, authors/citation metadata and immutable commit/tag; publish only after explicit authorization. |
| Whole-repository scope | AFRI and QSFormer/CIFE adaptations outside this package still need separate terms review; this package does not clear them. |

Excluded: raw audio/annotations, checkpoints, training/AFRI source, manuscripts,
private reviews, seminar notes, environments, caches, credentials, and git metadata.
Review the larger repository independently; its retained AFRI and QSFormer/CIFE adaptations have
unresolved distribution terms and are not cleared by this focused audit package.
