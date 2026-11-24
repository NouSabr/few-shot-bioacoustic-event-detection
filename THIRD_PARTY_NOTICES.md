# Third-party provenance and release checks

This is a partial inventory of the inherited DCASE code and the CIFE/AFRI
adaptations, not a dependency-license audit or a new license grant. Removing
reference copies does not resolve the terms for retained adapted code.

## DCASE baseline

The existing [MIT notice](LICENSE), copyright Centre for Digital Music (2022),
is preserved. Baseline sources recorded in this checkout:

- [DCASE few-shot bioacoustic repository](https://github.com/c4dm/dcase-few-shot-bioacoustic)
- [Surrey DCASE 2022 implementation](https://github.com/haoheliu/DCASE_2022_Task_5),
  credited in `baselines/dcase2024_task5/README.md`.

The root notice is not evidence that every third-party adaptation below has
been cleared for distribution under MIT.

## QSFormer / CIFE

[QSFormer](https://github.com/SissiW/QSFormer), by Xixi Wang, Xiao Wang, Bo Jiang
and Bin Luo (2023), is the source of the CIFE design adapted in
[`audio_cife.py`](baselines/dcase2024_task5/src/models/components/audio_cife.py).
The audio implementation and its same-map backbone integration are not a full
QSFormer model; see the [CIFE guide](baselines/dcase2024_task5/docs/CIFE/README.md).

The 37-file reference copy under `baselines/QSFormer/` was removed on
2026-11-23 after confirming it matched local commit
`68f275be62ce64ba00c6738750f0c713cb93ecdc`. This pins a local snapshot,
not a verified upstream revision. Inspect its CIFE source and attribution:

```bash
git show 68f275be62ce:baselines/QSFormer/Models/models/CIFE.py
git show 68f275be62ce:baselines/QSFormer/README.md
```

The reference README acknowledges DeepEMD and DETR. Its `Models/models/CIFE.py`
also credits `facebookresearch/low-shot-shrink-hallucinate`. Those are provenance
leads, not a finding that every component was copied into the audio adaptation.
No standalone license file was present in the removed QSFormer snapshot.
**Release check:** establish applicable permission/notices for retained adapted
code; attribution alone is not a completed licensing review.

## AFRI

[AFRI](https://github.com/kaoshenglun/AFRI), by Sheng-Lun Kao and Yi-Wen Liu,
is the source credited by
[`afri_core.py`](baselines/dcase2024_task5/src/utils/afri_core.py).
That module identifies direct ports of the original range-estimation routines;
the training integration uses this local module, not the reference script.

The reference script was removed on 2026-11-23 after confirming it matched
local commit `763cbf6bbf98d90fc1c7655a94a299e2ff13304c`. Retrieve that original:

```bash
git show 763cbf6bbf98:baselines/AFRI/AutomaticFrequencyRangeIdentification.py
```

The AFRI documentation cites line numbers in this local snapshot, not in the
current module or an independently verified upstream revision.

The upstream README, checked on 2026-11-23, states:

> This software may be used only for non-commercial purposes, including
> education and research. For other purposes, please contact us.

It requests citation of the authors' JASA paper, DOI
[10.1121/10.0037080](https://doi.org/10.1121/10.0037080).
**Release check:** resolve applicable terms/permission for retained adapted code
before public distribution. Do not label the adaptation MIT solely because of
the repository-root notice.

## Scope still open

The reference-only directories have been removed; the local implementations
remain. Source credit and the unresolved release checks above are retained.
This cleanup is not redistribution clearance. Dataset, checkpoint, dependency
and other inherited-code terms have not been audited here.
