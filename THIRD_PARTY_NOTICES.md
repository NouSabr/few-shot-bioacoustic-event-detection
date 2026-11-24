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
`68f275be62ce64ba00c6738750f0c713cb93ecdc`. That local commit remains a
historical identifier. On 2026-11-24, downloaded bytes from upstream revision
[`4ce37d592e3863a9827edcba84feb4ce36cd7ff5`](https://github.com/SissiW/QSFormer/commit/4ce37d592e3863a9827edcba84feb4ce36cd7ff5)
matched these two archived files exactly:

| Upstream file | Archived Git blob |
| --- | --- |
| [`Models/models/CIFE.py`](https://github.com/SissiW/QSFormer/blob/4ce37d592e3863a9827edcba84feb4ce36cd7ff5/Models/models/CIFE.py) | `3a02e3a2d05fed12d87ce457c39d59462aa7e0b8` |
| [`README.md`](https://github.com/SissiW/QSFormer/blob/4ce37d592e3863a9827edcba84feb4ce36cd7ff5/README.md) | `36cc298926a2a10d8bf96e492ab569d2128b9e96` |

Retrieve them independently of this repository's Git history:

```bash
curl --fail --location https://raw.githubusercontent.com/SissiW/QSFormer/4ce37d592e3863a9827edcba84feb4ce36cd7ff5/Models/models/CIFE.py
curl --fail --location https://raw.githubusercontent.com/SissiW/QSFormer/4ce37d592e3863a9827edcba84feb4ce36cd7ff5/README.md
```

Pipe either command to `git hash-object --stdin` to check its Git blob ID.
The upstream comparison covers these two files only.

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
local commit `763cbf6bbf98d90fc1c7655a94a299e2ff13304c`, with Git blob
`ca190e52878fa6913c5da1e8889061b712d85c4a`. These remain historical local
identifiers. The upstream script is
[`code/0.AutomaticFrequencyRangeIdentification_linux.py`](https://github.com/kaoshenglun/AFRI/blob/897df4da18e195eb579385856e3e5ac11b7c1a53/code/0.AutomaticFrequencyRangeIdentification_linux.py)
at revision
[`897df4da18e195eb579385856e3e5ac11b7c1a53`](https://github.com/kaoshenglun/AFRI/commit/897df4da18e195eb579385856e3e5ac11b7c1a53).
Retrieve it directly:

```bash
curl --fail --location https://raw.githubusercontent.com/kaoshenglun/AFRI/897df4da18e195eb579385856e3e5ac11b7c1a53/code/0.AutomaticFrequencyRangeIdentification_linux.py
```

On 2026-11-24, the downloaded file had Git blob
`3abfe14d54aec5080701988138e4db0374461a1d`. Converting CRLF to LF, removing
one initial blank line and removing the final newline reproduced the archived
blob exactly. A byte-identical upstream snapshot has not been established.
The AFRI documentation's line numbers refer to the archived local file; the
upstream script has one extra initial blank line.

The restriction quoted in the 2026-11-23 notice is:

> This software may be used only for non-commercial purposes, including
> education and research. For other purposes, please contact us.

The [pinned upstream README](https://github.com/kaoshenglun/AFRI/blob/897df4da18e195eb579385856e3e5ac11b7c1a53/README.md#use-and-cite-our-work)
also limits use to non-commercial educational and research purposes and
requests citation of the authors' JASA paper, DOI
[10.1121/10.0037080](https://doi.org/10.1121/10.0037080).
**Release check:** resolve applicable terms/permission for retained adapted code
before public distribution. Do not label the adaptation MIT solely because of
the repository-root notice.

## Scope still open

The reference-only directories have been removed; the local implementations
remain. Source credit and the unresolved release checks above are retained.
This cleanup is not redistribution clearance. Dataset, checkpoint, dependency
and other inherited-code terms have not been audited here.
