# Experiment artifacts

[`manifest.json`](manifest.json) inventories the archived cohort: eight
configurations, each with seeds **1234, 2026 and 3009**. The 24 runs have explicit
uncollected file slots. The audit's existing bundled predictions remain available
as frozen references. Checkpoints and supporting records still need recovery from
the training machine.

## Layout

```text
artifacts/
  README.md
  check.py
  manifest.json
  shared/
    environment.txt
    norm_stats.npz
  runs/
    baseline/
      seed_1234/
        selected.ckpt
        .hydra/config.yaml
        .hydra/overrides.yaml
        train.log
        evaluation.log
        selected_raw.csv
        selected_final.csv
        selection.json
      seed_2026/
      seed_3009/
    cife/
    ...
```

The configurations are `baseline`, `cife`, `phase1_random`, `slot_uniform`,
`phase2_energy`, `phase3_activation`, `afri_direct` and `afri_inverse`.
Filenames are illustrative. Preserve recovered bytes and record the actual paths.
Shared environment records, normalization statistics and AFRI profiles apply to
the complete cohort. Keep subset/run-specific records with the corresponding runs.

## Register files

Each run declares `checkpoint`, `config`, `overrides`, `training_log`,
`evaluation_log`, `selected_raw`, `selected_final` and `selection` roles. A `null`
value means that file has yet to be registered. Replace it with a record such as:

```json
{
  "path": "runs/baseline/seed_1234/selected.ckpt",
  "sha256": "REPLACE_WITH_64_LOWERCASE_HEX_DIGITS",
  "size_bytes": 123456
}
```

Replace the example size with the actual positive byte count. Paths are relative to this
directory, stay within the corresponding run directory, and must avoid symlinks.
Optional roles are `environment`, `normalization`, `afri_profiles` and
`source_snapshot`. `shared_files` accepts the first three roles with paths under
`shared/`. A file record may also contain a public HTTPS `url`; the checker makes
no network requests.

Record the original source commit when known. Keep it `null` until recovered;
the rebuilt release commit has a separate identity. Collect checkpoint-selection
criteria, epoch/metric, detection threshold and post-processing settings in
`selection.json`. The checker hashes these records; their contents and historical
relationships still need review.

The manifest pins the audit prediction-manifest hash and each run's selected-final
prediction hash. Register the original selected output under `selected_final`.
New inference outputs belong in a fresh `results/` directory for later comparison.
Keep recovered historical runs separate from reruns.

Track the manifest and checker in Git. Local `runs/` and `shared/` payloads are
ignored. Approved checkpoints and large logs can be distributed through versioned
archives; publication permissions, licensing and download locations remain pending.

## Check the inventory

From the repository root:

```bash
python3 -E -S -B artifacts/check.py
python3 -E -S -B artifacts/check.py --json
python3 -E -S -B artifacts/check.py --require-complete
```

Use `--artifacts-root /path/to/artifacts` for a separate collection directory.
`--audit-manifest /path/to/prediction_manifest.csv` supports relocation; the file
must match the pinned hash and cover the same 24 runs.

The checker validates the schema, cohort, paths, sizes, SHA-256 hashes and
selected-final correspondence to the audit. It reports uncollected slots, missing
registered files, unregistered files under `runs/` and `shared/`, and unknown
source commits. It reads payloads as bytes and leaves files unchanged.

Exit codes: `0` for a valid inventory, `1` for invalid metadata, unsafe files or
hash/size mismatches, `2` for an incomplete inventory with `--require-complete`.
Inventory completeness covers declared file slots. Model compatibility, source
provenance, environment coverage and checkpoint-to-prediction relationships need
separate validation. Training and inference retain Python **3.7.17**.

The initial 24-run inventory passed its metadata checks on Python 3.14.4;
strict mode returned `2` for the 192 uncollected slots.

## Remaining work

- Validate the checker under Python 3.7.17.
- Recover and register the files from the training machine, then compare their
  selected predictions against the audit's frozen hashes.
- Feed checkpoint/config bindings into the existing reproduction planner and add
  optional artifact input to audit matching. The audit keeps its default bundled
  replay. Exact external annotations and the recorded runtime remain necessary
  for matching.
- When runs are authorized, validate the research environment and preprocessing,
  then checkpoint inference for one run before the complete cohort. Training and
  evaluation execution in the wrapper remains pending.
