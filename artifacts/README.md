# Experiment artifacts

Plan for collecting the checkpoints and run records used in the reported
experiments. The manifest, payload directories and workflow integration are
pending. Training and inference use Python 3.7.17; execution remains deferred.

## Layout

Each configuration has seeds **1234, 2026 and 3009**, giving 24 run directories.
The configurations are `baseline`, `cife`, `phase1_random`, `slot_uniform`,
`phase2_energy`, `phase3_activation`, `afri_direct` and `afri_inverse`.

```text
artifacts/
  README.md
  manifest.json
  shared/
    environment.txt
    norm_stats.npz
  runs/
    baseline/
      seed_1234/
        selected.ckpt
        .hydra/
          config.yaml
          overrides.yaml
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

Filenames are illustrative. Preserve recovered file contents and record their
actual relative paths. Store files under `shared/` only when their provenance
establishes that they apply across those runs. Keep run-specific environment
records and AFRI profiles with their run. Missing files remain explicit.

## Manifest

Use one artifact manifest to identify the archived cohort and its 24 runs. Record:

- Configuration, seed, file paths, sizes and SHA-256 hashes.
- Original source commit or snapshot, saved Hydra configuration and environment.
- Checkpoint-selection rule, epoch/metric and supporting run evidence.
- Prediction stage, detection threshold and post-processing settings.
- The selected prediction's correspondence to the silence-audit manifest.
- Missing files, unresolved checkpoint-to-prediction relationships and verification
  results, with the evidence for each result.
- Versioned download locations when approved artifacts are published.

Hashes identify file contents. Checkpoint/config compatibility and the relationship
between a checkpoint and archived predictions require separate verification.
Preserve original source identities and historical receipts when rebuilding Git
history. Keep recovered historical runs distinct from later reruns.

## Workflow integration

The testing planner will read verified checkpoint/config bindings through an
adapter that supplies its existing reproduction-manifest fields. Write regenerated
predictions and logs to a fresh `results/` directory. The current wrapper executes
preparation only; training/evaluation execution and Hydra composition still need
implementation or validation.

The silence audit keeps its bundled inputs and default replay. Add an optional
artifact-manifest input to prediction matching. Historical reproduction must
verify external predictions against the audit's frozen prediction hashes. Compare
new inference outputs separately, preserving the frozen targets. Counts-only
replay remains independent of checkpoints. Exact external annotations and the
recorded matching runtime are still required for matching.

## Storage and publication

Track this README and the eventual manifest in Git. Before collecting payloads,
add ignore rules for the local run/shared files. Publish approved checkpoints and
large logs as versioned downloadable archives. Keep the audit's bundled inputs
self-contained and connect their prediction copies to the registry through hashes.
Data-use permissions, licensing and release identity require their own checks.

## Work order

1. Record this plan, then rebuild the clean history from `main` on a new branch.
   Pin the source tip, preserve `cleanup-public-release`, and use a separate
   worktree to leave local files untouched. Split the final diff into logical
   commits and verify the resulting tracked tree matches the pinned source.
2. Implement the artifact manifest, ignore rules and static inventory checks as a
   separate batch after the history rebuild.
3. Collect checkpoints, saved Hydra files, logs, predictions, environment records,
   source identities, normalization statistics and any AFRI profiles from the
   training machine. Compare recovered predictions with the audit's hashes.
4. Connect the testing planner and optional audit input. Keep model/scoring behavior
   and the existing directory hierarchy unchanged.
5. When runs are authorized, validate the Python 3.7.17 environment and real-data
   preparation, then checkpoint loading/inference for one run before all 24.
   Verify thresholds, post-processing and score comparisons. Full retraining is
   a later validation step.

Artifact recovery can proceed when the other machine is available. The history
rebuild can proceed with missing artifacts documented.
