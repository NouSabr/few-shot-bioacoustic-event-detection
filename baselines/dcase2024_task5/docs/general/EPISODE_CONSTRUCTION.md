# Episode construction

Training and validation sample class-based episodes. Evaluation processes one
recording at a time. This guide describes the checked-in implementation, not the
configuration or provenance of a historical run.

## Configuration and entry points

[PrototypeDataModule](../../src/datamodules/prototype_datamodule.py) selects the
datasets and constructs the loaders. Its settings come from
[the datamodule config](../../configs/datamodule/prototype.yaml) and
[the training config](../../configs/train.yaml).

The checked-in training defaults are:

| Setting | Value | Role |
| --- | --- | --- |
| `train_param.k_way` | `10` | Number of classes selected per episode |
| `train_param.n_shot` | `5` | Support samples per label |
| `train_param.seg_len` | `0.2` | Training crop length in seconds |
| `train_param.negative_train_contrast` | `true` | Return a negative crop with each positive crop |
| `train_param.use_validation_first_5` | `false` | Keep validation support annotations out of the training dataset |
| `train_param.adaptive_seg_len` | `true` | Select the adaptive evaluation loader |

Experiment and local configs can override these values. The adaptive-segment flag
selects the evaluation loader; it does not make training crop lengths adaptive.

## Training metadata and crops

[PrototypeDynamicArrayDataSet](../../src/datamodules/components/dynamic_pcen_dataset.py)
reads annotation CSVs under `path.train_dir`. `build_meta()` retains rows containing
`POS`, then expands their `Starttime` and `Endtime` by 25 ms on each side. For
annotations with a `CALL` column, the class name comes from the parent directory;
otherwise it comes from the positive label columns.

The per-class `meta` entries contain:

| Field | Contents |
| --- | --- |
| `info`, `duration` | Positive intervals in seconds and their lengths |
| `file` | Audio paths corresponding to the positive intervals |
| `neg_info` | Intervals constructed from the tracked previous end and next start |
| `neg_start_time` | End-time tracker used while building the metadata |

See `update_meta()` for the exact gap bookkeeping. These inferred intervals are
not a separate set of manually verified negative annotations.

`select_positive()` randomly chooses an annotated interval. `select_negative()`
randomly chooses a stored negative interval, retrying while its duration is below
0.2 seconds. `select_segment()` converts times to frames, crops longer intervals,
and tiles shorter ones to the requested length. Crops have shape
`[segment_frames, feature_bins]`; draws need not come from distinct events or
recordings.

For each recording, feature loading uses the configured `<stem>_<feature>.h5`
files if all are present, with `LazyHDF5Concat` for multiple feature types.
Otherwise it falls back to `FeatureLoader.extract_feature()`. When enabled,
frequency masking is applied to both positive and negative training crops.

## Episode sampling and support/query splits

[IdentityBatchSampler](../../src/datamodules/components/identity_sampler.py)
selects classes and tiles their index list. The datamodule requests `2 * n_shot`
draws per selected class. For selected classes `[0, 5, 8]` and `n_shot=2`, the
indices are interleaved:

```text
[0, 5, 8, 0, 5, 8, 0, 5, 8, 0, 5, 8]
```

With `N` selected classes and `K = n_shot`, this produces `2 * N * K` dataset
items, not a combined count of positive and negative crops. Each item returns:

- Without negative contrast: `(positive, class_id, class_name)`.
- With negative contrast: `(positive, negative, 2 * class_id, 2 * class_id + 1, class_name)`.

[PrototypeModule.step()](../../src/models/prototype_module.py) concatenates the
positive and negative batches before encoding. With negative contrast enabled,
there are therefore `4 * N * K` embeddings: `2 * K` for each positive label and
`2 * K` for its corresponding negative label.

[The prototypical losses](../../src/utils/loss.py) take the first `K` embeddings
per label as support and average them into prototypes. Without negative contrast,
the remaining samples for every label are queries. With negative contrast,
prototypes include both positive and negative labels, but only the remaining
positive-label samples are scored as queries.

## Validation episodes

[PrototypeDynamicArrayDataSetVal](../../src/datamodules/components/dynamic_pcen_dataset_val.py)
reads `path.eval_dir`. It uses the same item format and sampler repetition count,
with classes drawn from `eval_class_idxs` and no extra training-class pool.

Its metadata also tracks `neg_file`, `neg_duration`, and recording duration.
`update_meta()` resets the gap tracker for classes in each recording, clips gap
bounds, and includes a fallback that appends the recording remainder when the
checked negative-duration total is below two seconds. Consult that method for
the precise clipping and fallback conditions; training and validation metadata
construction are not identical.

## File-based evaluation

The datamodule selects the evaluation dataset using
`train_param.adaptive_seg_len`:

- `true` (checked-in default):
  [PrototypeAdaSeglenBetterNegTestSetV2](../../src/datamodules/components/test_loader_ada_seglen_better_neg_v2.py).
- `false`:
  [PrototypeTestSet](../../src/datamodules/components/test_loader.py), the
  fixed-length loader with a legacy return format.

### Adaptive loader

Each item corresponds to an annotation CSV under `path.eval_dir` or, when set,
`path.test_dir`. The loader:

1. Uses the first `n_shot` rows marked `POS`, in CSV order, as support annotations.
2. Opens the corresponding `<stem>_<feature>.h5` datasets for the configured
   feature types. Missing HDF5 files raise an error in `read_file()`.
3. Chooses segment and hop lengths from the support durations and feature settings.
4. Builds negatives from gaps before and between support annotations. If those
   gaps are too short, it can supplement them with estimated query-region
   negatives according to `eval_param.negative_estimate` (`rms` in the checked-in
   training config).
5. Builds query windows after the last support annotation, with both adaptive-
   and short-segment branches. Support crops are arrays; query windows are frame
   metadata for later streaming.

The sample has five outer fields:

```text
(data_tuple, strt_index_query, audio_path, seg_len, h5_handles)
```

`data_tuple` carries the two branches' support arrays, query metadata, hop lengths,
and negative-sampling metadata. The test loader uses `batch_size=1`,
`num_workers=0`, and
[streaming_collate_fn](../../src/datamodules/components/custom_collate.py).
[PrototypeModule](../../src/models/prototype_module.py) then constructs
[StreamingSegmentDataset](../../src/datamodules/components/streaming_segment_dataset.py)
instances to read query windows from the open HDF5 datasets.

### Fixed-length loader

`PrototypeTestSet` returns eagerly stacked positive, negative, and query arrays
in a three-field sample. Its negative windows scan the recording, not just the
region before the first positive annotation. This legacy format differs from
the adaptive loader's five-field streaming contract; check the downstream
collate and model contracts before using it as an alternative.
