# Prototype visualization

[PrototypeVisualizationCallback](../../src/callbacks/prototype_visualization_callback.py)
plots training embeddings, their support/query split, and class prototypes in two
dimensions. It reuses embeddings returned by the training step rather than
running another forward pass. It has no validation-batch hook.

## Configuration

The callback is enabled in
[configs/callbacks/default.yaml](../../configs/callbacks/default.yaml), selected
by the checked-in [training config](../../configs/train.yaml):

```yaml
prototype_visualization:
  _target_: src.callbacks.prototype_visualization_callback.PrototypeVisualizationCallback
  viz_episode_freq: 1000
  reduction_method: "pca"
  n_components: 2
  save_to_disk: false
  output_dir: "visualizations/"
  use_cache: true
  figsize: [12, 8]
  dpi: 100
  log_to_wandb: true
  log_to_tensorboard: false
```

These are the checked-in config values, not historical-run settings. Direct
callback construction defaults `log_to_tensorboard` to `true`; the config above
overrides it. Experiment and local configs may override other values too.

`viz_episode_freq` counts training batches, not optimizer steps. Use a positive
integer; a larger value plots less often. To disable plotting, remove the callback
entry from the selected callback config rather than setting the frequency to zero.

## Data contract

The callback expects the training-step output to be a dictionary containing:

- `embeddings`: a tensor shaped `[samples, embedding_features]`.
- `labels`: a tensor with one label per embedding.
- `class_name`: optional class names for display.

It also reads `pl_module.hparams.train.n_shot`.
[PrototypeModule.training_step()](../../src/models/prototype_module.py) supplies
detached CPU embeddings and labels in this format.

For each label, the callback takes the first `n_shot` samples as support and the
rest as queries. If a label has no more than `n_shot` samples, all are support.
Prototypes are means of the support embeddings, computed before projection.

[PrototypeVisualizer](../../src/utils/prototype_visualizer.py) uses circles for
support, squares for queries, and stars for prototypes. Its styling assumes the
negative-contrast label convention: even labels are positive and odd labels are
the corresponding negatives, drawn with lower opacity. It is not a general label
adapter for arbitrary models or the non-contrast label convention.

The plot includes remaining samples from negative labels as queries. This differs
from the negative-contrast [training loss](../../src/utils/loss.py), which scores
only positive-label queries. See
[episode construction](../general/EPISODE_CONSTRUCTION.md) for the training split.

## Projection and checkpoint state

[DimensionalityReducer](../../src/utils/dim_reduction.py) supports `pca`,
`incremental_pca`, and `umap`. If UMAP cannot be imported, selecting it logs a
warning and falls back to PCA. Keep `n_components=2` for these scatter plots.

The visualizer combines support embeddings, query embeddings, and prototypes
before reduction:

- With `use_cache=true`, it fits the projection on the first plotted episode and
  transforms later episodes with that fitted reducer.
- With `use_cache=false`, it fits a new projection for each plot. Coordinates
  from separate plots are then not a shared projection.
- `PrototypeVisualizer.reset_reducer()` clears the fitted projection.

The callback's `state_dict()` stores only `episode_count`. Resuming from a
checkpoint can restore that counter, but does not restore the fitted projection;
the reducer is fitted again at the next plot.

## Outputs

Destinations depend on both the callback flags and the configured loggers:

| Destination | Enable with | Output |
| --- | --- | --- |
| Weights & Biases | `log_to_wandb: true` and a configured WandB logger | `train/prototype_visualization` image key |
| TensorBoard | `log_to_tensorboard: true` and a configured TensorBoard logger | `train/prototype_visualization` image tag |
| Disk | `save_to_disk: true` | `<output_dir>/episode_<six-digit-count>.png` |

`output_dir` is relative to the process working directory unless an absolute path
is supplied. The [training entry point](../../train.py) uses Hydra run directories;
do not assume a relative output directory is at the repository root.

## Troubleshooting and limits

- **No plot:** check that the callback is selected, the batch count has reached
  `viz_episode_freq`, and at least one destination is enabled and available.
- **Missing data:** the callback skips outputs without `embeddings` and `labels`.
  Check the training-step return value, tensor shapes, and `hparams.train.n_shot`.
- **Projection errors:** inspect the logged exception, sample counts, and feature
  dimensions. Selecting Incremental PCA does not make embedding collection
  streaming: the visualizer still combines all episode embeddings before fitting.
- **Slow plotting:** plot less often or disable the callback. This guide provides
  no measured runtime or memory budget and does not rank projection quality.
- **Distributed training:** the callback has no explicit rank-zero guard. DDP
  logging and file-writing behavior have not been verified here; multi-GPU
  compatibility is not guaranteed by this guide.
