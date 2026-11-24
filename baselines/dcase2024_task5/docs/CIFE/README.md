# CIFE audio integration

This guide describes the current checkout, not the image-based reference model
or the exact source/configuration used by historical experiments.

## Wiring

The implementation is split between [AudioCIFE](../../src/models/components/audio_cife.py)
and its [ResNet call site](../../src/models/components/protonet.py):

1. The backbone produces the layer-2 feature map.
2. If spatial pooling is enabled, it pools that map before attention.
3. AudioCIFE receives the same map as both source and target; this call does not
   fuse two different backbone scales.
4. Pooled output is interpolated back to the original spatial size.
5. The backbone adds the output to the original layer-2 map, scaled by
   `features.cife_alpha`, before passing it to layer 3.

AudioCIFE also has its own learned blend with its target input. This internal
blend and the backbone's outer residual are separate operations. The backbone
retains its original layer-3 path if CIFE processing raises an exception.

## Configuration

Use [train.yaml](../../configs/train.yaml) as the configuration reference.
It currently enables CIFE and spatial pooling, with pool size `[2, 16]`.
Constructor fallbacks are not the same thing as checked-in YAML defaults.

The [ablation runner](../../run_ablation_experiments.py) overrides this setting:
its `baseline` experiment disables CIFE and its `cife` experiment enables it.
Do not infer a historical experiment's settings from today's defaults.

For an explicitly selected training variant, run from the implementation directory:

```bash
cd baselines/dcase2024_task5
python train.py path.root_dir=/path/to/dataset features.use_cife=false
# Or enable it explicitly:
python train.py path.root_dir=/path/to/dataset features.use_cife=true
```

These are command examples, not a verified environment or training replay.

## Checkpoints and tests

Do not assume that baseline and CIFE-enabled checkpoints are interchangeable.
The backbone's `load_weight` method calls `load_state_dict` without `strict=False`;
the automatic compatibility examples in the retired guides were not the current
implementation. Check the actual artifact and feature settings before use.

See the [existing CIFE tests](../../tests/README_CIFE_TESTS.md) for intended
coverage. No tests, checkpoints, memory benchmarks, or training runs were executed
as part of this documentation cleanup.

## Background

- [Audio backbone walkthrough](Audio_ResNet_Explanation.md): illustrative configurations,
  not the active experiment configuration.
- [Image-model CIFE walkthrough](CIFE_ResNet_Explanation.md): reference architecture,
  not the audio integration described above.
