# Few-shot bioacoustic event detection

This independent research fork of the DCASE few-shot bioacoustic codebase
contains prototypical-network experiments with CIFE feature enhancement and
frequency-masking ablations, including energy-, activation- and AFRI-guided
masking.

The active implementation, including this project's modifications to the
inherited baseline, lives under the historical path
[`baselines/dcase2024_task5/`](baselines/dcase2024_task5/).

## Main code paths

| Path | Role |
| --- | --- |
| [`baselines/dcase2024_task5/`](baselines/dcase2024_task5/) | Research implementation, configurations, experiment runners and tests |
| [`baselines/deep_learning/`](baselines/deep_learning/) | Inherited prototypical-network baseline |
| [`baselines/cross_correlation/`](baselines/cross_correlation/) | Inherited template-matching baseline |
| [`evaluation_metrics/`](evaluation_metrics/) | Inherited challenge evaluation code |

## Before running

1. Use **Python 3.7.17** and review the implementation's
   [data/environment requirements](baselines/dcase2024_task5/README.md#data-and-environment)
   and [dependency pins](baselines/dcase2024_task5/requirements.txt).
   A clean-environment installation still requires release validation.
2. Review [data preparation](baselines/dcase2024_task5/data_preparation.py) and the
   dataset paths in the [training](baselines/dcase2024_task5/configs/train.yaml)
   and [evaluation](baselines/dcase2024_task5/configs/test.yaml) configurations.
   Replace machine-specific paths, including `path.root_dir`; evaluation also
   needs `ckpt_path` set to a compatible checkpoint.
3. Select the experiment settings explicitly. In particular, check
   `features.use_cife` and the masking options under `train_param`. Establish
   historical settings from run-specific records. Evaluation model and feature
   settings must match the checkpoint.

Run entry points from the implementation directory:

```bash
cd baselines/dcase2024_task5
```

| Entry point | Purpose |
| --- | --- |
| [`train.py`](baselines/dcase2024_task5/train.py) | Training with `configs/train.yaml` |
| [`test.py`](baselines/dcase2024_task5/test.py) | Checkpoint inference/evaluation with `configs/test.yaml` |
| [`run_ablation_experiments.py`](baselines/dcase2024_task5/run_ablation_experiments.py) | Ablation presets and run orchestration |
| [`run_full_sweep.sh`](baselines/dcase2024_task5/run_full_sweep.sh) | Multi-seed sweep orchestration |

The test suite is under
[`baselines/dcase2024_task5/tests/`](baselines/dcase2024_task5/tests/).

## Implementation documentation

- [CIFE integration](baselines/dcase2024_task5/docs/CIFE/README.md)
- [Frequency masking and AFRI](baselines/dcase2024_task5/docs/dynamic_freq_masking/README.md)
- [Implementation guide](baselines/dcase2024_task5/README.md): local code map,
  entry points and upstream references.

## Silence audit

The [silence-audit package](analysis/silence_audit/README.md) reproduces the
scoring analysis independently of model training: counts-only replay, a
recording-level evidence follow-up, and optional prediction-to-reference matching.
It includes frozen inputs, comparison tables, pinned evaluator sources, and data
access instructions. No weights or GPU are needed for counts-only reproduction.
The promoted package remains a release candidate; see its validation status and
permissions checklist before publication.

### Direct support-consistent evaluation

The audit's counts-only replay already reports both accounting policies. Its
optional matching wrapper retains the original evaluator followed by count
correction. To score a prediction CSV directly with the supplied patched
implementation, run from the repository root in the recorded matching runtime:

```bash
PYTHONPATH="$PWD/analysis/silence_audit/vendor/dcase2024" python -B - <<'PYTHON'
from pathlib import Path
from src.utils.evaluation_support_consistent import evaluate

output = Path("results/support-consistent01")
output.mkdir(parents=True, exist_ok=False)
evaluate(
    "/path/to/predictions.csv",
    "/path/to/reference_root",
    team_name="local",
    dataset="VAL",
    savepath=str(output),
)
PYTHON
```

The prediction header must be `Audiofilename,Starttime,Endtime`. The reference
root contains subset directories with CSVs containing `Starttime`, `Endtime` and a
`Q`-label column, with at least five POS support events per recording. Retain
annotation order and the five-shot support policy. The function writes a report into
the new output directory. Use the function interface above; the vendored file's
`__main__` block is a legacy demo.

The [patch file](analysis/silence_audit/patches/support-consistent-empty-output.patch)
is available for compatible external checkouts. Keep the audit's original
vendored evaluator intact: its replay checks depend on those source hashes.

## Weights and reproducibility

A release manifest still needs to identify the checkpoints for these experiments.
Existing `.pth` files require provenance checks before linking them to these runs.

Portable end-to-end reproduction with the dependency pins and sweep scripts
awaits validation. The release needs checkpoint hashes and matching configurations,
dataset/split instructions, and a validated environment and training/evaluation
procedure.

The [reproduction wrapper](baselines/dcase2024_task5/README.md#reproduction-planning)
plans all eight configurations with explicit seeds. Preparation supports opt-in
execution; training and evaluation remain plan-only.

The [artifact inventory](artifacts/README.md) defines the configuration/seed
layout and provides a static checker. Checkpoint collection and testing/audit
integration remain pending.

## Attribution

This work builds on the
[DCASE few-shot bioacoustic repository](https://github.com/c4dm/dcase-few-shot-bioacoustic)
and the [Surrey DCASE 2022 implementation](https://github.com/haoheliu/DCASE_2022_Task_5).
Refer to the upstream challenge documentation for the inherited task description,
dataset acknowledgements and challenge-paper citations. The challenge rules and
baseline scores describe those upstream versions.

## License

Original contributions in this fork are licensed under the [MIT License](LICENSE).
The inherited DCASE copyright and license notice is preserved. Third-party code,
fonts and data retain their applicable terms; see the
[third-party notices](THIRD_PARTY_NOTICES.md) for sources and outstanding checks.
