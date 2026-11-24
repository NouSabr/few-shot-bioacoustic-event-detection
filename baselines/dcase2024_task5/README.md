# Bioacoustic detection research implementation

This tree contains the project's CIFE and frequency-masking changes to inherited
DCASE/Surrey code. The directory retains its historical name. See the
[repository overview](../../README.md) for setup prerequisites, checkpoint
availability and reproduction status.

## Code map

| Path | Contents |
| --- | --- |
| [`src/models/`](src/models/) | Prototypical-network model and CIFE components |
| [`src/datamodules/`](src/datamodules/) | Audio loading, episodic sampling and frequency maskers |
| [`src/utils/`](src/utils/) | AFRI range estimation, frequency-importance tracking and experiment utilities |
| [`configs/`](configs/) | Hydra configurations |
| [`tests/`](tests/) | Test suite |

## Entry points

Run from this directory after reviewing the environment and data requirements in
the [root README](../../README.md#before-running).

| Script | Purpose |
| --- | --- |
| [`train.py`](train.py) | Training with [train.yaml](configs/train.yaml) |
| [`test.py`](test.py) | Checkpoint inference/evaluation with [test.yaml](configs/test.yaml) |
| [`run_ablation_experiments.py`](run_ablation_experiments.py) | Ablation presets and run orchestration |
| [`run_full_sweep.sh`](run_full_sweep.sh) | Multi-seed sweep orchestration |

Set dataset paths and evaluation `ckpt_path` for your environment. Choose CIFE
and masking settings explicitly; evaluation model/feature settings must match
the checkpoint. Use run-specific records to establish historical settings.

## Data and environment

### Runtime

Use **Python 3.7.17** for preparation, training and inference. The
[dependency pins](requirements.txt) include `torch==1.11.0+cu113`,
`torchaudio==0.11.0+cu113` and `pytorch-lightning==1.6.0`. A clean installation,
package imports and CUDA/native-library compatibility still need validation.
Retain the working environment export and driver/GPU details with recovered runs.

### Dataset layout

Set the reproduction manifest's `dataset_root` to the directory containing
`Development_Set`:

```text
<dataset_root>/
  Development_Set/
    Training_Set/
      <subset-or-class>/
        recording.wav
        recording.csv
    Validation_Set/
      <subset>/
        recording.wav
        recording.csv
```

Keep WAV/CSV basenames paired and retain the original subset/class directories,
annotation bytes and row order. Use a fresh writable data copy for preparation;
features and statistics are written beside these inputs.

The [data-access notes](../../analysis/silence_audit/DATA_ACCESS.md) identify
published dataset records and the exact annotation hashes required by optional
audit matching. The original training archive/version and raw-input hashes remain
unverified. Record the downloaded archive/version and checksums before claiming a
historical training replay.

The wrapper plans evaluation on `Validation_Set` and sets `path.test_dir=null`.
Preparation also processes `Development_Set/Test_Set` when present; its preflight
requires paired CSVs there. The direct YAML defaults point `path.test_dir` at
`<dataset_root>/Test_Set`. Held-out test preparation/evaluation needs a separate
validated procedure.

### Stored features

Historical preparation used `data_preparation.py`; its producing commit remains
unpinned. The checked-in [script](data_preparation.py) uses its own hard-coded
extractor; changing Hydra's `features.*` values leaves preparation unchanged.

| Extraction setting | Value |
| --- | --- |
| Sample rate | 22,050 Hz |
| FFT length / hop | 1,024 / 256 samples |
| Mel bins / upper frequency | 128 / 11,025 Hz |
| Waveform scaling | Divide by the maximum absolute sample |
| Log transform | Natural logarithm of the mel spectrogram plus `1e-8` |

The extractor omits a `fmin` argument, while the YAML files declare
`features.fmin: 50`. Their `features.eps` values also differ from the extractor's
log offset. Recover the original script/config records to bind these settings to
the archived features; these differences are left unchanged.

The wrapper requests HDF5 output. Preparation writes each recording's
`<stem>_logmel.h5`, with the feature array under `data`. It computes one scalar
mean and population standard deviation from available training log-mel values using
Welford's algorithm, saves `logmel_mean` and `logmel_std` in
`Development_Set/norm_stats.npz`, and normalizes all processed splits with those
training statistics. Retain the statistics with their dataset/preparation identity.

The default [FeatureLoader](src/datamodules/components/pcen.py) prefers HDF5 over
NPY and uses stored features with runtime normalization skipped. Review the
[guarded preparation procedure](#preparation) before execution. Real-data feature
contents and normalization remain unvalidated.

### AFRI feature provenance

The author recalls using mel features for AFRI mask production and an additional
preparation pass that produced mel and log-mel. That pass's command, script
revision and normalization choices still need recovery. The checked-in extractor
computes mel internally, then discards it before writing; the current wrapper
only tracks log-mel outputs.

The [AFRI notebook](notebooks/visualize_afri_profiles.ipynb) was used for
analysis/profile creation. Its saved run loads an existing cache under a `logmel`
configuration; the inputs used to create that cache remain unverified.

The training AFRI profile builder reads the dataset's configured feature arrays
(`features.feature_types`, currently `logmel`) on a cache miss. It can load an
existing profile cache first. Its cache key includes sample rate, hop and mel-bin
count but omits feature type and normalization identity, so the key alone cannot
establish which inputs produced the archived profiles.

Recover the mel-generation procedure, whether those arrays were raw or normalized,
and the training cache's `metadata.json` and `profiles.npz`. The separate evaluation
frequency-mask path also reads `*_mel.h5`/`*_mel.npy` and external `mask_<key>` files;
its inputs need their own provenance if that path was used. AFRI recomputation
remains unvalidated until the historical input path is resolved.

## Reproduction planning

[`tools/reproduce.py`](tools/reproduce.py) plans the eight existing presets across
seeds **1234, 2026, 3009**: 24 runs. Only preparation supports execution; training
and evaluation are plan-only. Plans cover the training and validation splits;
held-out evaluation remains outside this workflow. One preparation stage serves
the complete run grid.

Training, inference and evaluation use Python 3.7.17. The wrapper targets the same
runtime. Set the research interpreter path in the manifest. Preview the
[example manifest](configs/reproduction.example.json):

```bash
python3 -E -S -B tools/reproduce.py \
  --config configs/reproduction.example.json --dry-run
```

Copy it to `configs/reproduction.local.json` (Git-ignored), set the interpreter,
dataset and output paths, then check preparation:

```bash
python3 -E -S -B tools/reproduce.py \
  --config configs/reproduction.local.json --stage prepare --preflight
```

Manifest paths are relative to the manifest; `--report-dir` is relative to the
calling directory. Seeds must be positive. Shared overrides cannot replace
preset choices, seeds or managed paths. Missing checkpoints leave blocked
evaluation entries in the plan.

Preflight checks package-version metadata, WAV/CSV pairing, required files and
output collisions. It does not validate package imports, CUDA, Hydra composition
or checkpoint compatibility. Real-data preparation and model runs remain unvalidated.

### Preparation

**This writes beside the dataset. Use a fresh working data copy.** After reviewing
preflight, execution requires an explicit stage and a new report directory outside
the dataset:

```bash
python3 -E -S -B tools/reproduce.py \
  --config configs/reproduction.local.json --stage prepare --execute \
  --report-dir results/prepare01
```

The existing script writes HDF5 features and training-derived statistics in
`Development_Set/norm_stats.npz`. It also processes `Development_Set/Test_Set` if
present. Existing outputs, unsafe symlinks and unsupported audio paths are refused.
Nothing is installed.

`Development_Set/.reproduction-prepare.json` blocks repeat attempts once execution
has claimed the dataset. The claim and partial outputs are not removed automatically.
Review the receipt before manual cleanup or retry. There is no rollback, and other
programs can still modify the data.

Reports contain the command/arguments, input manifest and source snapshot, input
file metadata, stdout/stderr, exit status and artifact hashes. Expected HDF5/NPZ
outputs must be regular, nonempty files; formats and normalization are not validated.
Raw inputs are not content-hashed, and partial outputs may have `sha256: null`.

Exit codes: `0` success, `1` failed preflight/execution/output checks, `2` invalid
input or report-path refusal, `130` interrupted execution. On POSIX, interrupting
the run stops the child process group.

### Checkpoints and logged settings

The [evidence record](configs/reproduction.evidence.json) confirms the seeds,
preset overrides and CIFE settings from the 24 cached headers. The example pins
**4 heads and pooling `[2,16]`**, overriding `test.yaml`'s conflicting defaults.
The planner checks the record's hash, cohort coverage and preset agreement without
needing the cached logs. Archived operating points appear as report metadata
and are excluded from command generation.

The record covers selected printed fields. Full historical configurations and
checkpoint identity remain unverified. Extraction reads at most 390 lines /
512 KiB per log, computes prefix hashes, and preserves archive-supplied full-log
hashes. Producer/helper hashes identify the extraction snapshot. With the local
cache available, regenerate to a new file:

```bash
python3 -E -S -B tools/extract_reproduction_evidence.py \
  --output results/config-evidence01.json
```

For a new cohort, omit `logged_evidence`. Existing records retain their original
cohort identities.
For later evaluation, register checkpoints explicitly under keys such as
`baseline_seed1234`; the wrapper never selects one by filename:

```json
{
  "path": "/path/to/selected.ckpt",
  "sha256": "REPLACE_WITH_CHECKPOINT_SHA256",
  "config_path": "/path/to/run/.hydra/config.yaml",
  "config_sha256": "REPLACE_WITH_SAVED_CONFIG_SHA256"
}
```

Replace both hashes with lowercase 64-digit digests to identify the file contents.
Model compatibility requires a separate check. The [silence audit](../../analysis/silence_audit/README.md)
is a separate workflow.

### Wrapper tests

```bash
python3 -E -S -B tests/unit/test_reproduction_prepare.py
```

The tests use temporary dummy programs, file fixtures and Python's standard library.

## Documentation

- [CIFE integration](docs/CIFE/README.md)
- [Frequency masking and AFRI](docs/dynamic_freq_masking/README.md)
- Background: [episode construction](docs/general/EPISODE_CONSTRUCTION.md) and
  [prototype visualization](docs/visualization/PROTOTYPE_VISUALIZATION_GUIDE.md).

## Upstream work

The inherited implementation builds on the
[DCASE bioacoustic repository](https://github.com/c4dm/dcase-few-shot-bioacoustic)
and [Haohe Liu's Surrey implementation](https://github.com/haoheliu/DCASE_2022_Task_5).
Citations for the upstream work:

```bibtex
@misc{liang2024mind,
      title={Mind the Domain Gap: a Systematic Analysis on Bioacoustic Sound Event Detection}, 
      author={Jinhua Liang and Ines Nolasco and Burooj Ghani and Huy Phan and Emmanouil Benetos and Dan Stowell},
      year={2024},
      eprint={2403.18638},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}

@techreport{Liu2022a,
    Author = "Liu, Haohe and Liu, Xubo and Mei, Xinhao and Kong, Qiuqiang and Wang, Wenwu and Plumbley, Mark D",
    title = "SURREY SYSTEM FOR DCASE 2022 TASK 5 : FEW-SHOT BIOACOUSTIC EVENT DETECTION WITH SEGMENT-LEVEL METRIC LEARNING",
    institution = "DCASE2022 Challenge Technical Report",
    year = "2022"
}
```

See the [existing license notice](../../LICENSE) and
[third-party provenance and release checks](../../THIRD_PARTY_NOTICES.md) for
CIFE/AFRI sources and unresolved distribution terms.
