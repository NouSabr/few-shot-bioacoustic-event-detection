# Frequency Masking — Overview

**Status:** current — describes what the code does, not what was planned.

This directory used to hold a forward-looking implementation plan with predicted
outcomes. The work is done and the results did not match those predictions, so
the planning documents have been removed or marked. What follows is the
implemented system.

---

## The five strategies

Selected by `train_param.freq_mask_mode`. They are **independent siblings** —
choosing one ignores the others' parameters entirely. All live in
[`src/datamodules/components/frequency_masker.py`](../../src/datamodules/components/frequency_masker.py).

| `freq_mask_mode` | Class | Signal used | Per class | Doc |
|---|---|---|---|---|
| `random` | `RandomFrequencyMasker` | none (SpecAugment) | – | [masker source](../../src/datamodules/components/frequency_masker.py) |
| `slot_uniform` | `UniformSlotFrequencyMasker` | none (the control) | – | this file |
| `energy` | `EnergyBasedFrequencyMasker` | mean spectrogram energy | yes | [masker source](../../src/datamodules/components/frequency_masker.py) |
| `activation` | `ActivationGuidedFrequencyMasker` | encoder layer-2 activations | yes | [masker source](../../src/datamodules/components/frequency_masker.py) |
| `afri` | `AFRIRangeMasker` | positive-minus-negative contrast | yes | [08](08_AFRI_INTEGRATION.md) |

The three guided strategies all derive their signal from **the same first
`afri_n_samples` positive segments of each class**, so the progression varies only
the signal, not the information available to it. Energy and AFRI ranges are
properties of the data and are precomputed and cached; activation importance is
the only signal tracked online, because it is the only one that changes as the
encoder learns.

Masking is applied at **training time only**, to both the positive and the
negative segment of each episode, in
[`PrototypeDynamicArrayDataSet.__getitem__`](../../src/datamodules/components/dynamic_pcen_dataset.py).
Neither the validation nor the test dataset has a masker, so nothing leaks into
evaluation.

### Slot geometry (slot_uniform / energy / activation)

The slot strategies partition the 128 mel bins into
`ceil(n_mels / max_band_width)` contiguous **slots** and draw `n_bands` of them
without replacement from a slot distribution P(b). They differ only in P(b):

```
slot_uniform  P(b) uniform                                       (the control)
energy        P(b) proportional to 1 - normalised_energy(b)      (spare loud slots)
activation    P(b) proportional to normalised_importance(b)      (hit used slots)
random        no slots at all - free start positions
```

`slot_uniform` is the baseline the guided strategies should be measured against,
**not** `random`: `random` differs from them in the placement rule as well as the
guidance, and that difference is not neutral.

Two consequences worth knowing before comparing them:

- At the ablation settings (`max_band_width=24`, `n_mels=128`) there are
  **6 slots and the last is only 8 bins wide**, so a slot is not a uniform unit.
- `random` samples free start positions rather than slots, which makes its
  coverage strongly non-uniform across frequency — bin 0 is masked at 0.025
  against 0.30 mid-spectrum.
- the slot strategies have their own non-uniformity: per-bin coverage ripples with
  period `max_band_width`, dipping to 0.083 at each slot boundary against 0.374
  at each slot centre.

Both are measured in [09_WIRING_VERIFICATION.md](09_WIRING_VERIFICATION.md).

### AFRI has no slots

`afri` masks a fixed area strictly inside or strictly outside the frequency
range AFRI identified for the class. It is a different geometry, not a fourth
P(b). See [08_AFRI_INTEGRATION.md](08_AFRI_INTEGRATION.md).

---

## Quick start

```yaml
train_param:
  use_freq_masking: true
  freq_mask_mode: 'random'   # or slot_uniform | energy | activation | afri
  freq_mask_n_bands: 4
  freq_mask_prob: 0.8
  freq_mask_max_width: 24
  freq_mask_seed: 1234
```

Full config surface: [05_HYPERPARAMETERS.md](05_HYPERPARAMETERS.md).

The eight-configuration ablation is driven by
[`run_ablation_experiments.py`](../../run_ablation_experiments.py).

---

## Verifying the wiring

Three layers, cheapest first. Run all three before launching the ablation.

```bash
# 1. Are the configs sane?  seconds, no data needed
python tools/sanity_check_configs.py

# 2. Does each strategy steer the mask as intended?  seconds, no data needed
python tools/verify_masking_wiring.py

# 3. Unit tests
pytest tests/unit/test_phase1_frequency_masking.py \
       tests/unit/test_phase2_frequency_masking.py \
       tests/unit/test_phase3_frequency_masking.py \
       tests/unit/test_afri_frequency_masking.py -v

# 4. Does each config actually run?  minutes, needs the data
python run_ablation_experiments.py --fast-dev-run
```

**`tools/sanity_check_configs.py`** reads the experiment list from
`run_ablation_experiments.py`, applies each set of Hydra overrides, and checks
that the config produces the intended masking: the mode builds the expected
class, every override key is one something reads, per-class profiles have an
attachment point and change the mask once attached, `afri` has `use_afri=true`,
and — the check that matters most — **no two masking configs behave
identically**. Eight runs is a long time to discover otherwise.

**`tools/verify_masking_wiring.py`** measures per-bin mask rates for all five
strategies and asserts them against recorded reference values, including that
each guided strategy gives different classes different masks. Needs only numpy
and `.afri_cache/profiles.npz` — no model, dataset or GPU.

**`--fast-dev-run`** runs one train/val batch per configuration, which covers
what the static checks cannot reach: real dataset construction, energy/AFRI
profile computation or cache load, the tracker → `training_step` → masker path,
HDF5 reads, collate and the forward pass.

Stage timings (setup, training, inference, and seconds per test recording) are
written to `logs/timings/<exp_name>.json` and tabulated at the end of an ablation
run.

## Known limits

Read [09_WIRING_VERIFICATION.md](09_WIRING_VERIFICATION.md) before drawing
conclusions from the ablation. In summary:

1. Slot 5 is 8 bins wide rather than 24, so a slot is not a uniform unit.
2. Per-bin coverage ripples at slot boundaries, independent of any guidance.
3. AFRI's per-class masked area ranges from 1 to 44 bins, so a few classes are
   trained nearly unaugmented.
4. Three of 46 classes have invalid AFRI profiles and fall back to unguided
   masking.

The runs in `paper/run_logs/` predate the per-class rework and used the
superseded designs described in that document; they must be repeated.
