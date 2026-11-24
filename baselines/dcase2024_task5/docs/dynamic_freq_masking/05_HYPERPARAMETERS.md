# Hyperparameters — frequency masking

**Status:** current. Every key below is read by the code; the previous version of
this document listed several that were not.

All keys live under `train_param` in `configs/train.yaml`.

---

## Shared by all five strategies

| Key | Default | Meaning |
|---|---|---|
| `use_freq_masking` | `false` | Master switch. |
| `freq_mask_mode` | `'random'` | `random` \| `slot_uniform` \| `energy` \| `activation` \| `afri`. |
| `freq_mask_n_bands` | `2` | Bands masked per spectrogram. |
| `freq_mask_prob` | `0.5` | Probability of masking a given segment. |
| `freq_mask_max_width` | `16` | Maximum width of one band, in mel bins. Also sets the slot width for every slot strategy. |
| `freq_mask_seed` | `1234` | Base seed for the masking RNG, separate from the global seed so masking is independent of episode sampling. Specialised per dataloader worker and epoch when `num_workers > 0`. |

`freq_mask_max_width` does double duty for the slot strategies: slot count is
`ceil(n_mels / freq_mask_max_width)`. At `n_mels=128`, a value of 24 gives 6
slots of which the last is only 8 bins wide. Per-bin coverage also ripples with
period `freq_mask_max_width`, dipping at every slot boundary — see
[09_WIRING_VERIFICATION.md](09_WIRING_VERIFICATION.md).

## `slot_uniform` (control)

No parameters of its own beyond the common ones. It is the slot sampler with
P(b) uniform, and it is the baseline the three guided strategies should be
compared against — not `random`, which differs from them in the placement rule
as well as the guidance.

## `energy` (Phase 2)

| Key | Default | Meaning |
|---|---|---|
| `freq_mask_energy_weight` | `1.0` | Guidance strength. `0.0` = uniform slots, `1.0` = pure energy guidance (the class's loudest slot is then never masked, since its weight is exactly 0). |

**There is no energy EMA.** The profile is per class and precomputed from the
same first `afri_n_samples` positive segments AFRI uses, because spectrogram
energy is a property of the data — nothing about it changes during training.
Cached in `.afri_cache/energy_profiles.npz`, keyed on the CSV mtimes, the sample
count and the feature config.

`freq_mask_energy_ema_alpha` is gone. See
[09_WIRING_VERIFICATION.md](09_WIRING_VERIFICATION.md) for why the online EMA
was wrong.

## `activation` (Phase 3)

| Key | Default | Meaning |
|---|---|---|
| `freq_mask_importance_weight` | `1.0` | Guidance strength. `0.0` = uniform slots, `1.0` = pure importance guidance. |
| `freq_mask_importance_ema_alpha` | `0.9` | Momentum for the **per-class** activation EMA, in `FrequencyImportanceTracker`. Convention `ema = alpha*old + (1-alpha)*new`, so higher is more stable. |

Importance is the only masking signal tracked online, because it is the only one
that changes as the encoder learns. It is updated in `training_step`, where the
class labels are known — a forward hook alone is label-blind and could only
produce one global profile shared by every class.

Phase 3 no longer takes `freq_mask_energy_weight`; the energy blend was never
used and made Phase 3 compute an energy EMA it discarded.

## `afri`

| Key | Default | Meaning |
|---|---|---|
| `use_afri` | `false` | Compute/load the per-class profiles. Must be `true`, or every class falls back to unguided slot masking. |
| `afri_mode` | `'direct'` | `direct` masks inside the class range, `inverse` outside it. |
| `freq_mask_afri_area_budget` | `44` | Upper bound on masked bins. Actual area is `min(budget, W, n_mels - W)` for range width `W`. |
| `afri_cache_enabled` | `true` | Cache profiles to `{train_dir}/.afri_cache`. |
| `afri_cache_dir` | `null` | Override the cache location. |
| `afri_force_recompute` | `false` | Recompute even if a valid cache exists (~2 hours). |
| `afri_n_samples` | `5` | Positives per class to profile. Must be `<= n_shot`. |
| `afri_value_threshold` | `0.23` | Peak-detection threshold in the range search. |
| `afri_search_threshold` | `7` | Bins searched left/right of the peak. |
| `afri_debris_size` | `10` | Frequency band size used when splitting. |
| `afri_min_samples` | `2` | Minimum usable positives before falling back to a uniform profile. |

Full description in [08_AFRI_INTEGRATION.md](08_AFRI_INTEGRATION.md).

---

## Removed keys

| Key | Why |
|---|---|
| `freq_mask_afri_weight` | AFRI is its own mode now, not a weight in the activation blend. |
| `freq_mask_ema_alpha` | Never read by the dataset. |
| `freq_mask_energy_ema_alpha` | Phase 2's energy profile is now precomputed per class, so there is no online estimate to tune. |
| `afri_min_bin_expand`, `afri_max_bin_expand` | Never passed to `compute_afri_profile`, which does not accept them. Range expansion has always used the original algorithm's hardcoded values (±3 if raw width ≤ 20, else −2/+6, clamped to [2,125]). |
| `afri_uniform_floor` | Existed only to keep the old slot lottery's sampling-without-replacement viable. AFRI has no lottery. |

`afri_min_bin_expand` / `afri_max_bin_expand` are **still present in the
`afri_config` dict that feeds `generate_afri_cache_key`**, with their former
default values. Removing them would change the cache key and discard
`.afri_cache/profiles.npz`, forcing a 2-hour recompute that produces identical
numbers. Do not "tidy" them out of that dict.

---

## The ablation settings

`run_ablation_experiments.py` overrides the defaults above. All masking
configurations share:

```
freq_mask_n_bands=4   freq_mask_prob=0.8   freq_mask_max_width=24
```

and then:

| Run | mode | Distinguishing overrides |
|---|---|---|
| `phase1_random` | `random` | — |
| `slot_uniform` | `slot_uniform` | — (the control) |
| `phase2_energy` | `energy` | `energy_weight=1.0` |
| `phase3_activation` | `activation` | `importance_weight=1.0` |
| `afri_direct` | `afri` | `use_afri=true`, `afri_mode=direct` |
| `afri_inverse` | `afri` | `use_afri=true`, `afri_mode=inverse` |

Eight configurations in total, with `baseline` and `cife`.
