# Wiring Verification — measured behaviour and known limits

**Status:** current. Reproduce with `python tools/verify_masking_wiring.py`.

Every number here was measured by running the real masker classes over synthetic
spectrograms with the exact ablation parameters (`n_bands=4`, `mask_prob=0.8`,
`max_band_width=24`, `n_mels=128`, `seed=1234`), and — for AFRI — against the real
cached profiles in `.afri_cache/profiles.npz`.

---

## What each strategy does

| Strategy | Intent | Measured |
|---|---|---|
| `random` | mask random bands | 0.804 of calls masked (`mask_prob=0.8`) ✓ |
| `slot_uniform` | every slot equally likely | 0.276 per full-width slot ✓ |
| `energy` | spare the slots where the class is loud | class's loud slot masked **0.000**, on positives *and* negatives ✓ |
| `activation` | target the slots the encoder uses for the class | class's own slot preferred over all others ✓ |
| `afri` direct | mask inside the class range | **1.000** of masked bins inside the range ✓ |
| `afri` inverse | spare the class range | **0.000** of masked bins inside the range ✓ |

All three guided strategies are per class, and a different class produces a
different slot distribution — the check that would have caught both of the
defects described under *Superseded designs* below.

Masked area per masked call, for context:

| Strategy | mean bins | worst single call |
|---|---|---|
| `random` | 43.0 | 82 |
| `slot_uniform` | 44.6 | 89 |
| `afri`, range-guided classes | 30.7 (min 1, max 44) | 44 |
| `afri`, unguided fallback classes | 44.3 | 88 |

Nothing can mask the whole spectrum: `n_bands=4` slots × 24 bins caps it at 96,
and AFRI is capped at `min(area_budget, W, n_mels - W)`.

---

## Coverage is not uniform per bin, in two different ways

Neither placement rule gives every mel bin equal coverage, and the two are shaped
differently. This is independent of any guidance signal, so a per-bin analysis of
"which frequencies get masked" has to account for it.

**`random`: a one-sided ramp from each spectrum edge.** A band covering bin 0 must
start exactly at 0, so:

```
bins 0-5    0.025 0.054 0.082 0.108 0.129 0.150
bins 60-65  ~0.30  (plateau)
```

Bin 0 is covered 9.2× less than the plateau.

**The slot strategies: a sawtooth with period `max_band_width`.** A band of width
w at a uniform offset inside a 24-bin slot covers the slot's boundary bins only
at extremal offsets:

```
slot boundaries (bins 0, 24, 48, 72, 96)    ~0.083
slot centres    (bins 12, 36, 60, 84, 108)  ~0.374
```

A 4.5× ripple, repeating at every slot boundary rather than only at the spectrum
edges. Milder than `random`'s edge deficit at bin 0 (0.083 vs 0.025) but present
across the whole axis.

**Consequence for the comparison.** `random` differs from the guided strategies in
the placement rule as well as the guidance, and that difference is not neutral.
`slot_uniform` is therefore the control: it holds the placement rule fixed and
turns off only the guidance, so a null result against it means the signal did not
help, rather than that the geometry could not express it.

Total masked *area* is nonetheless comparable across strategies (43–45 bins,
within 5%), so augmentation strength is constant even though its distribution
over frequency is not.

---

## AFRI's per-class area varies widely

`M = min(area_budget, W, n_mels - W)` gives min 1, median 31, max 44 bins across
the 43 range-guided classes. Both extremes come from the formula:

- a **narrow** range gives a small M (`c_17` has `W = 1`, so `M = 1`)
- a **very wide** range also gives a small M, because the complement is what
  limits `inverse` (`AMRE` has `W = 123`, so `M = 5`)

Direct and inverse are exactly paired for a given class, which is the comparison
that matters. But augmentation strength varies up to 44× *across* classes, so a
handful of classes are trained nearly unaugmented. That is inherent to masking
exactly the identified range under an equal-area constraint; flooring M would
break the strict inside/outside property for the narrow classes.

Three of 46 classes (`c_1`, `c_5`, `c_14`) have invalid profiles and fall back to
unguided `slot_uniform` masking, reported once per class at startup. They receive
an identical treatment under both modes and cannot contribute contrast, so report
43/46 as range-guided.

---

## Superseded designs, and why they were wrong

Recorded because the runs in `paper/run_logs/` used them. All of these must be
repeated.

**AFRI was a soft slot lottery that never read the range.** `auto_minbin` /
`auto_maxbin` were computed, cached and logged, and nothing consumed them; only
the broad `mask_weights` curve was used, averaged into slots.

| | masked bins inside the range | range that gets masked |
|---|---|---|
| direct (intent ≈1.00) | 0.49 | 0.40 |
| inverse (intent ≈0.00) | 0.30 | 0.21 |

Band placement within a slot also used a deterministic argmax over the bin-level
profile, gated so only the AFRI runs reached it. Disabling it dropped the
direct-vs-inverse L1 divergence from 30.7 to 9.2, so about two thirds of the
reported 3.51-point separation came from a mechanism the paper did not describe.

**Phase 2 kept one global online energy EMA.** `IdentityBatchSampler` tiles
`k_way = 10` distinct classes, so the class changes on every `__getitem__`; one
running EMA converged to a mixture over those classes rather than any one
profile:

```
global EMA after training (max-normalised)   [0.555 0.203 0.094 1.000 0.348 0.152]
```

| `alpha_e` | global EMA | per-class |
|---|---|---|
| 0.1 (as run) | 0.219 | 0.000 |
| 0.3 | 0.000 | 0.000 |

So ~22% of segments had their informative band masked. A per-segment estimate is
not the fix either: half of all masker calls are on negative segments, which
contain no target sound, and there a per-segment estimate spares an arbitrary
noise band (0.156, against 1/6 = 0.167 chance). Only a per-class profile is
correct on both, which is why energy is now precomputed per class.

**Phase 3 kept one global importance vector, updated in a forward hook.** A hook
sees activations but not labels, so it could only produce one profile shared by
every class. Measured: spectrograms with energy in entirely different slots
produced identical per-slot mask rates.

```
input energy in slot 0   [0.325 0.311 0.306 0.312 0.515 0.347]
input energy in slot 2   [0.334 0.308 0.311 0.308 0.517 0.333]
input energy in slot 5   [0.322 0.326 0.311 0.319 0.508 0.330]
```

Attribution now happens in `training_step`, where the class names are known.

**The train loader forked 2 workers while the config said 0.**
`configs/datamodule/prototype.yaml` set `num_workers: 0`, which reached
`self.hparams.num_workers` and was never read; the loader hardcoded 2. Every run
log printed 0 and every run forked 2. That mattered three ways: forked workers
each inherited the same seeded masking RNG, so they drew **identical** mask
sequences and each epoch's fresh workers replayed the same sequence; the
importance tracker was read as a frozen epoch-start snapshot; and the dataset's
open `h5py` handles were inherited across the fork, which HDF5 does not support
(the test loader already sets `num_workers=0` for exactly that reason). The
loader now honours the config.

---

## Current limits, deliberately accepted

- **Slot 5 is 8 bins wide, not 24.** `ceil(128/24) = 6` slots with a narrow tail,
  so a slot is not a uniform unit of masking and P(b) values are not directly
  comparable across slots. `max_band_width = 16` (8 equal slots) or `32` (4) would
  fix it, at the cost of changing what `n_bands=4` means.
- **The within-slot ripple** described above is not removable by equalising slot
  widths; it comes from uniform offset placement inside a slot.
- **Phase 3's importance is epoch-stale if `num_workers > 0`.** At the configured
  `0` it is read live in the same process. The `reseed()` path exists so that
  raising `num_workers` does not silently reintroduce duplicate mask streams, but
  HDF5 handle inheritance would still need solving first (reopen per worker in a
  `worker_init_fn`; never inherit).
