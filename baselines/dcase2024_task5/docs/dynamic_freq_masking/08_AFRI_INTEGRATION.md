# AFRI Range Masking

**Status:** current — describes what the code does.
**Supersedes:** the soft-profile slot-lottery integration used for the
`afri_direct` / `afri_inverse` runs in `paper/run_logs/`. Those runs do not
reflect this document and must be repeated.

---

## What it does

AFRI (Automatic Frequency Range Identification) identifies, per class, the
frequency range in which the target sound is most distinguishable from
background. `AFRIRangeMasker` then masks a fixed area of the spectrogram either
strictly **inside** that range (`afri_mode: direct`) or strictly **outside** it
(`afri_mode: inverse`).

```
class BBWA, range 107..123, area M = 16

direct    ...................................########........
inverse   ####..####...###..#####...................          <- never 107..123
                                    ^^^^^^^^^^^^^^^ spared
```

- **direct** removes the discriminative band, training the encoder to fall back
  on secondary spectral cues.
- **inverse** removes everything except that band, approximating a learned
  denoising of the input.

Both modes mask the **same number of bins for a given class**, on disjoint
regions. The pair is therefore an exactly matched comparison: any difference
between the two runs is attributable to *where* the mask falls, not how much of
the spectrogram it covers.

AFRI is a strategy in its own right (`freq_mask_mode: afri`). It does not run
through Phase 3: no slot lottery, no activation tracker, and none of the
energy/importance weights apply.

Phase 2 now profiles **the same first `afri_n_samples` positive segments** per
class, so Phase 2 and AFRI see identical information and differ only in the
transform applied to it — mean energy there, positive-minus-negative contrast
here. That is what makes the two directly comparable.

---

## Why it was rewritten

The previous implementation put AFRI inside `ActivationGuidedFrequencyMasker` as
a fourth term in a weight mix. It never read `auto_minbin`/`auto_maxbin` — the
range, which is AFRI's actual product. It used only the broad continuous
`mask_weights` curve, averaged into 24-bin slots and used as a lottery weight.

Measured over the 43 valid cached profiles:

| | masked bins inside the range | range that gets masked |
|---|---|---|
| direct (intent ≈1.00 / high) | **0.49** | **0.40** |
| inverse (intent ≈0.00 / ≈0.00) | **0.30** | **0.21** |

Direct masked the class's range less than half the time it masked anything and
left 60% of the range untouched; inverse, which is meant to *spare* the range,
put 30% of its masking inside it. The mel-averaged slot weights sat within
0.03–0.12 of uniform (uniform = 0.167), so the slot lottery was close to
unguided.

Two further problems:

1. **An undescribed second mechanism.** Band placement *within* a slot used a
   deterministic argmax over the bin-level profile, gated on `afri_weight > 0`
   so only the AFRI runs reached it. Disabling it dropped the direct-vs-inverse
   L1 divergence from 30.7 to 9.2 — about two thirds of the observed separation
   came from that argmax, not from the slot distribution the method was
   described by.
2. **Unequal intervention strength.** Realized per-slot selection swing
   (uniform = 0.667): Phase 2 energy 0.883, AFRI direct 0.652, Phase 3
   activation 0.400, AFRI inverse 0.211. Inverse was nearly indistinguishable
   from unguided slot masking while direct was a strong intervention, so the two
   modes were not mirror-image treatments.

Root cause: the shared code path was a 4-of-6 slot lottery, and range-based
masking has no slot distribution at all. Forcing AFRI through it is what
produced the compromise.

---

## Algorithm

Per class, from the cached profile: `lo = auto_minbin`, `hi = auto_maxbin`,
`W = hi - lo`.

**1. Eligible region**

```
direct    [lo, hi)
inverse   [0, lo)  union  [hi, n_mels)
```

**2. Masked area**

```
M = min(area_budget, W, n_mels - W)
```

Taking the minimum over both the range *and* its complement is what keeps M
placeable in either mode, which is what makes the two modes exactly paired. For
most classes `W` is below the budget, so direct masks the whole range.

`area_budget` defaults to **44** bins. That is calibrated to the *union* area the
slot-based phases actually mask — not the naive `n_bands × mean_width = 50`,
because their bands may overlap. The measured per-bin mask rate for uniform slot
selection is 0.276 at `mask_prob=0.8`, i.e. `0.276 / 0.8 × 128 ≈ 44` bins
conditional on masking. This makes AFRI's coverage comparable to Phases 1–3
wherever the range allows it.

**3. Band layout**

- Allocate M across the eligible intervals in proportion to their capacity.
  Allocating area to intervals *before* choosing widths is what makes the union
  area exact: each interval receives no more than it can hold, so placement can
  never fail. Doing it the other way round (widths first, then packing) can
  strand a wide band that fits in neither interval and silently under-mask.
- Split each interval's allocation into bands of random width, capped at
  `max_band_width`, aiming for `n_bands` in total.
- Lay them out non-overlapping at random offsets, distributing the slack over
  the gaps as a uniform random composition.

Every band lies entirely within one interval. In inverse mode a band spanning
the gap would mask inside the AFRI range, which is the one thing the mode must
never do.

**4. Stochasticity**

The region is fixed and range-determined; the bands within it are resampled on
every call. This keeps augmentation diversity without ever letting masking stray
outside the region. Union area is exactly M on every masked call.

`mask_prob` applies first, as for every other strategy.

---

## Range expansion and edge cases

The range is expanded before use exactly as the original AFRI does
(`src/utils/afri_core.py` Step 7, a line-for-line port of `binsexpand`,
lines 708-721 of the original [AFRI reference snapshot](../../../../THIRD_PARTY_NOTICES.md#afri)):

```python
if width <= 20:  lo -= 3;  hi += 3
if width  > 20:  lo -= 2;  hi += 6
lo = max(lo, 2);  hi = min(hi, 125)
```

The original enforces **no minimum width and excludes no class**, so neither do
we. On the DCASE 2024 training split, post-expansion:

- range width: min 1, 25th pct 21, median 37, 75th pct 48, max 123
- **13 of 43** classes have a range narrower than one 24-bin band — kept as-is
- **1** class (`c_17`, `[2,3]`) is degenerate from the low-edge clamp. `M = 1`
  in both modes, so the pairing still holds and no special case is needed.

The former `afri_min_bin_expand` / `afri_max_bin_expand` config keys were never
passed to `compute_afri_profile`, which does not accept them; the expansion has
always used the original's hardcoded values. The keys have been removed from the
YAML. **They are deliberately still present in the `afri_config` dict that feeds
`generate_afri_cache_key`** — removing them would change the cache key and
discard `.afri_cache/profiles.npz`, triggering a 2-hour recompute that would
produce identical numbers.

**Per-class area varies widely.** `M` ranges from 1 to 44 bins (median 31) across
the 43 range-guided classes, because both a very narrow range and a very wide one
give a small `M` — the latter because the complement is what limits `inverse`.
Direct and inverse stay exactly paired for a given class, but augmentation
strength varies up to 44× *across* classes, so a few classes are trained nearly
unaugmented. Flooring `M` would break the strict inside/outside property.

**Classes with no usable range** — profile missing, `valid: False`, or a
degenerate range — fall back to unguided uniform slot masking, reported once per
class at startup. On this split that is 3 of 46 classes (`c_1`, `c_5`, `c_14`),
so 43/46 are range-guided. Report this in any writeup: those three classes
receive an identical treatment under both modes and cannot contribute contrast.

---

## Cache

No recomputation is needed for this change. `auto_minbin` and `auto_maxbin` were
always part of the cached payload, and the cache key contains no masking or mode
field. `src/utils/afri_cache.py` is untouched.

If a future change alters what is *stored* in the profile, add an explicit
version field to the cache key rather than mutating `afri_config`.

---

## Configuration

```yaml
train_param:
  use_freq_masking: true
  freq_mask_mode: 'afri'
  freq_mask_n_bands: 4
  freq_mask_prob: 0.8
  freq_mask_max_width: 24
  freq_mask_afri_area_budget: 44   # masked bins, capped per class

  use_afri: true                   # must be true, or every class falls back
  afri_mode: 'direct'              # or 'inverse'
```

The dataset prints a warning at startup if `freq_mask_mode: afri` is set with
`use_afri: false`, since that silently reduces AFRI to unguided slot masking.

---

## Verification

```bash
python tools/verify_masking_wiring.py          # measured behaviour, all strategies
pytest tests/unit/test_afri_frequency_masking.py -v
```

The wiring check asserts direct masks only inside the range (1.000) and inverse
masks none of it (0.000), against the real cached profiles — the numbers the old
implementation scored 0.49 and 0.30 on.

See [09_WIRING_VERIFICATION.md](09_WIRING_VERIFICATION.md) for the full measured
behaviour of all four strategies and the known open issues.
