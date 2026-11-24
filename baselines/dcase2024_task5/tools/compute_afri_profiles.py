#!/usr/bin/env python3
"""
compute_afri_profiles.py
========================
Build (or rebuild) the AFRI profile cache on its own, without training, then
check the two properties the direct/inverse ablation depends on.

NOTE: the cache-building half of this script is current -- it drives the
production path, and the cached profile (including auto_minbin/auto_maxbin) is
exactly what training consumes. The direct/inverse *checks* below, however,
analyse the superseded soft-profile slot weights, which AFRI range masking no
longer uses. For a check of the current mechanism run
``tools/verify_masking_wiring.py``.

Why this exists
---------------
``configs/train.yaml`` puts profile computation at "2+ hours", and it currently
happens inside the first training job that needs it.  That couples a 2-hour
data pass to a ~4-hour training run, so a bad profile surfaces six hours in.
This script runs only the profile pass and gates the result.

It drives the production code path (``PrototypeDynamicArrayDataSet.__init__``
-> ``build_meta`` -> ``_compute_afri_profiles``) rather than reimplementing the
computation, so the cache key and the profile contents are byte-identical to
what a training run would produce and write.  Reimplementing would risk a key
mismatch, and the training job would then silently recompute.

The cache key does NOT depend on masking geometry
-------------------------------------------------
``_compute_afri_profiles`` keys the cache on the training CSV mtimes plus
``afri_config`` (afri_n_samples, afri_value_threshold, afri_search_threshold,
afri_debris_size, afri_min_bin_expand, afri_max_bin_expand) and
``feature_config`` (n_mels, sr, hop_mel).  ``freq_mask_n_bands``,
``freq_mask_max_width`` and ``afri_mode`` are absent from it.  So this script
needs no masking overrides: run it bare and the cache it writes is the one both
AFRI arms will load.

The gates below are the exception -- they are a property of the *selection*
geometry, so they are evaluated at the geometry the ablation actually uses
(see ABLATION_N_BANDS / ABLATION_MAX_WIDTH), not at the train.yaml defaults,
which differ.

What it verifies
----------------
1.  **Direct mode is samplable.**  ``_refine_frequency_range`` zeroes bins
    outside the detected band.  After averaging into
    ``ceil(n_mels/max_width)`` slots, a narrow band can leave fewer non-zero
    slots than ``n_bands``, and ``rng.choice(..., replace=False, p=w)`` then
    raises ``ValueError: Fewer non-zero entries in p than size`` part-way
    through training, whenever such a class is drawn.

2.  **Inverse mode is a contrast, not random masking.**  If
    ``inverse_weights`` is ``1 - mask_weights`` over a profile that was zeroed
    outside the band, it is ~1 across most of the axis and averages to a
    near-uniform slot distribution.  The inverse arm then reproduces Phase 1
    random masking instead of opposing direct, and the pair is not a control.
    The gate compares how far each mode sits from uniform.

Usage
-----
Compute if absent (or load an existing cache), then verify::

    python tools/compute_afri_profiles.py

Force a rebuild.  Required after editing ``src/utils/afri_core.py``: the cache
key hashes CSV mtimes and config but NOT the AFRI code, so an edited formula
otherwise reloads stale profiles in silence::

    python tools/compute_afri_profiles.py train_param.afri_force_recompute=true

Exits non-zero if either gate fails, so it can guard the launch::

    python tools/compute_afri_profiles.py train_param.afri_force_recompute=true \
        && python run_ablation_experiments.py
"""

from __future__ import annotations

import sys

import hydra
import numpy as np
from omegaconf import DictConfig, open_dict

# Selection geometry of the two AFRI arms, from run_ablation_experiments.py
# (freq_mask_n_bands=4, freq_mask_max_width=24).  These deliberately override
# the train.yaml defaults (2 and 16): the gates must describe the runs that
# will actually be launched.  Keep in sync if the ablation changes.
ABLATION_N_BANDS = 4
ABLATION_MAX_WIDTH = 24

# Gate 2: mean TVD(inverse, uniform) must reach at least this fraction of
# mean TVD(direct, uniform) for the modes to be a genuine contrast.
INVERSE_CONTRAST_MIN_RATIO = 0.5


def to_slots(weights: np.ndarray, n_mels: int, max_width: int) -> np.ndarray:
    """[n_mels] -> [n_slots], mirroring _get_afri_weights_for_class exactly."""
    n_slots = int(np.ceil(n_mels / max_width))
    slots = np.zeros(n_slots)
    for i in range(n_slots):
        lo, hi = i * max_width, min((i + 1) * max_width, n_mels)
        slots[i] = np.mean(weights[lo:hi])
    if slots.sum() > 0:
        return slots / slots.sum()
    return np.ones(n_slots) / n_slots


def report(profiles: dict, n_mels: int, max_width: int, n_bands: int) -> bool:
    """Print the per-class table and both gates.  True if all gates pass."""
    n_slots = int(np.ceil(n_mels / max_width))
    uniform = np.ones(n_slots) / n_slots

    print(f"\nGate geometry: n_mels={n_mels}  max_band_width={max_width}  "
          f"n_slots={n_slots}  n_bands={n_bands}")
    print(f"Slots any two selections must share: "
          f"{max(0, 2 * n_bands - n_slots)} of {n_bands}\n")
    print(f"{'class':10s} {'valid':>6s} {'nzD':>4s} {'D-uni':>7s} "
          f"{'I-uni':>7s} {'D-I':>7s}")

    unsamplable, d_uni, i_uni, n_invalid = [], [], [], 0
    for name in sorted(profiles):
        p = profiles[name]
        if not bool(p.get("valid", False)):
            # frequency_masker returns uniform for both modes here, so the
            # class contributes no direct/inverse contrast at all.
            n_invalid += 1
            print(f"{name[:10]:10s} {'False':>6s} {'-':>4s}   "
                  f"(uniform in both modes)")
            continue

        d = to_slots(np.asarray(p["mask_weights"], float), n_mels, max_width)
        i = to_slots(np.asarray(p["inverse_weights"], float), n_mels, max_width)
        nz = int((d > 0).sum())
        td = 0.5 * np.abs(d - uniform).sum()
        ti = 0.5 * np.abs(i - uniform).sum()
        d_uni.append(td)
        i_uni.append(ti)
        if nz < n_bands:
            unsamplable.append((name, nz))

        flag = "  <-- UNSAMPLABLE" if nz < n_bands else ""
        print(f"{name[:10]:10s} {'True':>6s} {nz:4d} {td:7.3f} {ti:7.3f} "
              f"{0.5 * np.abs(d - i).sum():7.3f}{flag}")

    print(f"\nvalid: {len(d_uni)}/{len(profiles)}   "
          f"invalid (uniform in both modes): {n_invalid}")
    if not d_uni:
        print("\nFAIL: no valid profiles -- every class fell back to uniform.")
        return False

    md, mi = float(np.mean(d_uni)), float(np.mean(i_uni))
    print(f"mean TVD(direct , uniform) = {md:.3f}")
    print(f"mean TVD(inverse, uniform) = {mi:.3f}")

    ok = True

    print("\n--- Gate 1: direct mode samplable for every class ---")
    if unsamplable:
        ok = False
        print(f"FAIL: {len(unsamplable)} class(es) have fewer than "
              f"n_bands={n_bands} non-zero slots. The direct run WILL raise "
              f"'Fewer non-zero entries in p than size' when one is drawn:")
        for name, nz in unsamplable:
            print(f"    {name}: {nz}/{n_slots} non-zero slots")
        print("  Fix: apply a uniform floor in "
              "frequency_masker._get_afri_weights_for_class.")
    else:
        print(f"PASS: every valid profile keeps >= {n_bands} non-zero slots.")

    print("\n--- Gate 2: inverse mode is a contrast, not random ---")
    ratio = mi / md if md > 0 else 0.0
    print(f"  TVD(inverse,uniform) / TVD(direct,uniform) = {ratio:.2f}  "
          f"(need >= {INVERSE_CONTRAST_MIN_RATIO})")
    if ratio < INVERSE_CONTRAST_MIN_RATIO:
        ok = False
        print("FAIL: inverse sits far closer to uniform than direct does, so "
              "the inverse arm approximates random masking and the pair is "
              "not a control.\n  Cause: inverse_weights = 1 - mask_weights is "
              "~1 wherever range refinement zeroed the profile.\n"
              "  Fix: reflect rather than complement in "
              "src/utils/afri_core.py.")
    else:
        print("PASS: both modes are comparably far from uniform.")

    return ok


@hydra.main(config_path="../configs/", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # Imported inside main so Hydra sets up the run directory first, matching
    # train.py.
    from src.datamodules.components.dynamic_pcen_dataset import (
        PrototypeDynamicArrayDataSet,
    )

    with open_dict(cfg):
        cfg.train_param.use_afri = True
        cfg.train_param.afri_cache_enabled = True
        # The masker plays no part in building profiles, and constructing it
        # here would want an importance tracker this script has no reason to
        # create.  _compute_afri_profiles() runs independently of it, and the
        # profile -> masker handoff in build_meta() is guarded on
        # use_freq_masking, so disabling it is safe.
        cfg.train_param.use_freq_masking = False

    n_mels = cfg.features.get("n_mels", 128)
    cfg_bands = cfg.train_param.get("freq_mask_n_bands", 2)
    cfg_width = cfg.train_param.get("freq_mask_max_width", 16)

    print("=" * 70)
    print("AFRI profile computation (standalone)")
    print(f"  train_dir  : {cfg.path.train_dir}")
    print(f"  cache_dir  : "
          f"{cfg.train_param.get('afri_cache_dir') or '{train_dir}/.afri_cache'}")
    print(f"  recompute  : {cfg.train_param.get('afri_force_recompute', False)}")
    print(f"  n_mels     : {n_mels}")
    print("=" * 70)
    if (cfg_bands, cfg_width) != (ABLATION_N_BANDS, ABLATION_MAX_WIDTH):
        print(f"Note: config geometry is n_bands={cfg_bands}, "
              f"max_width={cfg_width}, but the AFRI arms run with "
              f"n_bands={ABLATION_N_BANDS}, max_width={ABLATION_MAX_WIDTH}.\n"
              f"      Gates use the ablation values. The cache itself is "
              f"geometry-independent, so no override is needed here.")

    # Constructing the dataset runs build_meta(), which calls
    # _compute_afri_profiles(): that either loads the existing cache or
    # computes and writes profiles.npz + metadata.json.
    dataset = PrototypeDynamicArrayDataSet(
        path=cfg.path,
        features=cfg.features,
        train_param=cfg.train_param,
    )

    profiles = {
        cls: dataset.meta[cls]["afri_profile"]
        for cls in dataset.meta
        if "afri_profile" in dataset.meta[cls]
    }
    if not profiles:
        print("\nFAIL: no profiles were produced.")
        sys.exit(1)

    print(f"\nProfiles available for {len(profiles)} classes.")
    ok = report(profiles, n_mels, ABLATION_MAX_WIDTH, ABLATION_N_BANDS)

    print("\n" + "=" * 70)
    if ok:
        print("Both gates passed. The cache is warm; launch the AFRI runs.")
        print("Then confirm the mode is actually wired: each run's log should "
              "print `- afri_mask_mode:` with the expected value, and the two "
              "arms' first-step losses should differ.")
    else:
        print("Gates FAILED -- fix the causes above before spending ~9 h on "
              "the two runs.")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
