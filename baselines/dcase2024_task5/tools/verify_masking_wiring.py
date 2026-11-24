#!/usr/bin/env python
"""
Verify that every frequency-masking strategy affects the spectrogram as intended.

Runs the real masker classes over synthetic spectrograms with the exact
parameters from run_ablation_experiments.py, and checks that each strategy's
signal actually steers where the mask lands.

The five strategies and what each must demonstrate:

    random        masks at the configured rate; free placement, so edge bins are
                  covered much less than centre bins (this is why it is NOT the
                  control for the guided strategies)
    slot_uniform  the control: every slot equally likely
    energy        spares the slots where the CLASS is loud, on positives AND on
                  negatives, from its per-class precomputed profile
    activation    targets the slots the encoder uses for THAT class, and
                  different classes get different masks
    afri direct   masks only inside the class's AFRI range
    afri inverse  masks none of it, with the same area as direct

The per-class checks are the ones that would have caught the two defects this
tool was written after: a global energy EMA that tracked a mixture of classes
rather than any one of them, and a global activation profile that made Phase 3's
slot distribution independent of the input entirely.

Only needs numpy and (for the AFRI checks) the cached AFRI profiles; no model,
no dataset, no GPU.

Usage:
    python tools/verify_masking_wiring.py
    python tools/verify_masking_wiring.py --cache path/to/.afri_cache/profiles.npz
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Loaded straight from the file rather than as src.datamodules.components.*, so
# this stays runnable with numpy alone -- that package's __init__ imports torch.
_spec = importlib.util.spec_from_file_location(
    "frequency_masker",
    os.path.join(_REPO, "src", "datamodules", "components", "frequency_masker.py"),
)
_fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm)

AFRI_AREA_BUDGET = _fm.AFRI_AREA_BUDGET
AFRIRangeMasker = _fm.AFRIRangeMasker
ActivationGuidedFrequencyMasker = _fm.ActivationGuidedFrequencyMasker
EnergyBasedFrequencyMasker = _fm.EnergyBasedFrequencyMasker
RandomFrequencyMasker = _fm.RandomFrequencyMasker
UniformSlotFrequencyMasker = _fm.UniformSlotFrequencyMasker

N_MELS = 128
PARAMS = dict(n_bands=4, mask_prob=0.8, max_band_width=24, n_mels=N_MELS, seed=1234)
SLOTS = [(i * 24, min((i + 1) * 24, N_MELS)) for i in range(6)]
N_TRIALS = 4000
TOL = 0.02

REAL_CACHE = os.path.join(_REPO, ".afri_cache", "profiles.npz")

# Six synthetic classes, each loud in a different slot
CLASSES = {f"cls{i}": i for i in range(6)}


def check(label, actual, expected, tol=TOL):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    ok = bool(np.all(np.abs(actual - expected) <= tol))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"         expected {np.round(expected, 3)}")
        print(f"         actual   {np.round(actual, 3)}")
        print(f"         delta    {np.round(actual - expected, 3)} (tol {tol})")
    return ok


def make_spec_factory(seed=0):
    """Segments whose energy sits in a chosen slot; negatives have no peak."""
    rng = np.random.RandomState(seed)

    def positive(slot, t=30):
        s = rng.rand(t, N_MELS) * 0.05
        a, b = SLOTS[slot]
        s[:, a:b] += 1.0
        return s

    def negative(t=30):
        return rng.rand(t, N_MELS) * 0.05

    return positive, negative


def mask_rates(masker, class_name=None, n=N_TRIALS, spec_seed=0, slot=0,
               use_negative=False):
    """Per-bin fraction of calls in which each mel bin was zeroed."""
    positive, negative = make_spec_factory(spec_seed)
    hist = np.zeros(N_MELS)
    masked_calls = 0
    for _ in range(n):
        spec = negative() if use_negative else positive(slot)
        zeroed = (masker(spec, class_name) == 0.0).all(axis=0)
        if zeroed.any():
            masked_calls += 1
        hist += zeroed
    return hist / n, masked_calls / n


def per_slot(rates):
    return [rates[a:b].mean() for a, b in SLOTS]


# ----------------------------------------------------------------------

def verify_random():
    print("\nrandom (Phase 1, SpecAugment):")
    rates, call_rate = mask_rates(RandomFrequencyMasker(**PARAMS))
    ok = check("mask_prob honoured", [call_rate], [0.804])
    ok &= check("edge bins 0-5 under-masked (free placement bias)",
                rates[:6], [0.025, 0.054, 0.082, 0.108, 0.129, 0.150])
    ok &= check("centre bins 60-65", rates[60:66], [0.30] * 6, tol=0.03)
    return ok


def verify_slot_uniform():
    print("\nslot_uniform (control):")
    rates, call_rate = mask_rates(UniformSlotFrequencyMasker(**PARAMS))
    ok = check("mask_prob honoured", [call_rate], [0.80], tol=0.03)
    slots = per_slot(rates)
    ok &= check("every full-width slot equally likely", slots[:5], [0.276] * 5,
                tol=0.03)

    # Slot selection is uniform, but per-BIN coverage is not: a band of width w
    # at a uniform offset inside a 24-bin slot covers the slot's boundary bins
    # only at extremal offsets. The result is a sawtooth with period
    # max_band_width, dipping at every slot boundary. This affects all four
    # slot-based strategies and is independent of any guidance signal, so per-bin
    # coverage analyses must account for it.
    boundaries = [rates[b] for b in (0, 24, 48, 72, 96)]
    centres = [rates[b] for b in (12, 36, 60, 84, 108)]
    ok &= check("slot-boundary bins dip (within-slot placement ripple)",
                boundaries, [0.083] * 5, tol=0.03)
    ok &= check("slot-centre bins peak", centres, [0.374] * 5, tol=0.03)

    # The ripple is milder and differently shaped than random's edge ramp:
    # periodic at slot boundaries rather than a one-sided ramp from bin 0.
    random_rates, _ = mask_rates(RandomFrequencyMasker(**PARAMS))
    ok &= check("bin 0 is covered far better than under random",
                [rates[0] > 3 * random_rates[0]], [True], tol=0)
    return ok


def verify_energy():
    print("\nenergy (Phase 2, per-class precomputed profiles):")
    positive, negative = make_spec_factory(1)

    # Profiles built the way the dataset builds them: mean |value| per bin over
    # that class's first few positive segments.
    profiles = {
        name: np.mean([np.abs(positive(slot)).mean(axis=0) for _ in range(5)], axis=0)
        for name, slot in CLASSES.items()
    }
    masker = EnergyBasedFrequencyMasker(
        **PARAMS, energy_weight=1.0, energy_profiles=profiles)

    ok = True
    loud_on_pos, loud_on_neg = [], []
    for name, slot in CLASSES.items():
        a, b = SLOTS[slot]
        rates, _ = mask_rates(masker, name, n=1200, slot=slot)
        loud_on_pos.append(rates[a:b].mean())
        # The SAME class profile must also protect its band on a negative
        # segment, which contains no target sound at all. A per-segment or
        # global-EMA estimate cannot do this.
        rates_neg, _ = mask_rates(masker, name, n=1200, slot=slot,
                                  use_negative=True)
        loud_on_neg.append(rates_neg[a:b].mean())

    ok &= check("class's loud slot never masked, on positives",
                [max(loud_on_pos)], [0.0], tol=1e-9)
    ok &= check("class's loud slot never masked, on negatives too",
                [max(loud_on_neg)], [0.0], tol=1e-9)

    # And the guidance must be class-specific, not one shared profile
    distinct = {tuple(np.round(masker.slot_weights_for_class(n), 4))
                for n in CLASSES}
    ok &= check("each class gets its own slot distribution",
                [len(distinct)], [len(CLASSES)], tol=0)

    unknown, _ = mask_rates(masker, "never_seen", n=800)
    ok &= check("unknown class falls back to unguided (reaches every slot\n              across calls; per call it masks the same ~44 bins as slot_uniform)",
                [min(per_slot(unknown)) > 0.1], [True], tol=0)
    return ok


def verify_activation():
    print("\nactivation (Phase 3, per-class importance):")

    class StubTracker:
        """Per-class importance peaked on that class's own slot."""

        def __init__(self):
            self.profiles = {}
            for name, slot in CLASSES.items():
                v = np.full(N_MELS, 0.05)
                a, b = SLOTS[slot]
                v[a:b] = 1.0
                self.profiles[name] = v

        def get_importance(self, class_name=None):
            return self.profiles.get(class_name)

        def get_stats(self):
            return {'n_classes_tracked': len(self.profiles), 'total_updates': 1,
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 1.0,
                    'mean_spread': 0.95}

    masker = ActivationGuidedFrequencyMasker(
        **PARAMS, importance_weight=1.0, importance_tracker=StubTracker())

    ok = True
    own, other = [], []
    for name, slot in CLASSES.items():
        a, b = SLOTS[slot]
        rates, _ = mask_rates(masker, name, n=1200, slot=slot)
        own.append(rates[a:b].mean())
        other.append(np.mean([rates[x:y].mean()
                              for j, (x, y) in enumerate(SLOTS) if j != slot]))
    ok &= check("the class's important slot is preferred",
                [min(o - t for o, t in zip(own, other)) > 0.05], [True], tol=0)

    # The defect this replaces: a global profile made masking identical for
    # every class. Different classes must now produce different masks.
    distinct = {tuple(np.round(masker.slot_weights_for_class(n), 4))
                for n in CLASSES}
    ok &= check("each class gets its own slot distribution",
                [len(distinct)], [len(CLASSES)], tol=0)

    no_tracker = ActivationGuidedFrequencyMasker(
        **PARAMS, importance_weight=1.0, importance_tracker=None)
    rates, _ = mask_rates(no_tracker, "cls0", n=1200)
    ok &= check("no tracker falls back to uniform slots",
                per_slot(rates)[:5], [0.276] * 5, tol=0.03)

    untracked, _ = mask_rates(masker, "never_seen", n=800)
    ok &= check("untracked class falls back to unguided",
                per_slot(untracked)[:5], [0.276] * 5, tol=0.03)
    return ok


def verify_afri(profiles):
    print("\nafri (per-class range, direct and inverse):")
    ok = True
    probe = AFRIRangeMasker(**PARAMS, afri_profiles=profiles)
    valid = [c for c in profiles if probe.range_for_class(c) is not None]
    print(f"  profiles: {len(profiles)} classes, {len(valid)} with a usable range")

    stats = {}
    for mode in ("direct", "inverse"):
        m = AFRIRangeMasker(**PARAMS, afri_profiles=profiles, afri_mode=mode)
        inside_of_masked, area_exact = [], []
        for c in valid:
            lo, hi = m.range_for_class(c)
            area = m.masked_area_for_class(c)
            total = inside = 0
            areas_seen = []
            positive, _ = make_spec_factory(1)
            for _ in range(300):
                zeroed = (m(positive(0), c) == 0.0).all(axis=0)
                n_zero = int(zeroed.sum())
                if n_zero:
                    areas_seen.append(n_zero)
                total += n_zero
                inside += int(zeroed[lo:hi].sum())
            inside_of_masked.append(inside / max(total, 1))
            area_exact.append(all(a == area for a in areas_seen))
        stats[mode] = (float(np.mean(inside_of_masked)), all(area_exact))

    print(f"  direct  masked bins inside the range: {stats['direct'][0]:.3f}")
    print(f"  inverse masked bins inside the range: {stats['inverse'][0]:.3f}")
    print("  (superseded soft-profile implementation: 0.49 and 0.30)")

    ok &= check("direct masks only inside the range", [stats['direct'][0]], [1.0],
                tol=1e-9)
    ok &= check("inverse masks nothing inside the range", [stats['inverse'][0]],
                [0.0], tol=1e-9)
    ok &= check("union area is exactly M on every call, both modes",
                [stats['direct'][1], stats['inverse'][1]], [True, True], tol=0)

    md = AFRIRangeMasker(**PARAMS, afri_profiles=profiles, afri_mode="direct")
    mi = AFRIRangeMasker(**PARAMS, afri_profiles=profiles, afri_mode="inverse")
    ok &= check("areas paired across modes",
                [all(md.masked_area_for_class(c) == mi.masked_area_for_class(c)
                     for c in valid)], [True], tol=0)

    areas = [md.masked_area_for_class(c) for c in valid]
    widths = [md.range_for_class(c)[1] - md.range_for_class(c)[0] for c in valid]
    ok &= check("M == min(area_budget, W, n_mels - W)", areas,
                [min(AFRI_AREA_BUDGET, w, N_MELS - w) for w in widths], tol=0)
    print(f"         masked area: min={min(areas)} "
          f"median={int(np.median(areas))} max={max(areas)} bins "
          f"(budget {AFRI_AREA_BUDGET})")

    invalid = [c for c in profiles if md.range_for_class(c) is None]
    print(f"         {len(invalid)} classes fall back to unguided: {sorted(invalid)}")
    return ok


def verify_reseed():
    print("\nRNG specialisation (guards against duplicate masks across workers):")
    positive, _ = make_spec_factory(5)

    def stream(masker, n=8):
        return [tuple(np.flatnonzero((masker(positive(0)) == 0.0).all(axis=0)))
                for _ in range(n)]

    a = RandomFrequencyMasker(**PARAMS)
    b = RandomFrequencyMasker(**PARAMS)
    ok = check("same seed, no reseed -> identical streams (the hazard)",
               [stream(a) == stream(b)], [True], tol=0)

    c = RandomFrequencyMasker(**PARAMS)
    c.reseed(11)
    d = RandomFrequencyMasker(**PARAMS)
    d.reseed(22)
    ok &= check("different reseed offsets -> different streams",
                [stream(c) != stream(d)], [True], tol=0)

    e = RandomFrequencyMasker(**PARAMS)
    e.reseed(11)
    f = RandomFrequencyMasker(**PARAMS)
    f.reseed(11)
    ok &= check("same reseed offset -> reproducible", [stream(e) == stream(f)],
                [True], tol=0)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=REAL_CACHE,
                        help="path to .afri_cache/profiles.npz")
    args = parser.parse_args()

    print("=" * 74)
    print("Frequency masking: does each strategy steer the mask as intended?")
    print("=" * 74)

    ok = verify_random()
    ok &= verify_slot_uniform()
    ok &= verify_energy()
    ok &= verify_activation()
    ok &= verify_reseed()

    if os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        ok &= verify_afri({k: z[k].item() for k in z.keys()})
    else:
        print(f"\nSKIP afri checks: no profile cache at {args.cache}")

    print()
    print("=" * 74)
    print("RESULT:", "all checks passed" if ok else "FAILURES above")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
