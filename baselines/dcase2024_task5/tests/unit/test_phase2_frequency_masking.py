"""
Unit tests for Phase 2: EnergyBasedFrequencyMasker (per-class energy guidance).

Phase 2 spares the slots where a CLASS is loud, using a per-class profile
precomputed over that class's first N positive segments.

These tests replace a suite written against the superseded implementation, which
kept one global online energy EMA. That EMA had two defects the tests below are
built to prevent returning:

  * it was global, so under the interleaved episodic order (IdentityBatchSampler
    tiles k_way distinct classes, so the class changes on every __getitem__) it
    converged to a mixture over the k_way classes rather than any one class's
    profile, leaving ~22% of segments with their informative band masked;
  * it was estimated from whatever segments happened to arrive, so on the ~50% of
    calls that mask a NEGATIVE segment -- which contains no target sound -- it
    spared an arbitrary noise band.

Both are covered here by asserting per-class behaviour and by testing negatives
explicitly.

Run:
    pytest tests/unit/test_phase2_frequency_masking.py -v
    python tests/unit/test_phase2_frequency_masking.py
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _REPO)

try:  # Normal path when the full training environment is installed
    from src.datamodules.components.frequency_masker import (
        ActivationGuidedFrequencyMasker,
        AFRIRangeMasker,
        EnergyBasedFrequencyMasker,
        RandomFrequencyMasker,
        UniformSlotFrequencyMasker,
    )
except ImportError:  # pragma: no cover - keeps the suite runnable with numpy alone
    _spec = importlib.util.spec_from_file_location(
        "frequency_masker",
        os.path.join(_REPO, "src", "datamodules", "components",
                     "frequency_masker.py"),
    )
    _fm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fm)
    ActivationGuidedFrequencyMasker = _fm.ActivationGuidedFrequencyMasker
    AFRIRangeMasker = _fm.AFRIRangeMasker
    EnergyBasedFrequencyMasker = _fm.EnergyBasedFrequencyMasker
    RandomFrequencyMasker = _fm.RandomFrequencyMasker
    UniformSlotFrequencyMasker = _fm.UniformSlotFrequencyMasker


N_MELS = 128
MAX_W = 24
SLOTS = [(i * MAX_W, min((i + 1) * MAX_W, N_MELS))
         for i in range(int(np.ceil(N_MELS / MAX_W)))]
PARAMS = dict(n_bands=4, mask_prob=1.0, max_band_width=MAX_W, n_mels=N_MELS,
              seed=1234)

# Six classes, each loud in a different slot
CLASSES = {f"cls{i}": i for i in range(len(SLOTS))}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def positive(slot, t=30, seed=None):
    """A segment with its energy concentrated in `slot`."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    s = rng.rand(t, N_MELS) * 0.05
    a, b = SLOTS[slot]
    s[:, a:b] += 1.0
    return s


def negative(t=30, seed=None):
    """Background only: no target sound anywhere."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    return rng.rand(t, N_MELS) * 0.05


def make_profiles():
    """Per-class profiles, as the dataset builds them: mean |value| per bin."""
    return {
        name: np.mean([np.abs(positive(slot)).mean(axis=0) for _ in range(5)],
                      axis=0).astype(np.float32)
        for name, slot in CLASSES.items()
    }


def masked_bins(out):
    return np.flatnonzero((out == 0.0).all(axis=0))


def make_masker(profiles=None, energy_weight=1.0, **kwargs):
    params = dict(PARAMS)
    params.update(kwargs)
    return EnergyBasedFrequencyMasker(
        **params, energy_weight=energy_weight, energy_profiles=profiles)


# ----------------------------------------------------------------------

class TestConstruction:

    def test_defaults(self):
        m = EnergyBasedFrequencyMasker(n_mels=N_MELS)
        assert m.energy_profiles is None
        assert 0.0 <= m.energy_weight <= 1.0

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_invalid_energy_weight(self, bad):
        with pytest.raises(AssertionError, match="energy_weight"):
            make_masker(None, energy_weight=bad)

    def test_invalid_mask_prob(self):
        with pytest.raises(AssertionError, match="mask_prob"):
            make_masker(None, mask_prob=1.5)

    def test_max_band_width_cannot_exceed_n_mels(self):
        with pytest.raises(AssertionError, match="max_band_width"):
            make_masker(None, max_band_width=N_MELS + 1)

    def test_no_ema_attributes(self):
        """
        The online EMA is gone. Energy is a property of the data, so there is
        nothing to estimate during training.
        """
        m = make_masker(make_profiles())
        for attr in ('ema_alpha', 'band_energy_ema', 'n_samples_seen'):
            assert not hasattr(m, attr), f"{attr} should no longer exist"


class TestProfileLookup:

    def test_profile_returned_for_known_class(self):
        m = make_masker(make_profiles())
        profile = m.profile_for_class("cls2")
        assert profile is not None
        assert profile.shape == (N_MELS,)

    def test_none_without_profiles(self):
        assert make_masker(None).profile_for_class("cls0") is None

    def test_none_for_unknown_class(self):
        assert make_masker(make_profiles()).profile_for_class("nope") is None

    def test_none_for_none_class(self):
        assert make_masker(make_profiles()).profile_for_class(None) is None

    def test_none_for_wrong_shape(self):
        m = make_masker({"cls0": np.ones(64, dtype=np.float32)})
        assert m.profile_for_class("cls0") is None

    def test_none_for_all_zero_profile(self):
        m = make_masker({"cls0": np.zeros(N_MELS, dtype=np.float32)})
        assert m.profile_for_class("cls0") is None

    def test_none_for_non_finite_profile(self):
        bad = np.ones(N_MELS, dtype=np.float32)
        bad[10] = np.nan
        assert make_masker({"cls0": bad}).profile_for_class("cls0") is None


class TestSlotWeights:

    def test_loud_slot_gets_zero_at_full_strength(self):
        m = make_masker(make_profiles(), energy_weight=1.0)
        for name, slot in CLASSES.items():
            weights = m.slot_weights_for_class(name)
            assert weights[slot] == pytest.approx(0.0, abs=1e-12), name

    def test_weights_are_a_distribution(self):
        m = make_masker(make_profiles())
        for name in CLASSES:
            weights = m.slot_weights_for_class(name)
            assert weights.sum() == pytest.approx(1.0)
            assert (weights >= 0).all()

    def test_each_class_gets_its_own_distribution(self):
        """The defect this replaces: one global estimate shared by all classes."""
        m = make_masker(make_profiles())
        distinct = {tuple(np.round(m.slot_weights_for_class(n), 6))
                    for n in CLASSES}
        assert len(distinct) == len(CLASSES)

    def test_zero_weight_reduces_to_uniform(self):
        m = make_masker(make_profiles(), energy_weight=0.0)
        weights = m.slot_weights_for_class("cls0")
        assert weights == pytest.approx(np.ones(len(SLOTS)) / len(SLOTS))

    def test_partial_weight_keeps_loud_slot_eligible(self):
        m = make_masker(make_profiles(), energy_weight=0.5)
        weights = m.slot_weights_for_class("cls3")
        assert weights[3] > 0.0
        assert weights[3] < 1.0 / len(SLOTS)

    def test_none_when_unguided(self):
        assert make_masker(None).slot_weights_for_class("cls0") is None


class TestMasking:

    def test_loud_slot_never_masked_on_positives(self):
        m = make_masker(make_profiles(), energy_weight=1.0)
        for name, slot in CLASSES.items():
            a, b = SLOTS[slot]
            for _ in range(50):
                bins = masked_bins(m(positive(slot), name))
                assert not ((bins >= a) & (bins < b)).any(), name

    def test_loud_slot_never_masked_on_negatives(self):
        """
        Half of all masker calls are on negative segments, which contain no
        target sound. A per-segment or global estimate spares an arbitrary noise
        band there; a per-class profile still protects the class's band.
        """
        m = make_masker(make_profiles(), energy_weight=1.0)
        for name, slot in CLASSES.items():
            a, b = SLOTS[slot]
            for _ in range(50):
                bins = masked_bins(m(negative(), name))
                assert not ((bins >= a) & (bins < b)).any(), name

    def test_other_slots_are_masked(self):
        m = make_masker(make_profiles(), energy_weight=1.0)
        reached = set()
        for _ in range(300):
            reached.update(masked_bins(m(positive(0), "cls0")).tolist())
        assert len(reached) > N_MELS // 2

    def test_bands_respect_max_band_width(self):
        m = make_masker(make_profiles())
        for _ in range(200):
            bands = m._select_bands(positive(0), "cls0")
            assert all(e - s <= MAX_W for s, e in bands)

    def test_band_count_capped_by_slots(self):
        m = make_masker(make_profiles(), n_bands=99)
        bands = m._select_bands(positive(0), "cls0")
        assert len(bands) <= len(SLOTS)

    def test_mask_prob_gate(self):
        m = make_masker(make_profiles(), mask_prob=0.0)
        spec = positive(0)
        assert np.array_equal(m(spec, "cls0"), spec)

    def test_input_never_modified(self):
        m = make_masker(make_profiles())
        spec = positive(0)
        original = spec.copy()
        m(spec, "cls0")
        assert np.array_equal(spec, original)

    def test_shape_preserved(self):
        m = make_masker(make_profiles())
        for shape in [(30, N_MELS), (4, 30, N_MELS)]:
            spec = np.random.rand(*shape) + 1.0
            assert m(spec, "cls0").shape == shape


class TestFallback:

    def test_unknown_class_still_masks(self):
        m = make_masker(make_profiles())
        assert masked_bins(m(positive(0), "never_seen")).size > 0

    def test_no_profiles_still_masks(self):
        m = make_masker(None)
        assert masked_bins(m(positive(0), "cls0")).size > 0

    def test_fallback_is_not_confined(self):
        """Unguided masking must be able to reach the whole spectrum."""
        m = make_masker(None)
        reached = set()
        for _ in range(300):
            reached.update(masked_bins(m(positive(0), "cls0")).tolist())
        assert min(reached) < MAX_W
        assert max(reached) >= N_MELS - MAX_W

    def test_fallback_reported_once_per_class(self, capsys):
        m = make_masker(make_profiles())
        for _ in range(10):
            m(positive(0), "unknown_a")
            m(positive(0), "unknown_b")
        out = capsys.readouterr().out
        assert out.count("for 'unknown_a'") == 1
        assert out.count("for 'unknown_b'") == 1


class TestReseed:
    """Guards the per-worker RNG specialisation."""

    @staticmethod
    def _stream(masker, n=8):
        return [tuple(masked_bins(masker(positive(0), "cls0"))) for _ in range(n)]

    def test_same_seed_gives_same_stream(self):
        profiles = make_profiles()
        assert self._stream(make_masker(profiles)) == \
               self._stream(make_masker(profiles))

    def test_reseed_changes_the_stream(self):
        profiles = make_profiles()
        a = make_masker(profiles)
        a.reseed(1)
        b = make_masker(profiles)
        b.reseed(2)
        assert self._stream(a) != self._stream(b)

    def test_reseed_is_reproducible(self):
        profiles = make_profiles()
        a = make_masker(profiles)
        a.reseed(7)
        b = make_masker(profiles)
        b.reseed(7)
        assert self._stream(a) == self._stream(b)


class TestStrategiesAreSiblings:
    """
    The strategies must NOT inherit from one another.

    They used to form a Random -> Energy -> ActivationGuided chain, which
    asserted that activation-guided masking is-a energy masker. That is false as
    a domain claim, and it forced AFRI to be bolted on as a fourth weight inside
    the activation masker.
    """

    def test_no_inheritance_between_strategies(self):
        assert not issubclass(EnergyBasedFrequencyMasker, RandomFrequencyMasker)
        assert not issubclass(ActivationGuidedFrequencyMasker,
                              EnergyBasedFrequencyMasker)
        assert not issubclass(AFRIRangeMasker, ActivationGuidedFrequencyMasker)
        assert not issubclass(UniformSlotFrequencyMasker,
                              RandomFrequencyMasker)

    def test_all_share_one_call_interface(self):
        spec = np.random.rand(40, N_MELS) + 1.0
        for masker in (
            RandomFrequencyMasker(n_bands=2, mask_prob=1.0, n_mels=N_MELS),
            UniformSlotFrequencyMasker(n_bands=2, mask_prob=1.0, n_mels=N_MELS),
            EnergyBasedFrequencyMasker(n_bands=2, mask_prob=1.0, n_mels=N_MELS),
            ActivationGuidedFrequencyMasker(n_bands=2, mask_prob=1.0, n_mels=N_MELS),
            AFRIRangeMasker(n_bands=2, mask_prob=1.0, n_mels=N_MELS),
        ):
            out = masker(spec, "SOME_CLASS")
            assert out.shape == spec.shape
            assert np.any(out == 0.0)
            assert hasattr(masker, 'reseed')


class TestReprAndStats:

    def test_repr(self):
        text = repr(make_masker(make_profiles()))
        assert "EnergyBasedFrequencyMasker" in text
        assert "energy_weight" in text
        assert "n_profiles=6" in text

    def test_stats_without_profiles(self):
        stats = make_masker(None).get_energy_stats()
        assert stats['has_profiles'] is False
        assert stats['n_usable'] == 0

    def test_stats_counts_usable_profiles(self):
        profiles = make_profiles()
        profiles['broken'] = np.zeros(N_MELS, dtype=np.float32)
        stats = make_masker(profiles).get_energy_stats()
        assert stats['n_classes'] == len(CLASSES) + 1
        assert stats['n_usable'] == len(CLASSES)


def run_all_tests():
    """Manual driver, matching this suite's dual-mode convention."""
    import traceback

    groups = [TestConstruction, TestProfileLookup, TestSlotWeights, TestMasking,
              TestFallback, TestReseed, TestStrategiesAreSiblings,
              TestReprAndStats]
    passed = failed = skipped = 0
    for group in groups:
        print(f"\n{group.__name__}")
        instance = group()
        for name in sorted(n for n in dir(group) if n.startswith("test_")):
            method = getattr(instance, name)
            if method.__code__.co_argcount - 1 > 0:
                print(f"  SKIP {name} (needs pytest fixtures/params)")
                skipped += 1
                continue
            try:
                method()
                print(f"  PASS {name}")
                passed += 1
            except Exception:
                print(f"  FAIL {name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped "
          f"(run under pytest for the full suite)")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
