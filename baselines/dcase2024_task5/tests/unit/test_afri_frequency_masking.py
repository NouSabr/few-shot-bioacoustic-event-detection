"""
Unit tests for AFRIRangeMasker (AFRI range-based frequency masking).

AFRI masks a fixed area strictly INSIDE (direct) or strictly OUTSIDE (inverse)
the frequency range identified per class by
``src.utils.afri_core.compute_afri_profile`` (``auto_minbin``/``auto_maxbin``).

These tests replace an earlier suite written against the superseded
implementation, in which AFRI was a fourth weight inside
ActivationGuidedFrequencyMasker and direct/inverse were soft slot-lottery
weights derived from ``mask_weights``/``inverse_weights``. Those assertions
(band-averaged weights, the uniform floor, argmax placement) describe a
mechanism that no longer exists.

The invariant the suite is built around: for a given class, direct and inverse
mask an identical number of bins on disjoint regions, so the pair is an exactly
matched comparison.

Run:
    pytest tests/unit/test_afri_frequency_masking.py -v
    python tests/unit/test_afri_frequency_masking.py
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
        AFRI_AREA_BUDGET,
        AFRIRangeMasker,
    )
except ImportError:  # pragma: no cover - keeps the suite runnable with numpy alone
    _spec = importlib.util.spec_from_file_location(
        "frequency_masker",
        os.path.join(_REPO, "src", "datamodules", "components",
                     "frequency_masker.py"),
    )
    _fm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fm)
    AFRI_AREA_BUDGET = _fm.AFRI_AREA_BUDGET
    AFRIRangeMasker = _fm.AFRIRangeMasker


N_MELS = 128
PARAMS = dict(n_bands=4, mask_prob=1.0, max_band_width=24, n_mels=N_MELS,
              seed=1234)

REAL_CACHE = os.path.join(_REPO, ".afri_cache", "profiles.npz")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_profile(lo, hi, valid=True, n_mels=N_MELS):
    """A profile shaped like compute_afri_profile's output."""
    weights = np.full(n_mels, 0.1, dtype=np.float32)
    weights[lo:hi] = 0.9
    return {
        'raw_profile': weights.copy(),
        'mask_weights': weights,
        'inverse_weights': (1.0 - weights).astype(np.float32),
        'auto_minbin': lo,
        'auto_maxbin': hi,
        'valid': valid,
    }


def spec(t=30, n_mels=N_MELS, seed=0):
    return np.random.RandomState(seed).rand(t, n_mels) + 1.0


def masked_bins(out):
    """Indices of bins zeroed across all time frames."""
    return np.flatnonzero((out == 0.0).all(axis=0))


def make_masker(profiles, mode="direct", **kwargs):
    params = dict(PARAMS)
    params.update(kwargs)
    return AFRIRangeMasker(**params, afri_profiles=profiles, afri_mode=mode)


# ----------------------------------------------------------------------

class TestRangeLookup:
    """range_for_class is the single source of truth for the eligible region."""

    def test_range_read_from_profile(self):
        m = make_masker({'A': make_profile(40, 70)})
        assert m.range_for_class('A') == (40, 70)

    def test_no_profiles_attached_returns_none(self):
        m = make_masker(None)
        assert m.range_for_class('A') is None

    def test_missing_class_returns_none(self):
        m = make_masker({'A': make_profile(40, 70)})
        assert m.range_for_class('B') is None

    def test_none_class_name_returns_none(self):
        m = make_masker({'A': make_profile(40, 70)})
        assert m.range_for_class(None) is None

    def test_invalid_profile_returns_none(self):
        """Uniform fallback profiles carry valid=False and must not guide."""
        m = make_masker({'A': make_profile(0, N_MELS - 1, valid=False)})
        assert m.range_for_class('A') is None

    def test_degenerate_range_returns_none(self):
        m = make_masker({'A': make_profile(50, 50)})
        assert m.range_for_class('A') is None

    def test_range_clipped_to_n_mels(self):
        prof = make_profile(10, 40)
        prof['auto_maxbin'] = 999
        prof['auto_minbin'] = -5
        m = make_masker({'A': prof})
        lo, hi = m.range_for_class('A')
        assert (lo, hi) == (0, N_MELS)

    def test_missing_range_keys_returns_none(self):
        prof = make_profile(10, 40)
        del prof['auto_minbin']
        m = make_masker({'A': prof})
        assert m.range_for_class('A') is None


class TestMaskedArea:
    """M = min(area_budget, W, n_mels - W), identical in both modes."""

    @pytest.mark.parametrize("lo,hi", [(40, 70), (2, 3), (10, 120), (0, 44)])
    def test_area_formula(self, lo, hi):
        profiles = {'A': make_profile(lo, hi)}
        m = make_masker(profiles)
        width = hi - lo
        expected = min(AFRI_AREA_BUDGET, width, N_MELS - width)
        assert m.masked_area_for_class('A') == expected

    def test_area_paired_between_modes(self):
        """The whole point: direct and inverse mask the same count per class."""
        profiles = {f'c{i}': make_profile(lo, hi) for i, (lo, hi) in
                    enumerate([(40, 70), (2, 3), (10, 120), (0, 44), (60, 61)])}
        direct = make_masker(profiles, "direct")
        inverse = make_masker(profiles, "inverse")
        for cls in profiles:
            assert (direct.masked_area_for_class(cls) ==
                    inverse.masked_area_for_class(cls))

    def test_narrow_range_gives_area_one(self):
        """c_17 in the real cache is [2,3]: degenerate but must not crash."""
        m = make_masker({'A': make_profile(2, 3)})
        assert m.masked_area_for_class('A') == 1

    def test_wide_range_capped_by_complement(self):
        """A range wider than 128-budget is limited by what inverse can place."""
        m = make_masker({'A': make_profile(2, 125)})
        assert m.masked_area_for_class('A') == N_MELS - 123

    def test_area_none_when_unguided(self):
        m = make_masker({'A': make_profile(0, 0)})
        assert m.masked_area_for_class('A') is None


class TestDirectMode:

    def test_masks_only_inside_the_range(self):
        lo, hi = 40, 70
        m = make_masker({'A': make_profile(lo, hi)}, "direct")
        for _ in range(200):
            bins = masked_bins(m(spec(), 'A'))
            assert bins.size > 0
            assert bins.min() >= lo and bins.max() < hi

    def test_union_area_is_exactly_M(self):
        profiles = {'A': make_profile(40, 70), 'B': make_profile(10, 120),
                    'C': make_profile(2, 3)}
        m = make_masker(profiles, "direct")
        for cls in profiles:
            area = m.masked_area_for_class(cls)
            for _ in range(100):
                assert masked_bins(m(spec(), cls)).size == area

    def test_layout_varies_across_calls(self):
        """Region is fixed; the bands inside it are resampled every call."""
        m = make_masker({'A': make_profile(20, 100)}, "direct")
        seen = {tuple(masked_bins(m(spec(), 'A'))) for _ in range(50)}
        assert len(seen) > 1, "band layout should not be deterministic"

    def test_bands_respect_max_band_width(self):
        m = make_masker({'A': make_profile(0, 120)}, "direct")
        for _ in range(100):
            bins = masked_bins(m(spec(), 'A'))
            # Longest contiguous run must not exceed max_band_width
            runs = np.diff(np.flatnonzero(
                np.diff(np.concatenate(([0], np.isin(np.arange(N_MELS), bins)
                                        .astype(int), [0]))) != 0))[::2]
            assert runs.max() <= PARAMS['max_band_width']


class TestInverseMode:

    def test_masks_nothing_inside_the_range(self):
        lo, hi = 40, 70
        m = make_masker({'A': make_profile(lo, hi)}, "inverse")
        for _ in range(200):
            bins = masked_bins(m(spec(), 'A'))
            assert bins.size > 0
            assert not ((bins >= lo) & (bins < hi)).any()

    def test_never_straddles_the_gap(self):
        """
        A band spanning the excluded range would mask inside it. Every
        contiguous run must sit entirely below lo or entirely at/above hi.
        """
        lo, hi = 40, 70
        m = make_masker({'A': make_profile(lo, hi)}, "inverse")
        for _ in range(200):
            bins = masked_bins(m(spec(), 'A'))
            if bins.size == 0:
                continue
            splits = np.split(bins, np.flatnonzero(np.diff(bins) != 1) + 1)
            for run in splits:
                assert run[-1] < lo or run[0] >= hi

    def test_union_area_is_exactly_M(self):
        profiles = {'A': make_profile(40, 70), 'B': make_profile(2, 125),
                    'C': make_profile(2, 3)}
        m = make_masker(profiles, "inverse")
        for cls in profiles:
            area = m.masked_area_for_class(cls)
            for _ in range(100):
                assert masked_bins(m(spec(), cls)).size == area

    def test_disjoint_from_direct(self):
        lo, hi = 40, 70
        profiles = {'A': make_profile(lo, hi)}
        d = make_masker(profiles, "direct")
        i = make_masker(profiles, "inverse")
        d_bins = set().union(*[set(masked_bins(d(spec(), 'A'))) for _ in range(50)])
        i_bins = set().union(*[set(masked_bins(i(spec(), 'A'))) for _ in range(50)])
        assert not (d_bins & i_bins), "the two modes must have disjoint support"


class TestFallback:
    """Classes without a usable range get unguided uniform slot masking."""

    def test_invalid_profile_still_masks(self):
        m = make_masker({'A': make_profile(0, N_MELS - 1, valid=False)})
        assert masked_bins(m(spec(), 'A')).size > 0

    def test_no_profiles_still_masks(self):
        m = make_masker(None)
        assert masked_bins(m(spec(), 'A')).size > 0

    def test_fallback_reported_once_per_class(self, capsys):
        m = make_masker({'A': make_profile(0, 0), 'B': make_profile(0, 0)})
        for _ in range(10):
            m(spec(), 'A')
            m(spec(), 'B')
        out = capsys.readouterr().out
        assert out.count("for 'A'") == 1
        assert out.count("for 'B'") == 1

    def test_fallback_is_not_confined_to_any_range(self):
        """Unguided masking must be able to reach the whole spectrum."""
        m = make_masker({'A': make_profile(0, 0)})
        reached = set()
        for _ in range(300):
            reached.update(masked_bins(m(spec(), 'A')).tolist())
        assert min(reached) < 24 and max(reached) >= N_MELS - 24


class TestValidation:

    def test_invalid_mode_raises(self):
        with pytest.raises(AssertionError, match="afri_mode"):
            make_masker(None, "sideways")

    def test_invalid_area_budget_raises(self):
        with pytest.raises(AssertionError, match="area_budget"):
            make_masker(None, "direct", area_budget=0)

    def test_default_mode_is_direct(self):
        m = AFRIRangeMasker(n_mels=N_MELS, afri_profiles=None)
        assert m.afri_mode == "direct"

    def test_mask_prob_gate(self):
        m = make_masker({'A': make_profile(40, 70)}, mask_prob=0.0)
        s = spec()
        out = m(s, 'A')
        assert np.array_equal(out, s)

    def test_input_never_modified(self):
        m = make_masker({'A': make_profile(40, 70)})
        s = spec()
        original = s.copy()
        m(s, 'A')
        assert np.array_equal(s, original)

    def test_custom_area_budget_is_honoured(self):
        m = make_masker({'A': make_profile(10, 120)}, "direct", area_budget=8)
        assert m.masked_area_for_class('A') == 8
        assert masked_bins(m(spec(), 'A')).size == 8


class TestReprAndStats:

    def test_repr_names_mode_and_budget(self):
        m = make_masker({'A': make_profile(40, 70)}, "inverse")
        text = repr(m)
        assert "AFRIRangeMasker" in text
        assert "afri_mode=inverse" in text
        assert f"area_budget={AFRI_AREA_BUDGET}" in text

    def test_stats_without_profiles(self):
        stats = make_masker(None).get_afri_stats()
        assert stats['has_profiles'] is False
        assert stats['n_valid'] == 0

    def test_stats_counts_guided_and_fallback(self):
        profiles = {
            'good1': make_profile(40, 70),
            'good2': make_profile(10, 60),
            'bad': make_profile(0, N_MELS - 1, valid=False),
        }
        stats = make_masker(profiles).get_afri_stats()
        assert stats['n_classes'] == 3
        assert stats['n_valid'] == 2
        assert stats['mean_masked_area'] > 0


@pytest.mark.skipif(not os.path.exists(REAL_CACHE),
                    reason="no AFRI profile cache in the repo")
class TestRealProfileCache:
    """
    End-to-end intent check against the profiles actually used for training.

    The superseded implementation scored 0.49 (direct) and 0.30 (inverse) on
    "fraction of masked bins inside the range", where the intent is 1.00 and
    0.00. These assertions are what keep that regression from returning.
    """

    @staticmethod
    def _profiles():
        z = np.load(REAL_CACHE, allow_pickle=True)
        return {k: z[k].item() for k in z.keys()}

    def test_direct_masks_only_inside_every_class_range(self):
        profiles = self._profiles()
        m = make_masker(profiles, "direct")
        for cls in profiles:
            rng_pair = m.range_for_class(cls)
            if rng_pair is None:
                continue
            lo, hi = rng_pair
            for _ in range(20):
                bins = masked_bins(m(spec(), cls))
                assert bins.min() >= lo and bins.max() < hi, cls

    def test_inverse_masks_nothing_inside_any_class_range(self):
        profiles = self._profiles()
        m = make_masker(profiles, "inverse")
        for cls in profiles:
            rng_pair = m.range_for_class(cls)
            if rng_pair is None:
                continue
            lo, hi = rng_pair
            for _ in range(20):
                bins = masked_bins(m(spec(), cls))
                assert not ((bins >= lo) & (bins < hi)).any(), cls

    def test_areas_paired_across_modes_for_every_class(self):
        profiles = self._profiles()
        d = make_masker(profiles, "direct")
        i = make_masker(profiles, "inverse")
        for cls in profiles:
            assert d.masked_area_for_class(cls) == i.masked_area_for_class(cls), cls

    def test_every_class_is_either_guided_or_falls_back(self):
        profiles = self._profiles()
        m = make_masker(profiles, "direct")
        for cls in profiles:
            assert masked_bins(m(spec(), cls)).size > 0, cls


def run_all_tests():
    """Manual driver, matching this suite's existing dual-mode convention."""
    import traceback

    groups = [
        TestRangeLookup, TestMaskedArea, TestDirectMode, TestInverseMode,
        TestFallback, TestValidation, TestReprAndStats,
    ]
    if os.path.exists(REAL_CACHE):
        groups.append(TestRealProfileCache)

    passed = failed = skipped = 0
    for group in groups:
        print(f"\n{group.__name__}")
        instance = group()
        for name in sorted(n for n in dir(group) if n.startswith("test_")):
            method = getattr(instance, name)
            # parametrized and capsys-dependent tests need pytest
            argcount = method.__code__.co_argcount - 1
            if argcount > 0:
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
