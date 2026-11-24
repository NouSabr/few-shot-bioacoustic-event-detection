"""
Unit tests for Phase 3: per-class activation-guided frequency masking.

Covers FrequencyImportanceTracker and ActivationGuidedFrequencyMasker.

These tests replace a suite written against the superseded implementation, in
which the tracker kept ONE global importance vector updated directly inside a
forward hook. A hook is label-blind, so that design could only produce a single
profile shared by every class -- and measurement confirmed Phase 3's slot
distribution was then completely independent of the input: spectrograms with
energy in entirely different slots produced identical per-slot mask rates.

The hook now only captures activations; attribution happens in the training step
where the class names are known. The central assertions below are therefore that
different classes get different profiles, and that a class the tracker has not
seen falls back to unguided masking rather than borrowing another class's signal.

Run:
    pytest tests/unit/test_phase3_frequency_masking.py -v
    python tests/unit/test_phase3_frequency_masking.py
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _REPO)

from src.datamodules.components.frequency_masker import (  # noqa: E402
    ActivationGuidedFrequencyMasker,
)
from src.utils.frequency_importance_tracker import (  # noqa: E402
    FrequencyImportanceTracker,
)

N_MELS = 128
MAX_W = 24
SLOTS = [(i * MAX_W, min((i + 1) * MAX_W, N_MELS))
         for i in range(int(np.ceil(N_MELS / MAX_W)))]
PARAMS = dict(n_bands=4, mask_prob=1.0, max_band_width=MAX_W, n_mels=N_MELS,
              seed=1234)
CLASSES = {f"cls{i}": i for i in range(len(SLOTS))}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

class _StubEncoder(nn.Module):
    """Minimal module exposing a `layer2` for the capture hook."""

    def __init__(self):
        super().__init__()
        self.layer2 = nn.Identity()

    def forward(self, x):
        return self.layer2(x)


def activations(peaks, freq=N_MELS, channels=4, time=5):
    """
    (rows, channels, time, freq) activations peaked at the given freq index
    per row. `peaks[i]` is the peak bin for row i.
    """
    out = torch.full((len(peaks), channels, time, freq), 0.05)
    for row, peak in enumerate(peaks):
        out[row, :, :, peak] = 1.0
    return out


def spec(t=30, seed=0):
    return np.random.RandomState(seed).rand(t, N_MELS) + 1.0


def masked_bins(out):
    return np.flatnonzero((out == 0.0).all(axis=0))


def make_tracker(ema_alpha=0.9):
    return FrequencyImportanceTracker(n_mels=N_MELS, ema_alpha=ema_alpha)


def per_class_tracker():
    """A tracker where each class peaks in the middle of its own slot."""
    tracker = make_tracker(ema_alpha=0.0)   # 0.0 = take the newest value
    names = list(CLASSES)
    peaks = [(SLOTS[CLASSES[n]][0] + SLOTS[CLASSES[n]][1]) // 2 for n in names]
    tracker.update(activations(peaks), names)
    return tracker


# ----------------------------------------------------------------------

class TestTrackerConstruction:

    def test_initialization(self):
        tracker = make_tracker()
        assert tracker.n_mels == N_MELS
        assert tracker.ema_alpha == 0.9
        assert tracker.n_classes_tracked() == 0

    def test_invalid_ema_alpha(self):
        with pytest.raises(AssertionError, match="ema_alpha"):
            FrequencyImportanceTracker(n_mels=N_MELS, ema_alpha=1.5)

    def test_unknown_class_returns_none(self):
        assert make_tracker().get_importance("nope") is None

    def test_none_class_returns_none(self):
        assert make_tracker().get_importance(None) is None


class TestTrackerCapture:

    def test_attach_requires_layer2(self):
        with pytest.raises(AttributeError, match="layer2"):
            make_tracker().attach(nn.Identity())

    def test_hook_captures_forward_output(self):
        tracker = make_tracker()
        model = _StubEncoder()
        tracker.attach(model)
        batch = activations([10, 60])
        model(batch)
        assert tracker._last_activations is not None

    def test_update_from_last_consumes_the_capture(self):
        tracker = make_tracker(ema_alpha=0.0)
        model = _StubEncoder()
        tracker.attach(model)
        model(activations([10, 60]))
        tracker.update_from_last(["a", "b"])
        assert tracker.n_classes_tracked() == 2
        # A second call with nothing captured must be a no-op, not a repeat
        tracker.update_from_last(["a", "b"])
        assert tracker.total_updates == 1

    def test_remove_hook(self):
        tracker = make_tracker()
        model = _StubEncoder()
        tracker.attach(model)
        tracker.remove_hook()
        model(activations([10]))
        assert tracker._last_activations is None


class TestTrackerAttribution:

    def test_each_class_gets_its_own_profile(self):
        """The defect this replaces: one global profile for every class."""
        tracker = make_tracker(ema_alpha=0.0)
        tracker.update(activations([10, 70]), ["a", "b"])
        pa, pb = tracker.get_importance("a"), tracker.get_importance("b")
        assert not np.allclose(pa, pb)
        assert int(np.argmax(pa)) == 10
        assert int(np.argmax(pb)) == 70

    def test_rows_are_grouped_by_class(self):
        tracker = make_tracker(ema_alpha=0.0)
        # Two rows of class 'a' peaked at 20, one row of 'b' peaked at 100
        tracker.update(activations([20, 20, 100]), ["a", "a", "b"])
        assert int(np.argmax(tracker.get_importance("a"))) == 20
        assert int(np.argmax(tracker.get_importance("b"))) == 100

    def test_extra_rows_are_ignored(self):
        """
        The training batch is [positives; negatives] but class_name covers only
        the positives, so the negatives must not contribute.
        """
        tracker = make_tracker(ema_alpha=0.0)
        tracker.update(activations([30, 30, 90, 90]), ["a", "a"])
        assert int(np.argmax(tracker.get_importance("a"))) == 30
        assert tracker.n_classes_tracked() == 1

    def test_profile_is_normalised(self):
        tracker = make_tracker(ema_alpha=0.0)
        tracker.update(activations([40]) * 1000.0, ["a"])
        profile = tracker.get_importance("a")
        assert profile.max() == pytest.approx(1.0)
        assert profile.min() >= 0.0

    def test_resampled_to_n_mels(self):
        """Layer 2 is downsampled along frequency; profiles must be mel-length."""
        tracker = make_tracker(ema_alpha=0.0)
        tracker.update(activations([16], freq=N_MELS // 2), ["a"])
        assert tracker.get_importance("a").shape == (N_MELS,)

    def test_ema_momentum(self):
        tracker = make_tracker(ema_alpha=0.75)
        tracker.update(activations([10]), ["a"])
        first = tracker.get_importance("a").copy()
        tracker.update(activations([100]), ["a"])
        second = tracker.get_importance("a")
        # Old peak retained but decayed; new peak present but attenuated
        assert second[10] < first[10]
        assert second[100] > first[100]
        assert int(np.argmax(second)) == 10, "high momentum should resist the shift"

    def test_returned_profile_is_a_copy(self):
        tracker = make_tracker(ema_alpha=0.0)
        tracker.update(activations([10]), ["a"])
        profile = tracker.get_importance("a")
        profile[:] = 0.0
        assert tracker.get_importance("a").max() > 0.0

    def test_malformed_input_is_ignored(self):
        tracker = make_tracker()
        tracker.update(None, ["a"])
        tracker.update(torch.zeros(2, 3), ["a"])          # not 4-D
        tracker.update(activations([10]), [])
        assert tracker.n_classes_tracked() == 0

    def test_reset(self):
        tracker = per_class_tracker()
        assert tracker.n_classes_tracked() > 0
        tracker.reset()
        assert tracker.n_classes_tracked() == 0
        assert tracker.total_updates == 0


class TestTrackerStats:

    def test_stats_when_empty(self):
        stats = make_tracker().get_stats()
        assert stats['n_classes_tracked'] == 0
        assert stats['mean_spread'] == 0.0

    def test_stats_reports_spread(self):
        """
        Spread is the metric that says whether Phase 3 steers anything: a flat
        profile cannot, however many classes are tracked.
        """
        peaked = per_class_tracker().get_stats()
        assert peaked['n_classes_tracked'] == len(CLASSES)
        assert peaked['mean_spread'] > 0.5

        flat = make_tracker(ema_alpha=0.0)
        flat.update(torch.ones(1, 2, 3, N_MELS), ["a"])
        assert flat.get_stats()['mean_spread'] == pytest.approx(0.0, abs=1e-9)


class TestActivationMasker:

    def test_targets_the_class_own_slot(self):
        masker = ActivationGuidedFrequencyMasker(
            **PARAMS, importance_weight=1.0, importance_tracker=per_class_tracker())
        for name, slot in CLASSES.items():
            hist = np.zeros(N_MELS)
            for _ in range(400):
                hist += (masker(spec(), name) == 0.0).all(axis=0)
            hist /= 400
            rates = [hist[a:b].mean() for a, b in SLOTS]
            others = [r for j, r in enumerate(rates) if j != slot]
            assert rates[slot] > max(others), name

    def test_each_class_gets_its_own_distribution(self):
        masker = ActivationGuidedFrequencyMasker(
            **PARAMS, importance_weight=1.0, importance_tracker=per_class_tracker())
        distinct = {tuple(np.round(masker.slot_weights_for_class(n), 6))
                    for n in CLASSES}
        assert len(distinct) == len(CLASSES)

    def test_zero_weight_reduces_to_uniform(self):
        masker = ActivationGuidedFrequencyMasker(
            **PARAMS, importance_weight=0.0, importance_tracker=per_class_tracker())
        weights = masker.slot_weights_for_class("cls0")
        assert weights == pytest.approx(np.ones(len(SLOTS)) / len(SLOTS))

    def test_no_tracker_falls_back(self):
        masker = ActivationGuidedFrequencyMasker(
            **PARAMS, importance_weight=1.0, importance_tracker=None)
        assert masker.slot_weights_for_class("cls0") is None
        assert masked_bins(masker(spec(), "cls0")).size > 0

    def test_untracked_class_falls_back(self):
        masker = ActivationGuidedFrequencyMasker(
            **PARAMS, importance_weight=1.0, importance_tracker=per_class_tracker())
        assert masker.slot_weights_for_class("never_seen") is None
        assert masked_bins(masker(spec(), "never_seen")).size > 0

    def test_fallback_reported_once_per_class(self, capsys):
        masker = ActivationGuidedFrequencyMasker(
            **PARAMS, importance_weight=1.0, importance_tracker=None)
        for _ in range(10):
            masker(spec(), "unknown_a")
            masker(spec(), "unknown_b")
        out = capsys.readouterr().out
        assert out.count("for 'unknown_a'") == 1
        assert out.count("for 'unknown_b'") == 1

    def test_invalid_importance_weight(self):
        with pytest.raises(AssertionError, match="importance_weight"):
            ActivationGuidedFrequencyMasker(**PARAMS, importance_weight=1.5)

    def test_no_energy_parameters(self):
        """Phase 3 no longer carries the unused energy blend."""
        masker = ActivationGuidedFrequencyMasker(**PARAMS)
        for attr in ('energy_weight', 'ema_alpha', 'band_energy_ema'):
            assert not hasattr(masker, attr), f"{attr} should no longer exist"

    def test_mask_prob_gate(self):
        params = dict(PARAMS, mask_prob=0.0)
        masker = ActivationGuidedFrequencyMasker(
            **params, importance_tracker=per_class_tracker())
        s = spec()
        assert np.array_equal(masker(s, "cls0"), s)

    def test_input_never_modified(self):
        masker = ActivationGuidedFrequencyMasker(
            **PARAMS, importance_tracker=per_class_tracker())
        s = spec()
        original = s.copy()
        masker(s, "cls0")
        assert np.array_equal(s, original)

    def test_repr_and_stats(self):
        masker = ActivationGuidedFrequencyMasker(
            **PARAMS, importance_weight=1.0, importance_tracker=per_class_tracker())
        assert "ActivationGuidedFrequencyMasker" in repr(masker)
        stats = masker.get_importance_stats()
        assert stats['has_tracker'] is True
        assert stats['tracker_stats']['n_classes_tracked'] == len(CLASSES)


class TestIntegration:

    def test_live_update_changes_the_mask(self):
        """
        At num_workers=0 the masker and the tracker are the same objects in one
        process, so a training-step update is visible on the next __getitem__.
        """
        tracker = make_tracker(ema_alpha=0.0)
        masker = ActivationGuidedFrequencyMasker(
            **PARAMS, importance_weight=1.0, importance_tracker=tracker)

        assert masker.slot_weights_for_class("a") is None      # unseen yet

        tracker.update(activations([10]), ["a"])
        early = masker.slot_weights_for_class("a")
        assert int(np.argmax(early)) == 0

        tracker.update(activations([110]), ["a"])
        later = masker.slot_weights_for_class("a")
        assert int(np.argmax(later)) == 4
        assert not np.allclose(early, later)

    def test_full_capture_to_mask_path(self):
        tracker = make_tracker(ema_alpha=0.0)
        model = _StubEncoder()
        tracker.attach(model)
        masker = ActivationGuidedFrequencyMasker(
            **PARAMS, importance_weight=1.0, importance_tracker=tracker)

        # forward -> hook captures -> step attributes -> masker uses it
        model(activations([100, 100]))
        tracker.update_from_last(["cls4", "cls4"])

        hist = np.zeros(N_MELS)
        for _ in range(400):
            hist += (masker(spec(), "cls4") == 0.0).all(axis=0)
        rates = [(hist / 400)[a:b].mean() for a, b in SLOTS]
        assert int(np.argmax(rates)) == 4


def run_all_tests():
    """Manual driver, matching this suite's dual-mode convention."""
    import traceback

    groups = [TestTrackerConstruction, TestTrackerCapture, TestTrackerAttribution,
              TestTrackerStats, TestActivationMasker, TestIntegration]
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
