"""
Frequency Masking Augmentation for Bioacoustic Few-Shot Learning

Five independent masking strategies, selected by ``train_param.freq_mask_mode``.
They are siblings, not a chain: no strategy extends another, and the shared
mechanics live in ``_BaseFrequencyMasker``.

    'random'        Phase 1  RandomFrequencyMasker
                    Uniformly random band widths and positions over the whole
                    frequency axis. No spectral information. SpecAugment.

    'slot_uniform'  control  UniformSlotFrequencyMasker
                    The slot sampler with P(b) uniform. This is the correct
                    baseline for the three guided strategies below: it holds the
                    placement rule fixed and turns off only the guidance, which
                    'random' does not (it uses free placement).

    'energy'        Phase 2  EnergyBasedFrequencyMasker
                    Slot lottery weighted away from the slots where the CLASS is
                    loud, from a per-class profile precomputed over its first
                    N positive segments.

    'activation'    Phase 3  ActivationGuidedFrequencyMasker
                    Slot lottery weighted towards the slots the encoder
                    activates on for that CLASS.

    'afri'                   AFRIRangeMasker
                    Masks a fixed area strictly inside (direct) or strictly
                    outside (inverse) the class's AFRI frequency range.

Three of these are per-class by construction, and they use the same five
positive segments per class, so the progression varies only the signal:

    energy       mean spectrogram energy of those segments
    activation   encoder activation magnitude for that class
    afri         positive-minus-negative contrast on those segments

Energy is a property of the DATA, so it is precomputed once and cached; nothing
about it changes during training. Activation importance depends on the evolving
encoder, so it alone is tracked online. That asymmetry is deliberate.

See docs/dynamic_freq_masking/ for the design rationale and
09_WIRING_VERIFICATION.md for the measured behaviour of each strategy.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple


# Default masked area (in mel bins) for AFRIRangeMasker. Calibrated to the
# *union* area the slot-based strategies actually mask, which is smaller than
# the naive n_bands x mean_width because their bands may overlap: the measured
# per-bin mask rate for uniform slot selection is 0.276 at mask_prob=0.8,
# i.e. 0.276 / 0.8 * 128 ~= 44 bins conditional on masking.
AFRI_AREA_BUDGET = 44


class _BaseFrequencyMasker:
    """
    Shared construction, validation, RNG and band application.

    Holds no strategy of its own. Subclasses implement ``_select_bands`` and
    inherit the ``mask_prob`` gate and the zeroing step, so every strategy
    consumes its RNG in the same order: one ``rand()`` for the gate, then
    whatever ``_select_bands`` draws.
    """

    def __init__(
        self,
        n_bands: int = 2,
        mask_prob: float = 0.5,
        max_band_width: int = 16,
        n_mels: int = 128,
        seed: int = 1234
    ):
        """
        Args:
            n_bands: Number of frequency bands to mask per spectrogram.
            mask_prob: Probability of applying masking (0.0-1.0).
            max_band_width: Maximum width of each masked band in mel bins.
                           Also sets the slot width for the slot strategies.
            n_mels: Total number of mel frequency bins. Must match the feature
                   extraction configuration.
            seed: Base seed for the dedicated masking RNG, kept separate from
                 the global RNG so masking is independent of data sampling.
                 See reseed() for how dataloader workers specialise it.
        """
        self.n_bands = n_bands
        self.mask_prob = mask_prob
        self.max_band_width = max_band_width
        self.n_mels = n_mels
        self.seed = seed

        self.rng = np.random.RandomState(seed)

        # Classes already reported as falling back, so warnings are once-only
        self._fallback_reported = set()

        assert 0.0 <= mask_prob <= 1.0, (
            f"mask_prob must be in [0, 1], got {mask_prob}"
        )
        assert n_bands >= 0, (
            f"n_bands must be non-negative, got {n_bands}"
        )
        assert max_band_width > 0, (
            f"max_band_width must be positive, got {max_band_width}"
        )
        assert max_band_width <= n_mels, (
            f"max_band_width ({max_band_width}) cannot exceed n_mels ({n_mels})"
        )

    def reseed(self, offset: int) -> None:
        """
        Specialise the masking RNG for one dataloader worker and epoch.

        Without this every forked worker starts from the same ``seed``, so the
        workers draw IDENTICAL mask sequences and, because the parent's RNG never
        advances (masking only runs in workers), every epoch replays that same
        sequence. Callers pass ``torch.utils.data.get_worker_info().seed``, which
        already varies by worker and by epoch while staying reproducible from the
        global seed.

        Args:
            offset: Per-worker, per-epoch value to fold into the base seed.
        """
        self.rng = np.random.RandomState((int(self.seed) + int(offset)) % (2 ** 32))

    def _report_fallback(self, class_name: Optional[str], reason: str) -> None:
        """Warn once per class that this class is masked without guidance."""
        key = class_name if class_name is not None else "<no class>"
        if key not in self._fallback_reported:
            self._fallback_reported.add(key)
            print(f"{type(self).__name__}: {reason} for '{key}' -- masking this "
                  f"class with unguided uniform slot selection")

    @property
    def n_slots(self) -> int:
        """Number of contiguous frequency slots on the grid."""
        return int(np.ceil(self.n_mels / self.max_band_width))

    def slot_edges(self) -> List[Tuple[int, int]]:
        """
        Half-open (start, end) bin ranges of each slot.

        Note the last slot is narrower than the others whenever n_mels is not
        divisible by max_band_width (e.g. 128 / 24 leaves an 8-bin tail slot),
        so a slot is not a uniform unit of masking.
        """
        return [
            (i * self.max_band_width,
             min((i + 1) * self.max_band_width, self.n_mels))
            for i in range(self.n_slots)
        ]

    def _slot_means(self, per_bin: np.ndarray) -> np.ndarray:
        """Average a per-bin quantity within each slot."""
        return np.array([per_bin[a:b].mean() for a, b in self.slot_edges()])

    def _bands_from_slots(self, slot_indices: Sequence[int]) -> List[Tuple[int, int]]:
        """
        Turn chosen slot indices into concrete bands of random width and offset.

        Draws, per slot and in order, the width then the offset.
        """
        bands = []
        for slot_idx in slot_indices:
            slot_start = slot_idx * self.max_band_width
            slot_end = min((slot_idx + 1) * self.max_band_width, self.n_mels)

            available_width = slot_end - slot_start
            actual_width = self.rng.randint(
                1, min(self.max_band_width, available_width) + 1
            )

            if available_width > actual_width:
                offset = self.rng.randint(0, available_width - actual_width + 1)
            else:
                offset = 0

            start_freq = slot_start + offset
            bands.append((start_freq, start_freq + actual_width))
        return bands

    def _sample_slots(self, weights: np.ndarray) -> np.ndarray:
        """Draw min(n_bands, n_slots) slots without replacement under `weights`."""
        n_to_select = min(self.n_bands, len(weights))
        if n_to_select == 0:
            return np.array([], dtype=int)

        # A weight of exactly 0 makes a slot ineligible. Guard the case where
        # too few remain for sampling without replacement, which would raise.
        n_nonzero = int((weights > 0).sum())
        if n_nonzero < n_to_select:
            weights = weights + 1e-12
            weights = weights / weights.sum()

        return self.rng.choice(
            len(weights), size=n_to_select, replace=False, p=weights
        )

    def _uniform_slot_bands(self) -> List[Tuple[int, int]]:
        """Unguided slot masking: P(b) uniform. The guided strategies' fallback."""
        uniform = np.ones(self.n_slots) / self.n_slots
        return self._bands_from_slots(self._sample_slots(uniform))

    def _blend_with_uniform(self, weights: np.ndarray, strength: float) -> np.ndarray:
        """Mix guidance weights with uniform: strength 1.0 is pure guidance."""
        uniform = np.ones_like(weights) / len(weights)
        mixed = strength * weights + (1.0 - strength) * uniform
        total = mixed.sum()
        return mixed / total if total > 0 else uniform

    def _select_bands(self, spectrogram: np.ndarray,
                      class_name: Optional[str]) -> List[Tuple[int, int]]:
        raise NotImplementedError

    def __call__(self, spectrogram: np.ndarray,
                 class_name: Optional[str] = None) -> np.ndarray:
        """
        Apply frequency masking to a spectrogram.

        Args:
            spectrogram: (time, freq) or (batch, time, freq). The frequency
                        axis is last and must have n_mels bins.
            class_name: Class of this segment. Required by the per-class
                       strategies; ignored by 'random' and 'slot_uniform'.

        Returns:
            A copy of the input with the selected frequency bands set to 0.0.
            The input is never modified. If the mask_prob draw fails, an
            unmodified copy is returned.
        """
        if self.rng.rand() > self.mask_prob:
            return spectrogram.copy()

        masked = spectrogram.copy()
        for start_freq, end_freq in self._select_bands(spectrogram, class_name):
            masked[..., start_freq:end_freq] = 0.0
        return masked


class RandomFrequencyMasker(_BaseFrequencyMasker):
    """
    Phase 1: random frequency masking (SpecAugment-style).

    Each band takes a uniformly random width in [1, max_band_width] and a
    uniformly random start position over the whole frequency axis, drawn
    independently so bands may overlap. Uses no spectral information.

    Because start positions are free rather than slot-aligned, coverage is not
    uniform across frequency: a band covering bin 0 must start exactly at 0, so
    edge bins are masked far less often than centre bins (measured 0.025 at bin 0
    against 0.30 mid-spectrum at n_bands=4, max_band_width=24). The slot
    strategies do not share this bias, which is why 'slot_uniform' rather than
    'random' is their control.
    """

    def _select_bands(self, spectrogram: np.ndarray,
                      class_name: Optional[str]) -> List[Tuple[int, int]]:
        bands = []
        for _ in range(self.n_bands):
            band_width = self.rng.randint(1, self.max_band_width + 1)

            max_start = self.n_mels - band_width
            if max_start < 0:
                # Unreachable given the max_band_width <= n_mels assertion.
                continue

            start_freq = self.rng.randint(0, max_start + 1)
            bands.append((start_freq, start_freq + band_width))
        return bands

    def __repr__(self):
        return (
            f"RandomFrequencyMasker("
            f"n_bands={self.n_bands}, "
            f"mask_prob={self.mask_prob}, "
            f"max_band_width={self.max_band_width}, "
            f"n_mels={self.n_mels})"
        )


class UniformSlotFrequencyMasker(_BaseFrequencyMasker):
    """
    Control: the slot sampler with P(b) uniform.

    Draws n_bands of the slot grid without replacement, every slot equally
    likely, then places a band of random width and offset inside each.

    This is the baseline the guided strategies should be compared against.
    'random' is not, because it differs from them in the placement rule as well
    as in the guidance, and that difference is not neutral (see
    RandomFrequencyMasker). Comparing a guided strategy against this one holds
    the placement rule fixed and turns off only the guidance, so a null result
    means the signal did not help rather than that the geometry could not
    express it.
    """

    def _select_bands(self, spectrogram: np.ndarray,
                      class_name: Optional[str]) -> List[Tuple[int, int]]:
        return self._uniform_slot_bands()

    def __repr__(self):
        return (
            f"UniformSlotFrequencyMasker("
            f"n_bands={self.n_bands}, "
            f"mask_prob={self.mask_prob}, "
            f"max_band_width={self.max_band_width}, "
            f"n_mels={self.n_mels}, "
            f"n_slots={self.n_slots})"
        )


class EnergyBasedFrequencyMasker(_BaseFrequencyMasker):
    """
    Phase 2: energy-guided frequency masking, from a per-class profile.

    Slots are drawn with P(b) proportional to ``1 - normalised_energy(b)``,
    blended with uniform by ``energy_weight``. The slots where the class is loud
    -- likely the informative ones -- are the least likely to be masked, so
    augmentation is directed away from the class's dominant spectral content.

    The profile is **per class and precomputed**, from the same first N positive
    segments AFRI uses, so Phase 2 and AFRI see identical information and differ
    only in the transform (mean energy vs positive-minus-negative contrast).

    There is no online EMA. Spectrogram energy is a property of the data, not of
    the model, so there is nothing to estimate during training. The previous
    implementation kept one global running EMA, which under the interleaved
    episodic order converged to a mixture over the k_way classes rather than any
    one class's profile and left ~22% of segments with their informative band
    masked -- the opposite of the intent.

    At ``energy_weight = 1.0`` the loudest slot receives probability exactly 0
    and is never masked. Classes with no profile fall back to unguided uniform
    slot masking, reported once per class.
    """

    def __init__(
        self,
        n_bands: int = 2,
        mask_prob: float = 0.5,
        max_band_width: int = 16,
        n_mels: int = 128,
        energy_weight: float = 0.8,
        energy_profiles: Optional[Dict[str, np.ndarray]] = None,
        seed: int = 1234
    ):
        """
        Args:
            energy_weight: Guidance strength. 0.0 reduces to uniform slots,
                          1.0 is pure energy guidance.
            energy_profiles: Class name -> per-bin mean energy, shape (n_mels,).
                            Set after build_meta() completes; None until then.
        """
        super().__init__(n_bands, mask_prob, max_band_width, n_mels, seed)
        self.energy_weight = energy_weight
        self.energy_profiles = energy_profiles

        assert 0.0 <= energy_weight <= 1.0, (
            f"energy_weight must be in [0, 1], got {energy_weight}"
        )

    def profile_for_class(self, class_name: Optional[str]) -> Optional[np.ndarray]:
        """Per-bin energy profile for a class, or None to fall back."""
        if self.energy_profiles is None or class_name is None:
            return None
        profile = self.energy_profiles.get(class_name)
        if profile is None:
            return None
        profile = np.asarray(profile, dtype=float)
        if profile.ndim != 1 or profile.shape[0] != self.n_mels:
            return None
        if not np.isfinite(profile).all() or profile.max() <= 0:
            return None
        return profile

    def slot_weights_for_class(self, class_name: Optional[str]) -> Optional[np.ndarray]:
        """
        P(b) for a class: high energy -> low weight.

        Returns None when the class has no usable profile.
        """
        profile = self.profile_for_class(class_name)
        if profile is None:
            return None

        slot_energy = self._slot_means(profile)
        if slot_energy.max() <= 0:
            return None

        weights = 1.0 - slot_energy / slot_energy.max()
        total = weights.sum()
        if total <= 0:
            return None
        return self._blend_with_uniform(weights / total, self.energy_weight)

    def _select_bands(self, spectrogram: np.ndarray,
                      class_name: Optional[str]) -> List[Tuple[int, int]]:
        weights = self.slot_weights_for_class(class_name)
        if weights is None:
            self._report_fallback(class_name, "no energy profile")
            return self._uniform_slot_bands()
        return self._bands_from_slots(self._sample_slots(weights))

    def __repr__(self):
        n = len(self.energy_profiles) if self.energy_profiles is not None else 0
        return (
            f"EnergyBasedFrequencyMasker("
            f"n_bands={self.n_bands}, "
            f"mask_prob={self.mask_prob}, "
            f"max_band_width={self.max_band_width}, "
            f"n_mels={self.n_mels}, "
            f"energy_weight={self.energy_weight}, "
            f"n_profiles={n})"
        )

    def get_energy_stats(self) -> dict:
        """Energy statistics for logging. Read by FrequencyMaskingCallback."""
        profiles = self.energy_profiles or {}
        usable = [c for c in profiles if self.profile_for_class(c) is not None]
        return {
            'has_profiles': self.energy_profiles is not None,
            'n_classes': len(profiles),
            'n_usable': len(usable),
            'energy_weight': self.energy_weight,
        }


class ActivationGuidedFrequencyMasker(_BaseFrequencyMasker):
    """
    Phase 3: activation-guided frequency masking, per class.

    Slots are drawn with P(b) proportional to the encoder's activation magnitude
    for that class, blended with uniform by ``importance_weight``. Masking
    concentrates on the bands the encoder currently relies on for the class,
    pushing it to spread its frequency usage. This is the opposite direction to
    the energy strategy.

    Importance comes from ``FrequencyImportanceTracker``, which is updated in the
    training step where the class labels are known -- not from a label-blind
    forward hook, which could only produce one global profile shared by every
    class and every segment.

    Unlike energy, activation importance genuinely changes as the encoder learns,
    so it is the one signal tracked online. Note that dataloader workers fork
    once per epoch, so the importance a worker reads is a snapshot taken at epoch
    start rather than a live value.

    Classes the tracker has not seen yet fall back to unguided uniform slot
    masking, reported once per class.
    """

    def __init__(
        self,
        n_bands: int = 2,
        mask_prob: float = 0.5,
        max_band_width: int = 16,
        n_mels: int = 128,
        importance_weight: float = 1.0,
        importance_tracker: Optional['FrequencyImportanceTracker'] = None,
        seed: int = 1234
    ):
        """
        Args:
            importance_weight: Guidance strength. 0.0 reduces to uniform slots,
                              1.0 is pure importance guidance.
            importance_tracker: Provides get_importance(class_name). Attached
                               after construction by the datamodule.
        """
        super().__init__(n_bands, mask_prob, max_band_width, n_mels, seed)
        self.importance_weight = importance_weight
        self.importance_tracker = importance_tracker

        assert 0.0 <= importance_weight <= 1.0, (
            f"importance_weight must be in [0, 1], got {importance_weight}"
        )

    def slot_weights_for_class(self, class_name: Optional[str]) -> Optional[np.ndarray]:
        """
        P(b) for a class: high importance -> high weight.

        Returns None when no per-class importance is available.
        """
        if self.importance_tracker is None or class_name is None:
            return None

        scores = self.importance_tracker.get_importance(class_name)
        if scores is None:
            return None

        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1 or scores.shape[0] != self.n_mels:
            return None
        if not np.isfinite(scores).all() or scores.max() <= 0:
            return None

        slot_importance = self._slot_means(scores)
        if slot_importance.max() <= 0:
            return None

        weights = slot_importance / slot_importance.max()
        total = weights.sum()
        if total <= 0:
            return None
        return self._blend_with_uniform(weights / total, self.importance_weight)

    def _select_bands(self, spectrogram: np.ndarray,
                      class_name: Optional[str]) -> List[Tuple[int, int]]:
        weights = self.slot_weights_for_class(class_name)
        if weights is None:
            self._report_fallback(class_name, "no activation importance yet")
            return self._uniform_slot_bands()
        return self._bands_from_slots(self._sample_slots(weights))

    def __repr__(self):
        return (
            f"ActivationGuidedFrequencyMasker("
            f"n_bands={self.n_bands}, "
            f"mask_prob={self.mask_prob}, "
            f"max_band_width={self.max_band_width}, "
            f"n_mels={self.n_mels}, "
            f"importance_weight={self.importance_weight}, "
            f"has_tracker={self.importance_tracker is not None})"
        )

    def get_importance_stats(self) -> dict:
        """Importance statistics for logging. Read by FrequencyMaskingCallback."""
        stats = {
            'importance_weight': self.importance_weight,
            'has_tracker': self.importance_tracker is not None,
        }
        if self.importance_tracker is not None:
            stats['tracker_stats'] = self.importance_tracker.get_stats()
        else:
            stats['tracker_stats'] = None
        return stats


class AFRIRangeMasker(_BaseFrequencyMasker):
    """
    AFRI range-based frequency masking.

    Unlike the slot strategies this one has no lottery. It reads the frequency
    range AFRI identified for the class -- ``auto_minbin`` / ``auto_maxbin``
    from the cached profile -- and masks a fixed area strictly inside or
    strictly outside it:

        direct   eligible region = [lo, hi)
        inverse  eligible region = [0, lo) union [hi, n_mels)

    Direct trains the encoder to fall back on secondary spectral cues; inverse
    spares the discriminative band and approximates a learned denoising.

    Masked area is ``M = min(area_budget, W, n_mels - W)`` where
    ``W = hi - lo``. Taking the minimum over both the range and its complement
    makes M placeable in either mode, so **direct and inverse mask exactly the
    same number of bins for a given class** and form an exactly paired
    comparison. For most classes W is below the budget, so direct masks the
    whole range.

    The region is fixed and range-determined; the bands within it are resampled
    every call, which keeps augmentation diversity without letting masking
    stray outside the region. Union area is exactly M on every masked call.

    Classes whose profile is missing or has ``valid: False`` fall back to
    unguided uniform slot masking, logged once per class.
    """

    def __init__(
        self,
        n_bands: int = 2,
        mask_prob: float = 0.5,
        max_band_width: int = 16,
        n_mels: int = 128,
        afri_profiles: Optional[dict] = None,
        afri_mode: str = "direct",
        area_budget: int = AFRI_AREA_BUDGET,
        seed: int = 1234
    ):
        """
        Args:
            afri_profiles: Class name -> AFRI profile dict. Set after
                          build_meta() completes; None until then.
            afri_mode: "direct" masks inside the range, "inverse" outside it.
            area_budget: Upper bound on masked bins per call. The default is
                        calibrated to the slot strategies' union area.
        """
        super().__init__(n_bands, mask_prob, max_band_width, n_mels, seed)

        self.afri_profiles = afri_profiles
        self.afri_mode = afri_mode
        self.area_budget = area_budget

        assert afri_mode in ("direct", "inverse"), (
            f"afri_mode must be 'direct' or 'inverse', got {afri_mode}"
        )
        assert area_budget >= 1, (
            f"area_budget must be at least 1 bin, got {area_budget}"
        )

    def range_for_class(self, class_name: Optional[str]) -> Optional[Tuple[int, int]]:
        """
        The AFRI range for a class as a half-open (lo, hi) bin pair.

        Returns None -- meaning "fall back to unguided masking" -- when the
        profiles are not attached yet, the class is absent, the profile is a
        uniform fallback (``valid: False``), or the range is degenerate.
        """
        if self.afri_profiles is None or class_name is None:
            return None
        if class_name not in self.afri_profiles:
            return None

        profile = self.afri_profiles[class_name]
        if not profile.get('valid', False):
            return None
        if 'auto_minbin' not in profile or 'auto_maxbin' not in profile:
            return None

        lo = int(np.clip(int(profile['auto_minbin']), 0, self.n_mels))
        hi = int(np.clip(int(profile['auto_maxbin']), 0, self.n_mels))
        if hi <= lo:
            return None
        return lo, hi

    def masked_area_for_class(self, class_name: Optional[str]) -> Optional[int]:
        """
        Masked bins M for a class, identical in both modes. None if unguided.
        """
        rng_pair = self.range_for_class(class_name)
        if rng_pair is None:
            return None
        lo, hi = rng_pair
        width = hi - lo
        area = min(self.area_budget, width, self.n_mels - width)
        return area if area >= 1 else None

    def _eligible_intervals(self, lo: int, hi: int) -> List[Tuple[int, int]]:
        """Half-open intervals the mask may occupy, for the active mode."""
        if self.afri_mode == "direct":
            intervals = [(lo, hi)]
        else:
            intervals = [(0, lo), (hi, self.n_mels)]
        return [(a, b) for a, b in intervals if b > a]

    def _allocate_area(
        self, area: int, intervals: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Split the masked area across eligible intervals, respecting capacity.

        Allocating area to intervals *before* choosing band widths is what makes
        the union area exact: each interval receives no more than it can hold, so
        the per-interval layout below can never fail to place a band. Doing it
        the other way round (widths first, then packing) can strand a wide band
        that fits in neither interval and silently under-mask.
        """
        caps = np.array([b - a for a, b in intervals], dtype=int)
        area = min(area, int(caps.sum()))

        ideal = caps / caps.sum() * area
        alloc = np.minimum(np.floor(ideal).astype(int), caps)

        # Hand out the rounding remainder, largest fractional part first
        order = list(np.argsort(-(ideal - np.floor(ideal))))
        remaining = area - int(alloc.sum())
        while remaining > 0:
            progressed = False
            for j in order:
                if remaining == 0:
                    break
                if alloc[j] < caps[j]:
                    alloc[j] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        return alloc.tolist()

    def _split_area(self, area: int, max_width: int, k_target: int) -> List[int]:
        """
        Split `area` bins into positive band widths, each at most `max_width`.

        Aims for `k_target` bands, raising the count only when the width cap
        forces it. Falls back to an even split if the uniform random composition
        violates the cap.
        """
        k = max(1, min(k_target, area))
        k = max(k, int(math.ceil(area / max_width)))
        k = min(k, area)

        if k == 1:
            return [area]

        # Uniform random composition of `area` into k positive parts
        for _ in range(10):
            cuts = np.sort(
                self.rng.choice(np.arange(1, area), size=k - 1, replace=False)
            )
            widths = np.diff(np.concatenate(([0], cuts, [area]))).astype(int)
            if widths.max() <= max_width:
                return widths.tolist()

        # Even split: valid whenever area <= k * max_width, which the k above
        # guarantees.
        base, extra = divmod(area, k)
        return [base + 1] * extra + [base] * (k - extra)

    def _place_in_interval(
        self, start: int, end: int, widths: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Lay `widths` out non-overlapping inside [start, end) at random offsets.

        Distributes the slack over the len(widths)+1 gaps as a uniform random
        composition, so every valid non-overlapping arrangement is equally
        likely.
        """
        if not widths:
            return []

        length = end - start
        used = int(np.sum(widths))
        free = length - used
        m = len(widths)

        if free < 0:
            return []

        if m > 0 and free > 0:
            idx = np.sort(self.rng.choice(free + m, size=m, replace=False))
            gaps = np.diff(np.concatenate(([-1], idx, [free + m]))) - 1
        else:
            gaps = np.zeros(m + 1, dtype=int)

        bands = []
        cursor = start
        for i, width in enumerate(widths):
            cursor += int(gaps[i])
            bands.append((cursor, cursor + width))
            cursor += width
        return bands

    def _select_bands(self, spectrogram: np.ndarray,
                      class_name: Optional[str]) -> List[Tuple[int, int]]:
        rng_pair = self.range_for_class(class_name)
        area = self.masked_area_for_class(class_name)
        if rng_pair is None or area is None:
            self._report_fallback(class_name, "no usable AFRI range")
            return self._uniform_slot_bands()

        lo, hi = rng_pair
        intervals = self._eligible_intervals(lo, hi)
        if not intervals:
            self._report_fallback(class_name, "no eligible region")
            return self._uniform_slot_bands()

        allocation = self._allocate_area(area, intervals)

        bands = []
        for (start, end), interval_area in zip(intervals, allocation):
            if interval_area <= 0:
                continue
            # Spread n_bands over the intervals in proportion to their share
            k_target = max(1, int(round(self.n_bands * interval_area / area)))
            widths = self._split_area(
                interval_area, min(self.max_band_width, end - start), k_target
            )
            bands.extend(self._place_in_interval(start, end, widths))
        return bands

    def __repr__(self):
        n_profiles = len(self.afri_profiles) if self.afri_profiles is not None else 0
        return (
            f"AFRIRangeMasker("
            f"n_bands={self.n_bands}, "
            f"mask_prob={self.mask_prob}, "
            f"max_band_width={self.max_band_width}, "
            f"n_mels={self.n_mels}, "
            f"afri_mode={self.afri_mode}, "
            f"area_budget={self.area_budget}, "
            f"n_profiles={n_profiles})"
        )

    def get_afri_stats(self) -> dict:
        """AFRI coverage statistics for logging."""
        if self.afri_profiles is None:
            return {'has_profiles': False, 'n_classes': 0, 'n_valid': 0,
                    'afri_mode': self.afri_mode, 'area_budget': self.area_budget,
                    'mean_masked_area': 0.0}

        valid = [c for c in self.afri_profiles
                 if self.range_for_class(c) is not None]
        areas = [self.masked_area_for_class(c) for c in valid]
        areas = [a for a in areas if a is not None]
        return {
            'has_profiles': True,
            'n_classes': len(self.afri_profiles),
            'n_valid': len(valid),
            'afri_mode': self.afri_mode,
            'area_budget': self.area_budget,
            'mean_masked_area': float(np.mean(areas)) if areas else 0.0,
        }


#: freq_mask_mode -> strategy class; shared by the factory and config checks.
MASKER_FOR_MODE = {
    'random': RandomFrequencyMasker,
    'slot_uniform': UniformSlotFrequencyMasker,
    'energy': EnergyBasedFrequencyMasker,
    'activation': ActivationGuidedFrequencyMasker,
    'afri': AFRIRangeMasker,
}

#: Per-class strategies, and the attribute the dataset attaches their profiles
#: to after build_meta(). None means the strategy needs no attachment.
PROFILE_ATTRIBUTE_FOR_MODE = {
    'random': None,
    'slot_uniform': None,
    'energy': 'energy_profiles',
    'activation': 'importance_tracker',
    'afri': 'afri_profiles',
}


def build_frequency_masker(train_param, n_mels: int,
                           importance_tracker=None) -> Optional[_BaseFrequencyMasker]:
    """
    Construct the masker named by ``train_param.freq_mask_mode``.

    Kept out of the dataset so the mode -> class mapping can be exercised without
    audio, a model or a GPU -- see tools/sanity_check_configs.py. The dataset
    calls this and then attaches whatever profiles the strategy needs.

    Args:
        train_param: Mapping supporting ``.get`` (a dict or an OmegaConf node).
        n_mels: Number of mel bins, from the feature config.
        importance_tracker: Only used by the 'activation' strategy.

    Returns:
        The masker, or None when ``use_freq_masking`` is false.

    Raises:
        ValueError: if ``freq_mask_mode`` is not a known mode. Failing loudly
                   matters here: a typo silently falling back to random masking
                   would produce a plausible-looking run of the wrong thing.
    """
    if not train_param.get('use_freq_masking', False):
        return None

    mode = train_param.get('freq_mask_mode', 'random')
    if mode not in MASKER_FOR_MODE:
        raise ValueError(
            f"Unknown freq_mask_mode {mode!r}. "
            f"Expected one of: {sorted(MASKER_FOR_MODE)}"
        )

    common = dict(
        n_bands=train_param.get('freq_mask_n_bands', 2),
        mask_prob=train_param.get('freq_mask_prob', 0.5),
        max_band_width=train_param.get('freq_mask_max_width', 16),
        n_mels=n_mels,
        seed=train_param.get('freq_mask_seed', 1234),
    )

    if mode == 'energy':
        return EnergyBasedFrequencyMasker(
            **common,
            energy_weight=train_param.get('freq_mask_energy_weight', 1.0),
            energy_profiles=None,          # attached after build_meta()
        )
    if mode == 'activation':
        return ActivationGuidedFrequencyMasker(
            **common,
            importance_weight=train_param.get('freq_mask_importance_weight', 1.0),
            importance_tracker=importance_tracker,
        )
    if mode == 'afri':
        return AFRIRangeMasker(
            **common,
            afri_profiles=None,            # attached after build_meta()
            afri_mode=train_param.get(
                'afri_mode', train_param.get('afri_mask_mode', 'direct')),
            area_budget=train_param.get(
                'freq_mask_afri_area_budget', AFRI_AREA_BUDGET),
        )
    return MASKER_FOR_MODE[mode](**common)


def describe_masker(masker: Optional[_BaseFrequencyMasker]) -> str:
    """One-line summary for startup logs."""
    return "frequency masking disabled" if masker is None else repr(masker)
