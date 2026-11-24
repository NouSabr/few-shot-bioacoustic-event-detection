"""
Per-class Frequency Importance Tracker for activation-guided frequency masking.

Tracks, for each class, which mel bins the encoder's activations concentrate on,
and exposes that as a masking signal for ActivationGuidedFrequencyMasker.

Why per class, and why not from the hook alone
----------------------------------------------
This used to keep a single global importance vector, updated directly inside a
forward hook on ``model.layer2``. A hook sees activations but not labels, so it
could only produce one profile shared by every class and every segment --
measurement confirmed Phase 3's slot distribution was then completely
independent of the input.

The hook now only *captures* the activations; attribution happens in the
training step, where the episode's class names are known, so each class
accumulates its own profile.

Activation importance is the one masking signal that genuinely has to be tracked
online, because it changes as the encoder learns. Spectrogram energy (Phase 2)
and AFRI ranges are properties of the data and are precomputed instead.

Profiles are kept as CPU numpy arrays rather than CUDA buffers. At
``num_workers=0`` the masker reads them live, in the same process, on the very
next ``__getitem__``. If the loader is ever given workers, the profiles survive
the fork and are read as an epoch-start snapshot; a CUDA buffer would not be
readable there at all.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence


class FrequencyImportanceTracker(nn.Module):
    """
    Track per-class frequency importance from encoder activations.

    Importance for a class is the mean absolute activation over the batch rows
    belonging to that class, averaged across channels and time, resampled to mel
    resolution, max-normalised, and smoothed by an exponential moving average.

    EMA convention is momentum: ``ema = alpha * old + (1 - alpha) * new``, so a
    HIGHER ``ema_alpha`` is more stable and slower to shift.

    Usage::

        tracker.attach(model.net)                 # hook captures layer2 output
        ...                                       # inside the training step:
        tracker.update_from_last(class_names)     # attribute to those classes
        tracker.get_importance("OVEN")            # -> (n_mels,) or None
    """

    def __init__(self, n_mels: int = 128, ema_alpha: float = 0.9,
                 device: str = 'cpu'):
        """
        Args:
            n_mels: Number of mel frequency bins in the input spectrogram.
            ema_alpha: Momentum for the per-class EMA (0.9 = slow and stable).
            device: Retained for backwards compatibility; profiles are numpy on
                   CPU regardless, so they stay readable by dataloader workers.
        """
        super().__init__()
        self.n_mels = n_mels
        self.ema_alpha = ema_alpha
        self.device = device

        self._importance: Dict[str, np.ndarray] = {}
        self._update_counts: Dict[str, int] = {}
        self.total_updates = 0

        # Set by the forward hook, consumed by update_from_last()
        self._last_activations: Optional[torch.Tensor] = None
        self.hook_handle = None

        assert 0.0 <= ema_alpha <= 1.0, (
            f"ema_alpha must be in [0, 1], got {ema_alpha}"
        )

        print(f"✓ FrequencyImportanceTracker initialized (per class):")
        print(f"  - n_mels: {n_mels}")
        print(f"  - ema_alpha: {ema_alpha} (momentum: alpha*old + (1-alpha)*new)")

    def attach(self, model: nn.Module) -> None:
        """
        Hook ``model.layer2`` to capture its output on every forward pass.

        Layer 2 is used because it has 128 channels and still preserves a
        frequency axis, and it sits before the CIFE module. The hook only stores
        the tensor -- it cannot attribute it, having no access to labels.
        """
        if not hasattr(model, 'layer2'):
            raise AttributeError("Model must have 'layer2' attribute for hooking")

        def hook_fn(module, inputs, output):
            self._last_activations = output

        self.hook_handle = model.layer2.register_forward_hook(hook_fn)
        print(f"✓ Activation capture hook registered on model.layer2")

    def remove_hook(self) -> None:
        """Detach the capture hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print("✓ Hook removed")

    def update_from_last(self, class_names: Sequence[str]) -> None:
        """
        Attribute the most recently captured activations to `class_names`.

        Call from the training step, where the labels are known. Rows beyond
        ``len(class_names)`` are ignored, so passing the episode's class list
        attributes the positive half of a positive/negative concatenated batch
        and leaves the negatives out -- importance should describe where the
        encoder attends for a class's actual events.
        """
        activations = self._last_activations
        self._last_activations = None
        if activations is None:
            return
        self.update(activations, class_names)

    @torch.no_grad()
    def update(self, activations: torch.Tensor,
               class_names: Sequence[str]) -> None:
        """
        Fold one batch of encoder activations into the per-class profiles.

        Args:
            activations: Encoder feature map, (batch, channels, time, freq).
            class_names: Class name per row. Only the first
                        ``min(rows, len(class_names))`` rows are used.

        Only the rows belonging to a class contribute to that class's profile,
        which is what makes the signal per class rather than global.
        """
        if activations is None or activations.ndim != 4:
            return
        if class_names is None or len(class_names) == 0:
            return

        n_rows = min(activations.shape[0], len(class_names))
        if n_rows == 0:
            return

        # Mean |activation| over channels and time -> (rows, freq)
        per_row = activations[:n_rows].abs().mean(dim=(1, 2)).float().cpu().numpy()

        by_class: Dict[str, List[int]] = {}
        for i in range(n_rows):
            by_class.setdefault(str(class_names[i]), []).append(i)

        for class_name, rows in by_class.items():
            self._fold_in(class_name, per_row[rows].mean(axis=0))

        self.total_updates += 1

    def _fold_in(self, class_name: str, importance: np.ndarray) -> None:
        """Resample to mel resolution, normalise, and EMA into the class slot."""
        importance = np.asarray(importance, dtype=np.float64)

        # Layer-2 output is downsampled along frequency, so stretch it back to
        # mel resolution before it is used to weight mel slots.
        if importance.shape[0] != self.n_mels:
            src = np.linspace(0.0, 1.0, num=importance.shape[0])
            dst = np.linspace(0.0, 1.0, num=self.n_mels)
            importance = np.interp(dst, src, importance)

        if not np.isfinite(importance).all():
            return
        peak = importance.max()
        if peak > 0:
            importance = importance / peak

        previous = self._importance.get(class_name)
        if previous is None:
            self._importance[class_name] = importance
        else:
            self._importance[class_name] = (
                self.ema_alpha * previous + (1.0 - self.ema_alpha) * importance
            )
        self._update_counts[class_name] = self._update_counts.get(class_name, 0) + 1

    def get_importance(self, class_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Per-bin importance for a class, shape (n_mels,), values in [0, 1].

        Returns None when the class has not been seen yet, which tells the masker
        to fall back to unguided masking rather than invent a signal. Called by
        ActivationGuidedFrequencyMasker.
        """
        if class_name is None:
            return None
        profile = self._importance.get(str(class_name))
        return None if profile is None else profile.copy()

    def n_classes_tracked(self) -> int:
        return len(self._importance)

    def reset(self) -> None:
        """Drop all profiles."""
        self._importance.clear()
        self._update_counts.clear()
        self.total_updates = 0
        print("✓ Importance tracker reset")

    def get_stats(self) -> dict:
        """Aggregate statistics for logging."""
        if not self._importance:
            return {
                'n_classes_tracked': 0,
                'total_updates': self.total_updates,
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'mean_spread': 0.0,
            }
        stacked = np.stack(list(self._importance.values()))
        # Spread within a class is what actually steers masking: a flat profile
        # cannot express guidance however many classes are tracked.
        spread = float(np.mean(stacked.max(axis=1) - stacked.min(axis=1)))
        return {
            'n_classes_tracked': len(self._importance),
            'total_updates': self.total_updates,
            'mean': float(stacked.mean()),
            'std': float(stacked.std()),
            'min': float(stacked.min()),
            'max': float(stacked.max()),
            'mean_spread': spread,
        }

    def __repr__(self):
        return (f"FrequencyImportanceTracker(n_mels={self.n_mels}, "
                f"ema_alpha={self.ema_alpha}, "
                f"classes={len(self._importance)}, "
                f"updates={self.total_updates})")
