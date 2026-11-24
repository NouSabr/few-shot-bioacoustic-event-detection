"""
PyTorch Lightning callback for frequency-masking logging.

Logs the active masking strategy and its parameters once at train start, then
periodically logs whatever runtime statistics that strategy exposes.

The five strategies (see docs/dynamic_freq_masking/01_OVERVIEW.md):

    random        Phase 1  SpecAugment; no parameters beyond the common ones
    slot_uniform  control  slot placement with P(b) uniform
    energy        Phase 2  per-class precomputed energy profiles
    activation    Phase 3  per-class activation importance, tracked online
    afri                   per-class frequency range, direct or inverse

Statistics are read from the masker by duck typing (``get_energy_stats``,
``get_importance_stats``, ``get_afri_stats``), so adding a strategy does not
require changing how it is read here.

Note this reads the masker held by the *dataset*, which at num_workers=0 is the
same object that does the masking. With workers > 0 it is the parent's copy, so
the per-batch statistics describe the parent rather than any worker.
"""

import logging
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.datamodules.components.frequency_masker import AFRI_AREA_BUDGET

logger = logging.getLogger(__name__)

# Metric name prefix, kept in one place so the keys cannot drift apart
P = "freq_mask"


class FrequencyMaskingCallback(Callback):
    """
    Log frequency-masking configuration and runtime statistics.

    Args:
        log_freq: Log runtime statistics every N training batches. 0 disables
                 the periodic logging; the startup summary is always emitted.

    Example:
        >>> callback = FrequencyMaskingCallback(log_freq=100)
    """

    def __init__(self, log_freq: int = 1000):
        super().__init__()
        self.log_freq = log_freq
        self._logged_startup = False

    # ------------------------------------------------------------------
    # Startup summary
    # ------------------------------------------------------------------

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log the masking configuration once, to the log and to the logger."""
        if self._logged_startup:
            return
        self._logged_startup = True

        train_params = pl_module.hparams.train

        if not train_params.get('use_freq_masking', False):
            logger.info("=" * 70)
            logger.info("Frequency Masking DISABLED")
            logger.info("=" * 70)
            self._log_hparams(trainer, {f'{P}/enabled': False})
            return

        mode = train_params.get('freq_mask_mode', 'random')

        # Common to every strategy
        params = {
            'n_bands': train_params.get('freq_mask_n_bands', 2),
            'prob': train_params.get('freq_mask_prob', 0.5),
            'max_width': train_params.get('freq_mask_max_width', 16),
            'seed': train_params.get('freq_mask_seed', 1234),
        }
        titles = {
            'random': "Phase 1: Random Frequency Masking (SpecAugment)",
            'slot_uniform': "Control: Uniform-Slot Frequency Masking",
            'energy': "Phase 2: Energy-Guided Frequency Masking (per class)",
            'activation': "Phase 3: Activation-Guided Frequency Masking (per class)",
            'afri': "AFRI Range Frequency Masking (per class)",
        }

        # Strategy-specific
        extra = {}
        if mode == 'energy':
            extra['energy_weight'] = train_params.get('freq_mask_energy_weight', 1.0)
        elif mode == 'activation':
            extra['importance_weight'] = train_params.get(
                'freq_mask_importance_weight', 1.0)
            extra['importance_ema_alpha'] = train_params.get(
                'freq_mask_importance_ema_alpha', 0.9)
        elif mode == 'afri':
            extra['afri_mode'] = train_params.get(
                'afri_mode', train_params.get('afri_mask_mode', 'direct'))
            extra['afri_area_budget'] = train_params.get(
                'freq_mask_afri_area_budget', AFRI_AREA_BUDGET)
            extra['use_afri'] = train_params.get('use_afri', False)

        logger.info("=" * 70)
        logger.info(titles.get(mode, f"Frequency Masking ENABLED (mode: {mode})"))
        logger.info(f"  - mode: {mode}")
        for key, value in params.items():
            logger.info(f"  - {key}: {value}")
        for key, value in extra.items():
            logger.info(f"  - {key}: {value}")
        if mode == 'afri' and not extra.get('use_afri', False):
            logger.warning("  ⚠️  use_afri is False, so no profiles are computed "
                           "and every class falls back to unguided slot masking")
        logger.info("=" * 70)

        payload = {f'{P}/enabled': True, f'{P}/mode': mode}
        payload.update({f'{P}/{k}': v for k, v in params.items()})
        payload.update({f'{P}/{k}': v for k, v in extra.items()})
        self._log_hparams(trainer, payload)

    @staticmethod
    def _log_hparams(trainer: pl.Trainer, payload: dict) -> None:
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(payload)

    # ------------------------------------------------------------------
    # Periodic statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _get_masker(trainer: pl.Trainer):
        """The dataset's masker, or None if it cannot be reached."""
        loader = getattr(trainer, 'train_dataloader', None)
        dataset = getattr(loader, 'dataset', None)
        return getattr(dataset, 'freq_masker', None)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        *args,
    ) -> None:
        """Log runtime masking statistics every `log_freq` batches."""
        if self.log_freq <= 0 or batch_idx % self.log_freq != 0:
            return

        train_params = pl_module.hparams.train
        if not train_params.get('use_freq_masking', False):
            return

        mode = train_params.get('freq_mask_mode', 'random')
        pl_module.log(f"{P}/enabled", 1.0, prog_bar=False)

        # Statistics are best-effort: never fail a training step over logging.
        try:
            masker = self._get_masker(trainer)
            if masker is None:
                return

            if mode == 'energy' and hasattr(masker, 'get_energy_stats'):
                stats = masker.get_energy_stats()
                pl_module.log(f"{P}/energy_classes_usable",
                              float(stats['n_usable']), prog_bar=False)
                pl_module.log(f"{P}/energy_classes_fallback",
                              float(stats['n_classes'] - stats['n_usable']),
                              prog_bar=False)

            elif mode == 'activation' and hasattr(masker, 'get_importance_stats'):
                stats = masker.get_importance_stats()
                tracker_stats = stats.get('tracker_stats')
                if tracker_stats:
                    pl_module.log(f"{P}/classes_tracked",
                                  float(tracker_stats['n_classes_tracked']),
                                  prog_bar=False)
                    pl_module.log(f"{P}/tracker_update_count",
                                  float(tracker_stats['total_updates']),
                                  prog_bar=False)
                    # Spread is what decides whether Phase 3 steers anything: a
                    # flat profile cannot, however many classes are tracked.
                    pl_module.log(f"{P}/importance_spread",
                                  float(tracker_stats['mean_spread']), prog_bar=False)
                    pl_module.log(f"{P}/importance_mean",
                                  float(tracker_stats['mean']), prog_bar=False)
                    pl_module.log(f"{P}/importance_std",
                                  float(tracker_stats['std']), prog_bar=False)

            elif mode == 'afri' and hasattr(masker, 'get_afri_stats'):
                stats = masker.get_afri_stats()
                pl_module.log(f"{P}/afri_n_valid",
                              float(stats['n_valid']), prog_bar=False)
                pl_module.log(f"{P}/afri_n_fallback",
                              float(stats['n_classes'] - stats['n_valid']),
                              prog_bar=False)
                pl_module.log(f"{P}/afri_mean_masked_area",
                              float(stats['mean_masked_area']), prog_bar=False)
        except Exception:
            pass

    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer: pl.Trainer,
                           pl_module: pl.LightningModule) -> None:
        """Confirm once per epoch that masking is still active."""
        train_params = pl_module.hparams.train
        if train_params.get('use_freq_masking', False):
            mode = train_params.get('freq_mask_mode', 'random')
            logger.debug(f"Epoch {trainer.current_epoch}: "
                         f"frequency masking active (mode: {mode})")
